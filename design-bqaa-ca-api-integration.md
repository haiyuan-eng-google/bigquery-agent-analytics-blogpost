# Design: BQ Agent Analytics for 1P Conversational Analytics API Users

Date: 2026-04-03
Author: haiyuancao
Status: DRAFT
Stakeholder: Senior Director (BigQuery, focus on CA API adoption)

## 1. Problem Statement

The BigQuery Agent Analytics (BQ AA) plugin currently captures 160M events/month, all from **3P customers** running ADK-based agents. There is zero 1P adoption. The senior director wants to enable BQ AA for **Conversational Analytics (CA) API** users, Google's own 1P product for natural language data queries.

The CA API (`geminidataanalytics.googleapis.com`) serves users across BigQuery, Looker, Looker Studio, AlloyDB, Spanner, and Cloud SQL. Today, the CA API stores conversation history internally but provides no structured analytics layer for operators to understand how their data agents perform, what questions users ask, where the agent fails, or how query quality evolves over time.

Enabling BQ AA for CA API users means: every `chat` call to the CA API can optionally write structured analytics events into the customer's own BigQuery dataset, using the same `agent_events` table schema foundation as 3P ADK users. CA-specific event types (CA_QUESTION_RECEIVED, CA_SQL_GENERATED, etc.) extend the existing schema with new event types that require view-layer updates to be queryable through per-event-type views.

## 2. Architecture: How the CA API Works Today

```
┌──────────────┐     ┌─────────────────────────────┐     ┌─────────────────┐
│  End User     │────▶│  CA API                      │────▶│  Data Source     │
│  (or Agent)   │     │  (geminidataanalytics.        │     │  (BigQuery,      │
│               │◀────│   googleapis.com)             │◀────│   Looker, etc.)  │
└──────────────┘     └─────────────────────────────┘     └─────────────────┘
     ▲                        │
     │                        │ Conversation stored
     │                        │ internally by CA API
     │                        ▼
     │                  ┌─────────────┐
     │                  │ CA Internal  │
     │                  │ Storage      │
     │                  │ (messages,   │
     │                  │  history)    │
     │                  └─────────────┘
     │
     └── User gets: answer text, generated SQL, data results
         User does NOT get: structured analytics, latency metrics,
         tool usage breakdown, error patterns, query quality trends
```

### Key Auth Facts

| Layer | Identity | Mechanism |
|-------|----------|-----------|
| CA API access | User or Service Account | OAuth2 bearer token (`gcloud auth print-identity-token`) |
| Data source query execution | **User's own credentials** | CA API queries BQ/Looker/DB using the caller's identity, not a service account |
| CA IAM roles | Per-agent or per-project | `dataAgentCreator`, `dataAgentOwner`, `dataAgentUser`, etc. |
| Data source permissions | User must have BQ/Looker/DB access independently | CA IAM roles do NOT grant data source access |

**Critical design implication:** The CA API executes queries using the **user's credentials**. However, writing analytics events creates a **new durable disclosure surface** that is distinct from the user's transient query access. The user sees query results ephemerally during a chat session, but analytics persists question text, generated SQL, and potentially result summaries into a BigQuery dataset with a different access policy (readable by operators, admins, and anyone with `bigquery.dataViewer` on the analytics dataset). This means:

- A user's questions become visible to dataset operators who may not have been party to the original conversation
- Query patterns, SQL, and error messages are durably stored even after the CA conversation is deleted
- The analytics dataset's IAM policy governs who sees this data, NOT the original data source's IAM policy

**This is a genuine security and privacy surface.** The consent model in Section 5 must treat analytics as a new disclosure, not as a derivative of existing access. The three-tier consent model (admin opt-in, content control, user visibility) is designed to address this, but the default privacy mode must be chosen with this threat model in mind.

## 3. Proposed Architecture: CA API + BQ AA

```
┌──────────────┐     ┌─────────────────────────────┐     ┌─────────────────┐
│  End User     │────▶│  CA API                      │────▶│  Data Source     │
│  (or Agent)   │     │  + BQ AA Plugin Integration  │     │  (BigQuery,      │
│               │◀────│                               │◀────│   Looker, etc.)  │
└──────────────┘     └──────────┬──────────────────┘     └─────────────────┘
                                │
                    ┌───────────┴───────────┐
                    │ If analytics enabled: │
                    │ Write structured event│
                    │ to customer's BQ      │
                    ▼                       │
              ┌─────────────────┐          │
              │ Customer's BQ    │          │
              │ Dataset          │          │
              │ (agent_events    │          │
              │  table)          │          │
              └─────────────────┘          │
                                           │
              Events captured:             │
              - CA_QUESTION_RECEIVED       │
              - CA_SQL_GENERATED           │
              - CA_QUERY_EXECUTED          │
              - CA_RESPONSE_DELIVERED      │
              - CA_ERROR                   │
              └────────────────────────────┘
```

## 4. Auth Path Design

### 4.1 Who Writes the Analytics Events?

**Option A (Recommended): CA API Service Identity writes events**

The CA API's internal service identity writes analytics events to the customer's BQ dataset. The customer grants `bigquery.dataEditor` on the analytics dataset to the CA API's service account.

Pros:
- Customer doesn't need to manage separate credentials for analytics
- Consistent with how other Google Cloud services write audit data (e.g., Cloud Audit Logs)
- Analytics writes don't consume the user's BQ quota

Cons:
- Requires the customer to grant an IAM binding to a Google-managed service account
- Customer must trust Google's service identity with write access to their analytics dataset

**Option B: User's Own Credentials write events**

The user's credentials (already used for data source queries) also write analytics events.

Pros:
- No additional IAM grants needed
- User already has BQ access

Cons:
- Analytics writes consume the user's BQ slot quota
- If the user only has `bigquery.dataViewer` (read-only), analytics writes fail silently
- Mixing query workload and analytics workload under one identity makes cost attribution hard

**Recommendation: Option A.** This follows the pattern of Cloud Audit Logs and other Google Cloud observability features where a service identity writes telemetry, and the customer controls access to the output dataset.

### 4.2 Consolidated `analyticsConfig` Schema

```json
{
  "analyticsConfig": {
    "enabled": true,
    "bigqueryDataset": "projects/P/datasets/D",
    "tableId": "agent_events",
    "privacyMode": "metadata_only",
    "eventAllowlist": ["CA_QUESTION_RECEIVED", "CA_SQL_GENERATED", "CA_QUERY_EXECUTED", "CA_RESPONSE_DELIVERED", "CA_ERROR"],
    "eventDenylist": [],
    "allowUserOptOut": false
  }
}
```

All fields except `enabled` and `bigqueryDataset` have defaults. The `privacyMode` field accepts `"full"`, `"anonymized"`, or `"metadata_only"` (default: `"metadata_only"`). See Section 5.3 for what each mode logs.

**JSON convention:** All external API fields use camelCase (Google Cloud API standard).

### 4.3 Auth Flow Step by Step

```
1. Customer enables BQ AA for their CA data agent
   └── Via CA API: PATCH /dataAgents/{id} with analytics config
       {
         "analyticsConfig": {
           "enabled": true,
           "bigqueryDataset": "projects/P/datasets/D",
           "tableId": "agent_events"
         }
       }

2. Customer grants BQ write access to CA service identity
   ├── CA API service account: service-{PROJECT_NUM}@gcp-sa-geminidataanalytics.iam.gserviceaccount.com
   │
   ├── Required IAM binding (at dataset level, NOT project level):
   │   bq update --dataset \
   │     --service_account=service-{NUM}@gcp-sa-geminidataanalytics.iam.gserviceaccount.com \
   │     --role=roles/bigquery.dataEditor \
   │     PROJECT:DATASET
   │
   └── NOTE: roles/bigquery.jobUser is NOT required if using the BigQuery
       Storage Write API (which writes via AppendRows, not jobs.insert).
       The Storage Write API only needs bigquery.tables.updateData
       (included in roles/bigquery.dataEditor). Verify this before
       requesting project-level jobUser grants from customers, as
       project-level IAM is a harder sell for enterprise admins.
       If the implementation falls back to streaming inserts or load
       jobs for any reason, jobUser would then be required.

3. User makes a CA API chat call (normal flow)
   POST /v1beta/projects/P/locations/L:chat
   Authorization: Bearer {user_token}
   Body: { "messages": [{"userMessage": {"text": "top 10 customers by revenue"}}], ... }

4. CA API processes the request (normal flow)
   ├── LLM generates SQL
   ├── SQL validated (SELECT only, no DDL/DML)
   ├── Query executed using USER's credentials against data source
   └── Results returned to user

5. CA API writes analytics event (NEW — using service identity)
   ├── Async, non-blocking fire-and-forget (same Storage Write API
   │   transport as the ADK plugin, but WITHOUT synchronous flush;
   │   see Section 4.4)
   ├── Event schema matches existing agent_events table:
   │   {
   │     "timestamp": "2026-04-03T...",
   │     "event_type": "CA_QUESTION_RECEIVED",
   │     "agent": "my-data-agent",
   │     "session_id": "conversation-id",
   │     "user_id": "<per privacyMode>",
   │     "content": {"question": "<per privacyMode>"},
   │     "attributes": {
   │       "source": "conversational_analytics",
   │       "agent_type": "data_agent",
   │       "data_source": "bigquery",
   │       "schema_version": "1.0"
   │     }
   │   }
   ├── Subsequent events: CA_SQL_GENERATED, CA_QUERY_EXECUTED (with latency),
   │   CA_RESPONSE_DELIVERED
   └── Fire-and-forget write (see 4.4 for latency tradeoff)
```

### 4.4 Latency Tradeoff: Non-Blocking vs Immediate Queryability

The ADK plugin flushes synchronously at the end of each invocation (commit 9579bea) because ADK agents run in serverless environments where the process may be killed after the response. This adds latency to the agent's response but guarantees events are written.

**The CA API has a different latency profile.** It serves interactive product traffic where every millisecond of user-visible latency matters. A synchronous flush with retries (potentially 1s+2s+4s = 7s worst case) is unacceptable on the hot path.

**Decision: CA analytics writes are strictly non-blocking (fire-and-forget).**

- Events are queued in-process and written asynchronously by a background writer.
- The CA API response is returned immediately, without waiting for analytics writes.
- Events may be delayed by up to `batch_flush_interval` (default 1s) before appearing in BigQuery.
- **If the CA API process is long-lived** (standard server, not serverless): the background writer runs continuously. Events are reliably written within seconds.
- **If the CA API runs in a serverless/ephemeral context:** a periodic background flush (every 500ms) ensures most events are written. Some events may be lost on abrupt termination. This is the explicit tradeoff: we choose zero user-visible latency impact over guaranteed delivery.
- Events are NOT guaranteed to be queryable at the time the chat response is returned. This is different from the ADK plugin's guarantee.

### 4.5 Write Failure Behavior

Analytics writes must NEVER block or fail the chat response. If a write fails:

1. **Retry in background:** Up to 3 retries with exponential backoff (1s, 2s, 4s), all in the background writer thread. Zero impact on response latency.
2. **Log to Cloud Logging:** On final failure, emit a structured log entry to Cloud Logging (`resource.type="geminidataanalytics"`) with the error, event type, and agent ID. This is visible in the customer's Logs Explorer.
3. **Drop the event.** Do not queue indefinitely. Analytics is best-effort.
4. **Do NOT surface in `analyticsDisclosure`.** The disclosure field reflects the configuration, not the write health.

### 4.6 Table Lifecycle

1. **Auto-creation:** On first analytics write, the CA service identity creates the `agent_events` table using the same DDL as the ADK plugin (partitioned by timestamp, clustered by event_type/agent/user_id). Requires `bigquery.tables.create` permission on the dataset (included in `roles/bigquery.dataEditor`).
2. **Schema versioning:** Each event includes `attributes.schema_version` (e.g., `"1.0"`). When the CA integration adds new event types or fields in future releases, the schema version increments. The existing ADK plugin's `auto_schema_upgrade` pattern (additive-only column changes, version label on the table) applies here.
3. **Shared table:** CA events and ADK events coexist in the same `agent_events` table. The `attributes.source` field (`"conversational_analytics"` vs `"adk"`) distinguishes them. This is intentional: operators get a single analytics surface for all agent types.

## 5. User Consent Path

### 5.1 Why Consent Matters

The CA API runs using the **user's credentials** against their data. Analytics creates a **durable disclosure surface** that persists:
- What questions the user asked
- What SQL was generated
- What data sources were queried
- Performance metrics (latency, errors)

This data is written to a BigQuery dataset with its own IAM policy, potentially readable by operators and admins who were not party to the original conversation. In regulated environments, this persistent record may be subject to data governance requirements (GDPR, SOC2, internal compliance).

### 5.2 Three-Tier Consent Model

```
┌─────────────────────────────────────────────────────┐
│ Tier 1: Admin Enables (Required)                    │
│                                                     │
│ The Data Agent Owner (admin) explicitly enables      │
│ analytics via the agent config. This is the gate.   │
│ Analytics are OFF by default. No events are written  │
│ until an admin turns this on.                       │
│                                                     │
│ Who: roles/geminidataanalytics.dataAgentOwner       │
│      (or dataAgentCreator, who can also modify      │
│      agent configs)                                 │
│ How: PATCH /dataAgents/{id} with analyticsConfig    │
│ Scope: Per-agent (not project-wide)                 │
└─────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────┐
│ Tier 2: Content Control (Configurable)              │
│                                                     │
│ Admin configures what gets logged:                  │
│ - eventAllowlist / eventDenylist                    │
│ - user ID anonymization (hash, redact, or raw)      │
│ - content redaction (strip question text, SQL, or   │
│   results from events)                              │
│ - privacyMode: "full" | "anonymized" |              │
│   "metadata_only"                                   │
│                                                     │
│ eventAllowlist and eventDenylist reuse existing      │
│ BigQueryLoggerConfig concepts. All other controls   │
│ (privacyMode, content redaction, user ID            │
│ anonymization, allowUserOptOut) are NEW CA-layer     │
│ controls requiring new implementation.              │
└─────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────┐
│ Tier 3: User Visibility (Informational)             │
│                                                     │
│ When analytics is enabled, the CA API response      │
│ includes a field:                                   │
│   "analyticsDisclosure": {                          │
│     "enabled": true,                                │
│     "destination": "projects/P/datasets/D/tables/T",│
│     "privacyMode": "metadata_only"                  │
│   }                                                 │
│                                                     │
│ The end user is INFORMED that their interactions    │
│ are being logged. They do NOT need to individually   │
│ consent (the admin made this decision at Tier 1).   │
│ This follows the same model as Cloud Audit Logs.    │
│                                                     │
│ Optional: Admin can enable user opt-out by setting  │
│ "allowUserOptOut": true. Users then send:           │
│   X-CA-Analytics-OptOut: true                       │
│ in their request headers to suppress logging for    │
│ that request.                                       │
└─────────────────────────────────────────────────────┘
```

### 5.3 Privacy Modes

| Mode | user_id | Question Text | Generated SQL | Query Results | Latency/Errors |
|------|---------|---------------|---------------|---------------|----------------|
| `full` | Raw email | Logged | Logged | Logged | Logged |
| `anonymized` | SHA-256 hash | Logged | Logged | NOT logged | Logged |
| `metadata_only` | SHA-256 hash | NOT logged | NOT logged | NOT logged | Logged |

**Default: `metadata_only`.** Given the corrected threat model (analytics is a new durable disclosure surface), the safest default logs only structural metadata: event types, latency, error presence, row counts, with user identity hashed and question text/SQL redacted. This gives operators performance analytics without creating a durable record of user questions or generated SQL. Admins can escalate to `anonymized` (adds question text and SQL but hashes user identity and omits results) or `full` (everything, for debugging). This default is more defensible in privacy/legal review.

Note: `anonymized` mode anonymizes the **user identity** (SHA-256 hash) but still logs **question text and SQL**. This is content disclosure, not just metadata. Naming this mode "anonymized" may be misleading to admins who expect it to mean "no PII." Consider renaming to `identity_only_masked` or adding a clear admin-facing description: "User email is hashed. Question text and SQL are logged in full."

## 6. New Event Types for CA

Extend the existing `agent_events` schema with CA-specific event types:

| Event Type | Trigger | Content (JSON) |
|---|---|---|
| `CA_QUESTION_RECEIVED` | User sends a chat message | `{"question": "...", "agentId": "...", "dataSource": "bigquery"}` |
| `CA_SQL_GENERATED` | LLM generates a SQL query | `{"sql": "SELECT ...", "confidence": 0.85}` |
| `CA_QUERY_EXECUTED` | SQL runs against the data source | `{"rowsReturned": 150, "bytesProcessed": 1048576}` |
| `CA_RESPONSE_DELIVERED` | Final answer returned to user | `{"hasVisualization": true, "responseLength": 342}` |
| `CA_ERROR` | Any failure in the pipeline | `{"stage": "sql_generation", "error": "..."}` |
| `CA_VERIFIED_QUERY_MATCH` | Question matched a verified query | `{"verifiedQueryId": "...", "matchConfidence": 0.92}` |

**Privacy mode impact on content fields:**
- `full`: All content fields logged as shown above
- `anonymized`: `question` text is logged, `sql` is logged, query results are NOT logged (replaced with `{"redacted": true}`)
- `metadata_only`: `question`, `sql`, and results are all replaced with `{"redacted": true}`. Only structural metadata (event type, latency, error presence, row counts) is retained.

These extend (not replace) the existing ADK event types. The `attributes.source` field distinguishes CA events from ADK events in the same table.

## 7. CA API Server-Side Integration

The CA API server needs to:

1. Read `analyticsConfig` from the data agent configuration
2. Initialize a BQ AA writer using the BigQuery Storage Write API (same transport as the ADK plugin)
3. Emit events at each stage of the chat pipeline via fire-and-forget background writes (see Section 4.4, no synchronous flush on the response path)
4. Include `analyticsDisclosure` in chat responses when analytics is enabled
5. Respect `privacyMode` when constructing event content (redact fields per Section 5.3)
6. Respect `X-CA-Analytics-OptOut` header when `allowUserOptOut` is true

### Per-event-type BQ views

The BQ AA plugin's `create_views` feature auto-generates per-event-type BigQuery views (e.g., `v_llm_request`, `v_tool_completed`) that unnest JSON into flat columns. The CA integration must register the 6 new CA_* event types so the view layer can generate corresponding views (`v_ca_question_received`, `v_ca_sql_generated`, etc.). This is a plugin-side change, not a CA API change.

## 8. Implementation Phases

### Phase 1: Server-Side Event Emission (4 weeks implementation effort)

Note: Calendar time depends on CA API team prioritization. This is implementation effort, not calendar commitment. Requires CA team to accept the design and allocate sprint capacity.

- Add `analyticsConfig` to the DataAgent protobuf/API surface
- Implement BQ AA writer initialization in the CA API chat handler
- Emit 6 core CA_ prefixed event types (CA_QUESTION_RECEIVED through CA_VERIFIED_QUERY_MATCH)
- Non-blocking fire-and-forget writes with background flush (see Section 4.4, NOT synchronous flush)
- Privacy mode support (full, anonymized, metadata_only; default: metadata_only)
- Auto-create table on first write with schema versioning
- Write failure handling (retry in background, log, drop; see Section 4.5)
- Analytics OFF by default
- Register CA_* event types in the BQ AA plugin's view layer so `create_views` generates CA-specific views

### Phase 2: IAM Automation (Weeks 5-6)

- Script or gcloud commands to automate granting `bigquery.dataEditor` to the CA service identity
- Automated detection of CA service account identity from project number
- Dataset-level (not project-level) IAM grant
- Verification command to confirm analytics pipeline is healthy

### Phase 3: Consent UX + Disclosure (Weeks 7-8)

- `analyticsDisclosure` field in chat API responses
- Optional `X-CA-Analytics-OptOut` header support
- Admin console integration (if CA has a console UI)
- Documentation for admin consent workflow

### Phase 4: Dashboards + Documentation (Weeks 9-12)

- Looker Studio dashboard template for CA analytics
- Codelab: "Monitor your CA data agent with BQ Agent Analytics"
- End-to-end integration test: create agent -> enable analytics -> chat -> verify events in BQ

## 9. Open Questions

1. **CA service account name:** Is it `service-{NUM}@gcp-sa-geminidataanalytics.iam.gserviceaccount.com`? Verify with the CA API team.
2. **Cross-project analytics:** If a CA agent in Project A queries BQ in Project B, should analytics events go to Project A's dataset (where the agent lives) or Project B's (where the data lives)? Recommendation: Project A (the agent operator's project).
3. **Looker/Database sources:** When CA queries Looker or AlloyDB (not BigQuery), the BQ AA writer still writes to BigQuery. The analytics dataset is always in BQ regardless of the queried data source. Is this confusing to users?
4. **Rate limiting:** Should analytics writes be rate-limited for high-volume CA agents? The BQ Storage Write API has quotas. Need projected CA chat volume to size the writer.
5. **Existing CA conversation storage:** The CA API already stores conversation history internally. How does BQ AA relate to that? Is it a replacement, a complement, or a migration path?
6. **Pre-GA status:** The CA API is pre-GA. Adding analytics to a pre-GA API adds complexity. Should analytics wait for GA, or ship alongside to gather early feedback?

## 10. Success Criteria

- Phase 1: CA API can emit 6 core event types to a customer's BQ dataset. Analytics is OFF by default, ON when admin enables it.
- Phase 2: IAM setup is scriptable and verified end-to-end.
- Phase 3: 10+ CA data agents have analytics enabled in customer projects.
- 6-month: CA analytics events/month reaches 10% of total BQ AA events (currently 160M/month from 3P).
- 12-month: Unified dashboard covering both ADK and CA agent analytics used by 5+ enterprise customers.

## 11. Risks

- **CA API team buy-in:** This requires changes to the CA API server. If the CA team doesn't prioritize it, this blocks.
- **Pre-GA instability:** CA API is in preview. Analytics adds another failure mode to an already unstable surface.
- **Consent complexity:** The three-tier model is clean on paper but may need legal/privacy review at Google.
- **Scope creep:** Supporting all 6 CA data sources (BQ, Looker, Looker Studio, AlloyDB, Spanner, Cloud SQL) is a lot. Consider starting with BigQuery-sourced CA agents only.
