# BigQuery Agent Analytics Blog Post

Blog post advocating for the BigQuery Agent Analytics Plugin in [Agent Starter Pack](https://github.com/GoogleCloudPlatform/agent-starter-pack).

## Contents

- **[blog-post.md](blog-post.md)** — Full blog post: *"Your AI Agent Is a Black Box — BigQuery Agent Analytics Fixes That"*

## Summary

This blog post walks Agent Starter Pack users through enabling BigQuery Agent Analytics with the `--bq-analytics` CLI flag. It covers:

- What the plugin captures (event types, schema, token usage)
- Why production agents need structured observability
- Step-by-step setup: project generation, local testing, Cloud Run deployment
- SQL query examples for cost tracking, debugging, and behavior analysis
- Advanced configuration (event filtering, multimodal content, batching)
- Dashboard setup with Looker Studio

## Target Audience

Agent Starter Pack users who are not yet familiar with BigQuery Agent Analytics and want to add production-grade observability to their agents.

## References

- [Agent Starter Pack](https://github.com/GoogleCloudPlatform/agent-starter-pack)
- [ADK Documentation](https://google.github.io/adk-docs/)
- [BigQuery Agent Analytics Codelab](https://codelabs.developers.google.com/adk-bigquery-agent-analytics-plugin)
- [Introducing BigQuery Agent Analytics — Google Cloud Blog](https://cloud.google.com/blog/products/data-analytics/introducing-bigquery-agent-analytics)
