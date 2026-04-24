# Categorical CI gate — metrics.json + one command

Companion to Medium post #2, section 5
("Going deeper — add a categorical gate").

Requires bigquery-agent-analytics >= 0.2.2, which ships
`categorical-eval --exit-code --pass-category --min-pass-rate`.

## 1. Metrics file

Save as `metrics.json`:

```json
{
  "metrics": [
    {
      "name": "response_usefulness",
      "definition": "Did the assistant give a useful, actionable answer?",
      "categories": [
        {"name": "useful", "definition": "Direct, complete answer."},
        {"name": "partially_useful", "definition": "Answer is partial or missing key info."},
        {"name": "not_useful", "definition": "Refusal, hallucination, or off-topic."}
      ]
    }
  ]
}
```

## 2. Gate command

```bash
bq-agent-sdk categorical-eval \
  --metrics-file=metrics.json \
  --last=24h --agent-id=calendar_assistant \
  --pass-category=response_usefulness=useful \
  --pass-category=response_usefulness=partially_useful \
  --min-pass-rate=0.9 \
  --exit-code \
  --project-id="$PROJECT_ID" --dataset-id="$DATASET_ID"
```

What the flags mean:

- `--pass-category METRIC=CATEGORY` (repeatable) declares which
  classifications count as passing. Multiple values for the same
  metric OR together — `useful` OR `partially_useful` both pass,
  `not_useful` fails.
- `--min-pass-rate 0.9` sets the threshold; the run passes iff at
  least 90% of the last 24 hours of sessions land in a pass
  category.
- `--exit-code` turns pass/fail into a process exit code so CI
  runners can gate on it natively.

Parse errors, missing classifications, and out-of-bounds categories
all count as failing for the declared metric — a broken
classification run cannot silently pass CI.

## 3. Example failure output

```
--exit-code: 1 metric(s) under min-pass-rate 0.9
  FAIL metric=response_usefulness pass_rate=0.82 (102/124) min=0.9 pass_categories=partially_useful,useful
```

## 4. Wire it into the GitHub Actions workflow

Add a step to
[`evaluate_thresholds.yml`](https://gist.github.com/TBD-workflow-gist)
from post #2 section 4:

```yaml
      - name: Categorical usefulness gate
        run: >
          bq-agent-sdk categorical-eval --metrics-file=metrics.json
          --last=24h --agent-id=calendar_assistant
          --pass-category=response_usefulness=useful
          --pass-category=response_usefulness=partially_useful
          --min-pass-rate=0.9
          --exit-code
          --project-id=${{ vars.PROJECT_ID }}
          --dataset-id=${{ vars.DATASET_ID }}
```
