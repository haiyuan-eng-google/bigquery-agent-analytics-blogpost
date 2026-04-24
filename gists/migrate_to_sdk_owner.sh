#!/usr/bin/env bash
# migrate_to_sdk_owner.sh — Re-create the four post-#2 Gists under
# the Google Cloud / SDK-owner GitHub handle.
#
# Run this as whoever holds the SDK-owner `gh auth` — i.e. the
# @google-cloud, @GoogleCloudPlatform, or equivalent handle that
# owns the other companion assets in this series.
#
# Why this exists: post #1's PUBLISH_CHECKLIST recommends Gists on
# the SDK-owner account so the "Open in GitHub" backlink in the
# Medium embed points at an authoritative source. The initial
# drafts of post #2 created Gists under `caohy1988` for PR-review
# convenience; this script promotes them.

set -eu

cd "$(dirname "$0")"

need() { command -v "$1" >/dev/null 2>&1 || { echo "missing: $1" >&2; exit 1; }; }
need gh
need awk

ACCT=$(gh api user --jq .login 2>/dev/null || true)
if [ -z "$ACCT" ]; then
  echo "gh not authenticated. Run: gh auth login (as the SDK-owner handle)." >&2
  exit 1
fi
echo "Authenticated as: $ACCT"
case "$ACCT" in
  caohy1988|haiyuan-eng-google)
    cat <<EOT >&2
Heads up — you're authenticated as "$ACCT", which looks personal.
If the goal is to put the Gists under an SDK-owner handle
(@google-cloud, @GoogleCloudPlatform, etc.), stop here and
re-auth with 'gh auth login --hostname github.com' under the
right account before re-running this script.
EOT
  read -p "Continue under $ACCT anyway? [y/N] " yn
  case "$yn" in [yY]*) : ;; *) echo "Aborted."; exit 0 ;; esac
  ;;
esac

create_one() {
  local path="$1"
  local desc="$2"
  echo "--- Creating Gist for $path"
  gh gist create --public --desc "$desc" "$path" 2>&1 | awk '
    /^https:\/\// { print "URL: " $0; next }
    { print "  " $0 }'
}

create_one 04_evaluate_exit_code_one_liner.sh \
  "Medium post #2 — section 3 hero command: evaluate --exit-code one-liner"
create_one 05_evaluate_thresholds_workflow.yml \
  "Medium post #2 — section 4 reference GitHub Actions workflow: four deterministic agent quality gates"
create_one 06_categorical_eval_metrics_and_gate.md \
  "Medium post #2 — section 5 categorical CI gate: metrics.json + one command"
create_one 07_information_schema_ci_cost.sql \
  "Medium post #2 — section 6 INFORMATION_SCHEMA cost-per-feature pivot for SDK-labeled BigQuery jobs"

cat <<EOT

Done. Next steps:

1. Collect the four new URLs from the output above.
2. Paste them into the blog draft — three spots to update:
   a) Section 4 micro-CTA ('If you only copy one thing...')
      → replace the Gist 05 URL.
   b) Section 7 'Fork the workflow' CTA
      → replace the Gist 05 URL.
   c) Editorial-notes 'Gists for embedded code blocks' block
      → replace all four per-Gist URLs.
3. Optionally delete the draft-review Gists under caohy1988
   (https://gist.github.com/caohy1988) once the authoritative set
   is live.
EOT
