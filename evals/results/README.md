# Evaluation results

This directory stores local evaluation histories and artifacts. Generated
contents are ignored by Git; this README is tracked.

```text
results/
├── README.md
├── eval_history.jsonl              # MCP scenario evaluation records
├── skill_history.jsonl             # Skill evaluation and re-scoring records
├── <run-id>/
│   └── <attempt-id>/
│       ├── artifact.json           # Attempt metadata, task, output and tool schemas
│       └── messages.json           # Native Pydantic AI conversation history
└── ...                             # Optional reports and logs
```

- `eval_history.jsonl` contains one JSON object per MCP scenario evaluation,
  with the run ID, git and environment metadata, metric scores and timing.
- `skill_history.jsonl` contains one JSON object per scoring of a skill attempt,
  including checks, scores, outcome and artifact path. Re-scoring an attempt
  appends a record with the same run and attempt IDs and a new scoring ID.
- `<run-id>/<attempt-id>/` contains the saved evidence for a skill attempt,
  including partial or failed conversations. `artifact.json` includes the
  scenario and generation metadata; `messages.json` holds the conversation.
  These files allow `make eval-rescore` to score an attempt again.
- Comparison reports (for example, `navigator-comparison.md` and `.json`) and
  logs may be written here when explicitly requested; their names are not fixed.

## Run, attempt and scoring IDs

A **run ID** identifies an evaluation session. All skill attempts in the same
pytest session share it, even when they use different scenarios or models; a new
session gets a new run ID. MCP scenario records also share a run ID per session.

An **attempt ID** identifies one execution of a skill scenario against one
model. Each execution gets its own ID and artifact directory, so a run can
contain many attempts. For example, `results/abc123/def456/` is attempt
`def456` within run `abc123`.

A **scoring ID** identifies one evaluation of a saved attempt. Re-scoring keeps
the original run and attempt IDs, creates a new scoring ID, and appends another
record to `skill_history.jsonl`; it does not run the agent again.

## Cleanup

Run `make evals-clean` to remove all contents, including hidden files and
subdirectories, except this README. Cleaning also removes history needed for
trends and artifacts needed for replay. Evaluation targets keep existing results.

See [the evaluation guide](../../docs/EVALS.md) for running evaluations, reports
and re-scoring.
