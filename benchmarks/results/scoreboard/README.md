# Recorded bests

The best scoreboard result for each hardware configuration and model:
`<hardware>/<model>.json`, for example `apple-m1-16gb/gemma-3-1b.json`.
[`benchmarks/scoreboard.py`](../../scoreboard.py) compares every run with
these files ([rules](../../../docs/IMPROVEMENT_LOOP.md#scoreboard-rules)).

Each file is a compact native benchmark report: the full hardware, software,
model, workload and correctness records and every repetition's metrics. Only
the first repetition's token ids are kept, and the local snapshot path is
dropped. It is hardware evidence for exactly that chip, memory size, model
revision and workload.

To promote, copy a record from the run's `promote/` directory (in the
`apple-silicon-evidence-*` artifact, or the comment's "Records to promote"
block) to the same path here, and commit it with the change that earned it.
Unlike the raw records one level up, a best is replaced when it is beaten.
`git log` keeps the history.
