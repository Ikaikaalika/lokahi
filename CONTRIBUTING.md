# Contributing to Lōkahi

Lōkahi is owned by Common Compute LLC and published under the PolyForm
Noncommercial License 1.0.0. Common Compute LLC also uses Lōkahi
commercially, so every contribution must come with rights that allow that.

## Contributor License Agreement

Before a pull request can be merged, its author must sign the Common Compute
Contributor License Agreement in [CLA.md](CLA.md). The agreement lets Common
Compute LLC license your contribution under any terms, including
commercially, while you keep your copyright.

The CLA is currently a draft awaiting legal review. Until it is final,
Common Compute LLC does not merge outside contributions; issues and bug
reports are welcome.

## Engineering rules

Read [AGENTS.md](AGENTS.md) first. In short:

- label every result as correctness, synthetic, or hardware evidence;
- never claim ANE, SSD, or throughput behavior without the measurements
  [`docs/PERFORMANCE_CONTRACT.md`](docs/PERFORMANCE_CONTRACT.md) requires;
- run the local gates before sending a change:

```sh
PYTHONPATH=src python3 -m pytest -q
PYTHONPATH=src python3 -m compileall -q src tests benchmarks
git diff --check
```
