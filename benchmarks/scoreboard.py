#!/usr/bin/env python3
"""Scoreboard for the improvement loop (docs/IMPROVEMENT_LOOP.md).

For every pinned model it records, on this Mac:

* the native engine's warm benchmark (``benchmark_lokahi_native.py``);
* a fresh MLX-LM control on the same OS build and thermal context
  (``benchmark_mlx_lm.py``) and the comparison between them;
* where decode time goes (``lokahi profile`` + ``attribution.py``);
* the change against the best result recorded for this chip, memory size,
  model revision and workload, with a noise threshold from the repetitions.

It writes ``scoreboard.json`` (hardware evidence), ``scoreboard.md`` (the PR
comment) and, for models that improved or have no recorded best, a compact
record under ``promote/`` to commit to ``benchmarks/results/scoreboard/``.
The exit status is 1 when any model regressed or produced nondeterministic
tokens, so CI fails. Never downloads.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.attribution import attribute, peak_bandwidth  # noqa: E402
from benchmarks.attribution import render_markdown as render_attribution  # noqa: E402
from benchmarks.commoncompute_model_ladder import load_model_ladder, model_entry  # noqa: E402
from benchmarks.compare_mlx_baseline import BaselineComparisonError, compare_mlx_baseline  # noqa: E402

DEFAULT_MODELS = ("qwen3-0.6b", "llama-3.2-1b", "gemma-3-1b")
DEFAULT_LADDER = ROOT / "benchmarks" / "targets" / "commoncompute_m1_model_ladder_v1.json"
BEST_DIR = ROOT / "benchmarks" / "results" / "scoreboard"
MARKER = "<!-- lokahi-scoreboard -->"
# Metrics judged against the recorded best; decode and TTFT gate the run.
METRICS = {
    "decode_tokens_per_second": "higher",
    "ttft_ms": "lower",
    "prompt_tokens_per_second": "higher",
    "peak_memory_gb": "lower",
}
GATING = ("decode_tokens_per_second", "ttft_ms")
MIN_NOISE = 0.03  # changes under 3% are never called a regression or an improvement
COMMENT_LIMIT = 60_000  # GitHub rejects comment bodies over 65,536 characters


class ScoreboardError(ValueError):
    """Raised for reports that cannot be scored."""


# ---------------------------------------------------------------------------
# Pure scoring (unit tested without hardware)
# ---------------------------------------------------------------------------


def metric_stats(report: Mapping[str, Any], metric: str) -> tuple[float, float]:
    """Mean and coefficient of variation of `metric` over the repetitions."""

    values = [float(rep[metric]) for rep in report.get("repetitions", [])]
    if not values or not all(math.isfinite(v) for v in values):
        raise ScoreboardError(f"no finite {metric} repetitions")
    mean = statistics.fmean(values)
    if mean == 0:
        raise ScoreboardError(f"{metric} mean is zero")
    cv = statistics.stdev(values) / abs(mean) if len(values) > 1 else 0.0
    return mean, cv


def hardware_key(report: Mapping[str, Any]) -> str:
    """Directory name for a chip and memory size, e.g. ``apple-m1-16gb``."""

    hardware = report.get("hardware") or {}
    chip = str(hardware.get("chip", "unknown"))
    memory_gb = round(int(hardware.get("unified_memory_bytes", 0)) / 2**30)
    return re.sub(r"[^a-z0-9]+", "-", chip.lower()).strip("-") + f"-{memory_gb}gb"


def comparable(candidate: Mapping[str, Any], best: Mapping[str, Any]) -> str | None:
    """Why two native reports cannot be compared, or None when they can."""

    if hardware_key(candidate) != hardware_key(best):
        return "different chip or memory size"
    if (candidate.get("model") or {}).get("revision") != (best.get("model") or {}).get("revision"):
        return "different model revision"
    if candidate.get("workload") != best.get("workload"):
        return "different workload"
    if (candidate.get("software") or {}).get("lokahi_backend") != (best.get("software") or {}).get("lokahi_backend"):
        return "different backend"
    return None


def compare_to_best(candidate: Mapping[str, Any], best: Mapping[str, Any] | None) -> dict[str, Any]:
    """Per-metric change against the recorded best, with a noise threshold.

    ``change`` is signed so that positive is better. A change counts only
    beyond max(3%, 2 x the combined coefficient of variation).
    """

    if best is None:
        return {"verdict": "no baseline", "reason": "no best recorded for this Mac, model and workload"}
    reason = comparable(candidate, best)
    if reason is not None:
        return {"verdict": "no baseline", "reason": reason}
    metrics = {}
    for metric, direction in METRICS.items():
        mean, cv = metric_stats(candidate, metric)
        best_mean, best_cv = metric_stats(best, metric)
        change = (mean - best_mean) / best_mean
        if direction == "lower":
            change = -change
        noise = max(MIN_NOISE, 2.0 * math.sqrt(cv * cv + best_cv * best_cv))
        verdict = "improved" if change > noise else "regressed" if change < -noise else "unchanged"
        metrics[metric] = {"mean": mean, "best_mean": best_mean, "change": change, "noise": noise, "verdict": verdict}
    gating = [metrics[name]["verdict"] for name in GATING]
    verdict = "regressed" if "regressed" in gating else "improved" if "improved" in gating else "unchanged"
    return {"verdict": verdict, "metrics": metrics,
            "best_revision": (best.get("software") or {}).get("lokahi_revision")}


def token_agreement(a: Sequence[int], b: Sequence[int]) -> int:
    """Number of leading tokens two greedy sequences share."""

    count = 0
    for left, right in zip(a, b):
        if left != right:
            break
        count += 1
    return count


def compact_record(native: Mapping[str, Any]) -> dict[str, Any]:
    """The native report to commit as a best: no per-repetition token ids
    (the first run's are kept) and no local snapshot path."""

    record = json.loads(json.dumps(native))
    (record.get("model") or {}).pop("local_snapshot", None)
    repetitions = record.get("repetitions", [])
    for index, repetition in enumerate(repetitions):
        if index > 0:
            repetition.pop("token_ids", None)
    return record


def score_model(
    model_id: str,
    native: Mapping[str, Any] | None,
    control: Mapping[str, Any] | None,
    attribution: Mapping[str, Any] | None,
    best: Mapping[str, Any] | None,
    notes: Sequence[str] = (),
) -> dict[str, Any]:
    """One scoreboard row. ``native`` None means the engine did not produce a result."""

    entry: dict[str, Any] = {"model_id": model_id, "notes": list(notes)}
    if native is None:
        entry["status"] = "failed" if notes else "not staged"
        return entry
    correctness = native.get("correctness") or {}
    deterministic = correctness.get("repetition_token_invariance") is True and correctness.get(
        "requested_token_count_passed"
    ) is True
    entry["native"] = {metric: metric_stats(native, metric)[0] for metric in METRICS}
    entry["deterministic"] = deterministic
    entry["versus_best"] = compare_to_best(native, best)
    if control is not None:
        entry["control"] = {metric: metric_stats(control, metric)[0] for metric in METRICS}
        try:
            comparison = compare_mlx_baseline(control, native)
            entry["versus_mlx_lm"] = {
                row["metric"]: row["improvement_percent"] / 100.0 for row in comparison["comparisons"]
            }
            entry["standalone_mvp_gate_passed"] = comparison["standalone_mvp_gate_passed"]
        except BaselineComparisonError as exc:
            entry["notes"].append(f"MLX-LM comparison skipped: {exc}")
        ours = native["repetitions"][0]["token_ids"]
        theirs = control["repetitions"][0]["token_ids"]
        if ours and isinstance(ours[0], list):  # batched control layout
            ours = ours[0]
        if theirs and isinstance(theirs[0], list):
            theirs = theirs[0]
        entry["tokens_matching_mlx_lm"] = [token_agreement(ours, theirs), len(theirs)]
    if attribution is not None:
        entry["attribution"] = attribution
    verdict = entry["versus_best"]["verdict"]
    entry["status"] = "regressed" if verdict == "regressed" or not deterministic else verdict
    entry["promote"] = deterministic and verdict in ("improved", "no baseline")
    return entry


def exit_status(scoreboard: Mapping[str, Any]) -> int:
    return 1 if any(entry["status"] in ("regressed", "failed") for entry in scoreboard["models"]) else 0


def _percent(value: float | None) -> str:
    return "–" if value is None else f"{100 * value:+.1f}%"


def render_markdown(scoreboard: Mapping[str, Any]) -> str:
    """The PR comment: summary table, attribution, and records to promote."""

    hw = scoreboard.get("hardware") or {}
    wl = scoreboard.get("workload") or {}
    lines = [
        MARKER,
        f"## Lōkahi scoreboard: {hw.get('chip', '?')}, {round(int(hw.get('unified_memory_bytes', 0)) / 2**30)} GB, "
        f"macOS {hw.get('macos_version', '?')} ({hw.get('macos_build', '?')})",
        "",
        f"Hardware evidence at `{str(scoreboard.get('revision', '?'))[:12]}`: warm runs, "
        f"{wl.get('prompt_tokens_per_sequence', '?')} prompt + {wl.get('output_tokens_per_sequence', '?')} generated "
        f"tokens, greedy, batch 1, {wl.get('warmups', '?')} warmups + {wl.get('repetitions', '?')} repetitions, "
        "fresh MLX-LM control on the same run.",
        "",
        "| Model | Status | Decode tok/s | vs MLX-LM | TTFT ms | vs MLX-LM | Peak GB | vs best (decode, TTFT) | Tokens = MLX-LM |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for entry in scoreboard["models"]:
        if "native" not in entry:
            lines.append(f"| {entry['model_id']} | {entry['status']} | – | – | – | – | – | – | – |")
            continue
        native = entry["native"]
        versus = entry.get("versus_mlx_lm", {})
        best = entry["versus_best"]
        if best["verdict"] == "no baseline":
            best_text = f"no baseline ({best['reason']})"
        else:
            m = best["metrics"]
            best_text = (f"{best['verdict']} ({_percent(m['decode_tokens_per_second']['change'])}, "
                         f"{_percent(m['ttft_ms']['change'])})")
        match = entry.get("tokens_matching_mlx_lm")
        lines.append(
            f"| {entry['model_id']} | {entry['status']} | {native['decode_tokens_per_second']:.1f} | "
            f"{_percent(versus.get('decode_tokens_per_second'))} | {native['ttft_ms']:.1f} | "
            f"{_percent(versus.get('ttft_ms'))} | {native['peak_memory_gb']:.2f} | {best_text} | "
            f"{f'{match[0]}/{match[1]}' if match else '–'} |"
        )
    status = "regression" if exit_status(scoreboard) else "no regression"
    lines += ["", f"**Result: {status}.** Changes count only beyond max(3%, 2× the combined run-to-run variation)."]
    for entry in scoreboard["models"]:
        if entry.get("attribution"):
            lines += ["", f"### Where decode time goes: {entry['model_id']}", "",
                      render_attribution(entry["attribution"]).rstrip()]
        for note in entry.get("notes", []):
            lines.append(f"- {entry['model_id']}: {note}")
    promotions = [entry for entry in scoreboard["models"] if entry.get("promote")]
    if not promotions:
        return "\n".join(lines) + "\n"
    target = f"benchmarks/results/scoreboard/{scoreboard.get('hardware_key', '')}/"
    records = ["", f"<details><summary>Records to promote into {target}</summary>", ""]
    for entry in promotions:
        records += [f"`{entry['model_id']}.json`", "", "```json",
                    json.dumps(entry["record"], sort_keys=True, separators=(",", ":")), "```", ""]
    records.append("</details>")
    if len("\n".join(lines + records)) > COMMENT_LIMIT:
        names = ", ".join(f"`{entry['model_id']}.json`" for entry in promotions)
        records = ["", f"Records to promote into {target}: {names}, under `promote/` in the evidence artifact "
                       "(too long for a comment)."]
    return "\n".join(lines + records) + "\n"


# ---------------------------------------------------------------------------
# Orchestration on the self-hosted Mac
# ---------------------------------------------------------------------------


def _gpu_cores() -> int | None:
    try:
        raw = subprocess.run(["/usr/sbin/system_profiler", "SPDisplaysDataType", "-json"],
                             capture_output=True, text=True, check=True, timeout=60).stdout
        return int(json.loads(raw)["SPDisplaysDataType"][0]["sppci_cores"])
    except (OSError, subprocess.SubprocessError, KeyError, IndexError, ValueError):
        return None


def _run_json(command: list[str], output: Path) -> tuple[Mapping[str, Any] | None, str | None]:
    output.unlink(missing_ok=True)  # never score a stale report from an earlier run
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if output.exists():
        try:
            return json.loads(output.read_text(encoding="utf-8")), None
        except json.JSONDecodeError as exc:
            return None, f"{output.name}: {exc}"
    tail = (result.stderr or result.stdout).strip().splitlines()[-3:]
    return None, f"{Path(command[1]).name} exited {result.returncode}: {' / '.join(tail)}"


def _profile(cli: Path, snapshot: Path, prompt: list[int], steps: int, backend: str) -> tuple[Mapping[str, Any] | None, str | None]:
    with tempfile.TemporaryDirectory(prefix="lokahi-profile-") as directory:
        tokens = Path(directory) / "prompt.txt"
        tokens.write_text(",".join(map(str, prompt)), encoding="utf-8")
        result = subprocess.run(
            [str(cli), "profile", "--model", str(snapshot), "--backend", backend, "--tokens-file", str(tokens),
             "--max-new", str(steps), "--max-context", str(len(prompt) + steps + 2)],
            capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return None, f"profile failed: {result.stderr.strip()[-300:]}"
    return json.loads(result.stdout.strip().splitlines()[-1]), None


def build_scoreboard(
    runs: Sequence[Mapping[str, Any]],
    best_dir: Path,
    revision: str,
    workload: Mapping[str, Any],
    gpu_cores: int | None = None,
) -> dict[str, Any]:
    """Score measured runs against the best records under ``best_dir``.

    Each run has ``model_id`` and optional ``native``, ``control``,
    ``profile`` reports and ``notes``.
    """

    hardware: Mapping[str, Any] = {}
    models = []
    for run in runs:
        native = run.get("native")
        if native is None:
            models.append(score_model(run["model_id"], None, None, None, None, run.get("notes", ())))
            continue
        hardware = native.get("hardware") or hardware
        attribution = None
        if run.get("profile") is not None:
            attribution = attribute(run["profile"], peak_bandwidth(str(hardware.get("chip", "")), gpu_cores))
        best_file = best_dir / hardware_key(native) / f"{run['model_id']}.json"
        best = json.loads(best_file.read_text(encoding="utf-8")) if best_file.exists() else None
        scored = score_model(run["model_id"], native, run.get("control"), attribution, best, run.get("notes", ()))
        scored["record"] = compact_record(native)
        models.append(scored)
    return {
        "schema_version": 1,
        "evidence_kind": "hardware",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "revision": revision,
        "hardware": dict(hardware, gpu_cores=gpu_cores),
        "hardware_key": hardware_key({"hardware": hardware}) if hardware else None,
        "workload": dict(workload),
        "models": models,
    }


def write_outputs(scoreboard: Mapping[str, Any], out: Path) -> str:
    """Write scoreboard.json, scoreboard.md and promote/ records; return the Markdown."""

    out.mkdir(parents=True, exist_ok=True)
    for entry in scoreboard["models"]:
        if entry.get("promote"):
            target = out / "promote" / scoreboard["hardware_key"] / f"{entry['model_id']}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(entry["record"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "scoreboard.json").write_text(json.dumps(scoreboard, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = render_markdown(scoreboard)
    (out / "scoreboard.md").write_text(markdown, encoding="utf-8")
    return markdown


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--model-root", type=Path, required=True, help="approved SSD Hugging Face cache root")
    parser.add_argument("--native-cli", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--best-dir", type=Path, default=BEST_DIR)
    parser.add_argument("--ladder", type=Path, default=DEFAULT_LADDER)
    parser.add_argument("--backend", default="metal", choices=("metal", "cpu"))
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--profile-steps", type=int, default=32)
    parser.add_argument("--skip-control", action="store_true", help="omit the MLX-LM control")
    args = parser.parse_args()

    # MLX-LM loads the tokenizer that builds the shared prompt; import lazily
    # so the scoring functions stay importable without it.
    from benchmarks.benchmark_lokahi_native import _load_tokenizer
    from benchmarks.benchmark_mlx_lm import _prompt_tokens

    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    ladder = load_model_ladder(args.ladder)
    root = args.model_root.expanduser().resolve()
    workload = ["--prompt-tokens", str(args.prompt_tokens), "--output-tokens", str(args.output_tokens),
                "--warmups", str(args.warmups), "--repetitions", str(args.repetitions)]
    python = sys.executable
    runs: list[dict[str, Any]] = []
    for model_id in [m.strip() for m in args.models.split(",") if m.strip()]:
        entry = model_entry(ladder, model_id)
        snapshot = root / f"models--{entry['repo'].replace('/', '--')}" / "snapshots" / entry["revision"]
        run: dict[str, Any] = {"model_id": model_id, "notes": []}
        runs.append(run)
        if not snapshot.is_dir():
            continue
        common = ["--model-id", model_id, "--model-path", str(snapshot), "--approved-ssd-root", str(root), *workload]
        if not args.skip_control:
            run["control"], note = _run_json([python, str(ROOT / "benchmarks/benchmark_mlx_lm.py"), *common,
                                              "--output", str(out / f"mlx_lm_{model_id}.json")],
                                             out / f"mlx_lm_{model_id}.json")
            if note:
                run["notes"].append(f"MLX-LM control did not run: {note}")
        run["native"], note = _run_json([python, str(ROOT / "benchmarks/benchmark_lokahi_native.py"), *common,
                                         "--native-cli", str(args.native_cli), "--backend", args.backend,
                                         "--output", str(out / f"lokahi_{model_id}.json")],
                                        out / f"lokahi_{model_id}.json")
        if run["native"] is None:
            run["notes"].append(note or "the native benchmark wrote no report")
            continue
        prompt = _prompt_tokens(_load_tokenizer(snapshot), args.prompt_tokens)
        run["profile"], note = _profile(args.native_cli, snapshot, prompt, args.profile_steps, args.backend)
        if run["profile"] is None:
            run["notes"].append(note)
        else:
            (out / f"profile_{model_id}.json").write_text(json.dumps(run["profile"], indent=2) + "\n", encoding="utf-8")

    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    scoreboard = build_scoreboard(
        runs, args.best_dir, revision or "unknown",
        {"prompt_tokens_per_sequence": args.prompt_tokens, "output_tokens_per_sequence": args.output_tokens,
         "warmups": args.warmups, "repetitions": args.repetitions, "backend": args.backend},
        _gpu_cores(),
    )
    print(write_outputs(scoreboard, out), end="")
    return exit_status(scoreboard)


if __name__ == "__main__":
    raise SystemExit(main())
