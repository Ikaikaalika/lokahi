#!/usr/bin/env python3
"""Benchmark the native Lokahi engine on one pinned, locally staged artifact.

The report uses the MLX-LM baseline schema (``benchmark_mlx_lm.py``) so
``compare_mlx_baseline.py`` can score it against the recorded control. The
prompt is built with the same rule as the control: BOS followed by a repeated
non-EOS token. Timing comes from inside the native engine; Python only
prepares inputs and writes the report. Never downloads.

``--speculate N`` times greedy speculative decoding instead (``--drafter``
lookup, mtp or both; ``--draft-model`` a local ``qwen3_5_mtp`` drafter
snapshot). The engine first decodes plainly, untimed, and the run fails
unless every repetition reproduces it. How many drafts are accepted depends
on the text, so ``--prompt-file`` replaces the repeated-token prompt with a
text one, as one user turn in the model's chat template.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from benchmarks.benchmark_mlx_lm import (
        DEFAULT_LADDER,
        _artifact_metadata,
        _distribution,
        _hardware,
        _prompt_tokens,
        summarize_repetitions,
        validate_local_snapshot,
    )
    from benchmarks.commoncompute_model_ladder import load_model_ladder, model_entry
except ModuleNotFoundError:  # Direct ``python benchmarks/...`` execution.
    from benchmark_mlx_lm import (  # type: ignore[no-redef]
        DEFAULT_LADDER,
        _artifact_metadata,
        _distribution,
        _hardware,
        _prompt_tokens,
        summarize_repetitions,
        validate_local_snapshot,
    )
    from commoncompute_model_ladder import load_model_ladder, model_entry  # type: ignore[no-redef]


def _load_tokenizer(snapshot: Path) -> Any:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from mlx_lm.utils import load_tokenizer

    return load_tokenizer(snapshot)


_SHA40 = re.compile(r"[0-9a-f]{40}")


def validate_drafter_snapshot(path: Path, approved_ssd_root: Path) -> Path:
    """A local ``qwen3_5_mtp`` drafter snapshot, pinned by its 40-hex revision directory."""
    resolved = path.expanduser().resolve()
    if _SHA40.fullmatch(resolved.name) is None:
        raise ValueError("draft model path must end in its exact 40-hex snapshot revision")
    snapshot = validate_local_snapshot(resolved, approved_ssd_root, resolved.name)
    model_type = json.loads((snapshot / "config.json").read_text(encoding="utf-8")).get("model_type")
    if model_type != "qwen3_5_mtp":
        raise ValueError(f"draft model must be a qwen3_5_mtp drafter, not {model_type!r}")
    return snapshot


def text_prompt_tokens(tokenizer: Any, text: str) -> list[int]:
    """``text`` as one user turn in the tokenizer's chat template, or plainly encoded without one."""
    if getattr(tokenizer, "chat_template", None):
        tokens = tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True)
    else:
        tokens = tokenizer.encode(text)
    return [int(token) for token in tokens]


def speculation_arguments(max_draft: int, drafter: str, draft_model: Path | None) -> list[str]:
    if max_draft <= 0:
        return []
    arguments = ["--speculate", str(max_draft), "--drafter", drafter]
    return arguments + (["--draft-model", str(draft_model)] if draft_model is not None else [])


def summarize_speculation(repetitions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Accepted drafts over drafted ones, and tokens emitted per verify pass (accepted drafts plus one)."""
    rates, per_pass = [], []
    for repetition in repetitions:
        stats = repetition["speculation"]
        rates.append(stats["accepted"] / stats["drafted"] if stats["drafted"] else 0.0)
        per_pass.append((stats["accepted"] + stats["passes"]) / stats["passes"] if stats["passes"] else 0.0)
    return {"acceptance_rate": _distribution(rates), "tokens_per_verify_pass": _distribution(per_pass)}


def run_native_bench(
    cli: Path,
    snapshot: Path,
    prompt: list[int],
    output_tokens: int,
    warmups: int,
    repetitions: int,
    backend: str,
    max_context: int,
    extra: Sequence[str] = (),
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="lokahi-bench-") as directory:
        tokens_file = Path(directory) / "prompt.txt"
        tokens_file.write_text(",".join(map(str, prompt)), encoding="utf-8")
        result = subprocess.run(
            [
                str(cli), "bench",
                "--model", str(snapshot),
                "--backend", backend,
                "--tokens-file", str(tokens_file),
                "--max-new", str(output_tokens),
                "--warmups", str(warmups),
                "--repetitions", str(repetitions),
                "--max-context", str(max_context),
                *extra,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    if result.returncode not in (0, 1):
        raise RuntimeError(f"native bench failed: {result.stderr.strip()}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ladder", type=Path, default=DEFAULT_LADDER)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--approved-ssd-root", type=Path, required=True)
    parser.add_argument("--native-cli", type=Path, required=True)
    parser.add_argument("--backend", default="metal", choices=("metal", "cpu"))
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--prompt-file", type=Path, help="a text prompt instead of the repeated-token one")
    parser.add_argument("--speculate", type=int, default=0, help="draft up to N tokens per verify pass")
    parser.add_argument("--drafter", default="lookup", choices=("lookup", "mtp", "both"))
    parser.add_argument("--draft-model", type=Path, help="a local qwen3_5_mtp drafter snapshot")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.prompt_tokens, args.output_tokens, args.repetitions) <= 0 or args.warmups < 0:
        parser.error("token counts and repetitions must be positive")
    if args.speculate < 0 or (args.draft_model is not None and args.speculate == 0):
        parser.error("--speculate must not be negative, and --draft-model needs it")

    ladder = load_model_ladder(args.ladder)
    entry = model_entry(ladder, args.model_id)
    snapshot = validate_local_snapshot(args.model_path, args.approved_ssd_root, entry["revision"])
    drafter = (validate_drafter_snapshot(args.draft_model, args.approved_ssd_root)
               if args.draft_model is not None else None)
    speculating = args.speculate > 0
    tokenizer = _load_tokenizer(snapshot)
    if args.prompt_file is not None:
        text = args.prompt_file.read_text(encoding="utf-8")
        prompt = text_prompt_tokens(tokenizer, text)
        prompt_source: Any = {"file": args.prompt_file.name,
                              "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                              "chat_template": bool(getattr(tokenizer, "chat_template", None))}
    else:
        prompt = _prompt_tokens(tokenizer, args.prompt_tokens)
        prompt_source = "BOS then a repeated non-EOS token"
    native = run_native_bench(
        args.native_cli.expanduser().resolve(),
        snapshot,
        prompt,
        args.output_tokens,
        args.warmups,
        args.repetitions,
        args.backend,
        max_context=len(prompt) + args.output_tokens,
        extra=speculation_arguments(args.speculate, args.drafter, drafter),
    )
    repetitions = native["repetitions"]
    matches_plain = not speculating or native["speculation"]["matches_plain_greedy"]
    report = {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_kind": "hardware",
        "runner": native["runner"],
        "target_id": ladder["target_id"],
        "model": {
            "model_id": entry["model_id"],
            "repo": entry["repo"],
            "revision": entry["revision"],
            "local_snapshot": str(snapshot),
            "artifact": _artifact_metadata(snapshot),
        },
        "hardware": _hardware(),
        "software": {
            "python": platform.python_version(),
            "lokahi_backend": native["backend"],
            "lokahi_revision": subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
            ).stdout.strip()
            or "unknown",
        },
        "workload": {
            "prompt_tokens_per_sequence": len(prompt),
            "prompt_source": prompt_source,
            "output_tokens_per_sequence": args.output_tokens,
            "batch_size": 1,
            "sampling": "greedy",
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "prefix_cache": False,
            "speculation": {
                "max_draft": args.speculate,
                "drafter": args.drafter,
                "draft_model": None if drafter is None else {
                    "local_snapshot": str(drafter),
                    "revision": drafter.name,
                    "artifact": _artifact_metadata(drafter),
                },
            } if speculating else None,
        },
        "weight_residency": {
            "ssd_offload_parameter": "disabled",
            "model_load_lazy": False,
            "manager": "lokahi-native",
            "lokahi_managed_paging": False,
            "semantics": "weights copied once into resident GPU buffers at load",
        },
        "correctness": {
            "requested_token_count_passed": all(
                len(rep["token_ids"]) == args.output_tokens for rep in repetitions
            ),
            "repetition_token_invariance": native["repetition_token_invariance"],
            "speculative_tokens_match_plain_greedy": matches_plain if speculating else None,
            "numerical_error": "not measured here; see tests/test_native_engine.py for oracle parity",
        },
        "measurement": {
            "units": {
                "latency": "milliseconds",
                "throughput": "tokens per second",
                "memory": "decimal gigabytes of process peak resident set size",
            },
            "timing_boundaries": {
                "ttft_ms": "native: prompt submission to first sampled token visible on the host",
                "inter_token_latency_ms": (
                    "native: decode time per token, the mean (a verify pass emits its tokens together)"
                    if speculating else "native: median host-observed interval between tokens"
                ),
                "prompt_tokens_per_second": "prompt tokens divided by ttft",
                "decode_tokens_per_second": "(output tokens - 1) divided by first-to-last token time",
                "aggregate_tokens_per_second_including_prefill": "output tokens divided by ttft plus decode time",
            },
            "model_load_timed": False,
            "load_seconds": native["load_seconds"],
            "download_timed": False,
        },
        "summary": summarize_repetitions(repetitions),
        "speculation_summary": summarize_speculation(repetitions) if speculating else None,
        "repetitions": repetitions,
        "claim_boundary": (
            "Warm native Lokahi engine on one exact pinned artifact; comparable to the "
            "MLX-LM baseline only for the same hardware, artifact, and workload."
            + (" Speculative speed holds only for this prompt and its output, whose drafts the model accepts "
               "at the recorded rate." if speculating else "")
        ),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.expanduser().resolve().write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if native["repetition_token_invariance"] and matches_plain else 1


if __name__ == "__main__":
    raise SystemExit(main())
