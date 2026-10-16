#!/usr/bin/env python3
"""MLX-VLM control for greedy speculative decoding: plain and MTP runs of one local Qwen3.5 snapshot.

The control for ``benchmark_lokahi_native.py --speculate``: the same prompt
token ids go in as ``input_ids``, end of sequence never stops generation (as
in ``lokahi bench``), and timing is host-observed with the native bench's
boundaries: time to first token from the call, and the decode rate as the
tokens after the first over first-to-last token time. MLX-VLM's block size
counts the verified token, so ``--speculate N`` is ``draft_block_size``
N + 1. A run without ``--speculate`` is plain greedy decoding, and every run
first decodes plainly, untimed, to check that speculation keeps the tokens.
Never downloads.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

try:
    from benchmarks.benchmark_lokahi_native import summarize_speculation, validate_drafter_snapshot
    from benchmarks.benchmark_mlx_lm import _artifact_metadata, _hardware, summarize_repetitions
except ModuleNotFoundError:  # Direct ``python benchmarks/...`` execution.
    from benchmark_lokahi_native import summarize_speculation, validate_drafter_snapshot  # type: ignore[no-redef]
    from benchmark_mlx_lm import _artifact_metadata, _hardware, summarize_repetitions  # type: ignore[no-redef]


def load_control(snapshot: Path, drafter: Path | None) -> tuple[Any, Any, Any, str | None]:
    """MLX-VLM's model and processor, with end of sequence disabled, and its drafter when given."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from mlx_vlm.utils import load

    model, processor = load(str(snapshot))
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    tokenizer.stopping_criteria = lambda token: False
    if drafter is None:
        return model, processor, None, None
    from mlx_vlm.speculative.drafters import load_drafter

    draft_model, kind = load_drafter(str(drafter))
    return model, processor, draft_model, kind


def generate_once(model: Any, processor: Any, prompt: list[int], max_new: int, draft_model: Any = None,
                  kind: str | None = None, max_draft: int = 0) -> dict[str, Any]:
    """One greedy generation: its tokens, host token times and, when drafting, its rounds."""
    import mlx.core as mx
    from mlx_vlm.generate import stream_generate

    options: dict[str, Any] = {}
    if draft_model is not None:
        options = {"draft_model": draft_model, "draft_kind": kind, "draft_block_size": max_draft + 1}
        draft_model.accept_lens, draft_model.draft_lens = [], []  # this run's rounds only
    mx.reset_peak_memory()
    start = time.perf_counter()
    times: list[float] = []
    last = None
    for result in stream_generate(model, processor, "", input_ids=mx.array([prompt]), max_tokens=max_new,
                                  temperature=0.0, **options):
        if result.token is not None and len(times) < max_new:
            times.append(time.perf_counter())
        last = result
    if last is None or last.token_ids is None or len(last.token_ids) != max_new or len(times) != max_new:
        raise RuntimeError("MLX-VLM did not generate the requested number of tokens")
    run: dict[str, Any] = {"start": start, "times": times, "token_ids": [int(t) for t in last.token_ids],
                           "prompt_tokens": int(last.prompt_tokens), "peak_memory_gb": mx.get_peak_memory() / 1e9}
    if draft_model is not None:
        accepted = list(draft_model.accept_lens)
        drafted = list(getattr(draft_model, "draft_lens", None) or [])
        run["speculation"] = {"passes": len(accepted), "drafted": sum(drafted), "accepted": sum(accepted)}
    return run


def repetition_metrics(run: dict[str, Any], prompt_tokens: int, speculating: bool) -> dict[str, Any]:
    """The native bench's per-repetition fields from one run's host token times."""
    times, start = run["times"], run["start"]
    count = len(times)
    decode = times[-1] - times[0]
    intervals = [1000.0 * (b - a) for a, b in zip(times, times[1:])]
    metrics = {
        "ttft_ms": 1000.0 * (times[0] - start),
        "inter_token_latency_ms": 1000.0 * decode / (count - 1) if speculating else statistics.median(intervals),
        "prompt_tokens_per_second": prompt_tokens / (times[0] - start),
        "decode_tokens_per_second": (count - 1) / decode,
        "aggregate_tokens_per_second_including_prefill": count / (times[-1] - start),
        "peak_memory_gb": run["peak_memory_gb"],
        "token_ids": run["token_ids"],
    }
    if speculating:
        metrics["speculation"] = run["speculation"]
    return metrics


def bench(model: Any, processor: Any, prompt: list[int], max_new: int, warmups: int, repetitions: int,
          draft_model: Any = None, kind: str | None = None, max_draft: int = 0) -> dict[str, Any]:
    speculating = draft_model is not None
    plain = generate_once(model, processor, prompt, max_new)["token_ids"] if speculating else None
    reps = []
    for index in range(warmups + repetitions):
        run = generate_once(model, processor, prompt, max_new, draft_model, kind, max_draft)
        if run["prompt_tokens"] != len(prompt):
            raise RuntimeError("MLX-VLM changed the prompt tokens")
        if index >= warmups:
            reps.append(repetition_metrics(run, len(prompt), speculating))
    tokens = [rep["token_ids"] for rep in reps]
    return {
        "repetitions": reps,
        "repetition_token_invariance": all(t == tokens[0] for t in tokens),
        "matches_plain_greedy": all(t == plain for t in tokens) if speculating else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True, help="a local snapshot at its 40-hex revision")
    parser.add_argument("--model-repo", required=True)
    parser.add_argument("--approved-ssd-root", type=Path, required=True)
    parser.add_argument("--tokens-file", type=Path, required=True, help="the prompt's token ids, comma-separated")
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--speculate", type=int, default=0, help="draft up to N tokens per verify pass")
    parser.add_argument("--draft-model", type=Path, help="a local qwen3_5_mtp drafter snapshot")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output_tokens < 2 or args.repetitions <= 0 or args.warmups < 0 or args.speculate < 0:
        parser.error("needs --output-tokens >= 2, positive repetitions and non-negative warmups and --speculate")
    if (args.speculate > 0) != (args.draft_model is not None):
        parser.error("--speculate and --draft-model go together")

    from benchmarks.benchmark_mlx_lm import validate_local_snapshot

    snapshot = validate_local_snapshot(args.model_path, args.approved_ssd_root, args.model_path.resolve().name)
    drafter = validate_drafter_snapshot(args.draft_model, args.approved_ssd_root) if args.draft_model else None
    prompt = [int(t) for t in args.tokens_file.read_text(encoding="utf-8").replace(",", " ").split()]
    model, processor, draft_model, kind = load_control(snapshot, drafter)
    result = bench(model, processor, prompt, args.output_tokens, args.warmups, args.repetitions, draft_model, kind,
                   args.speculate)
    speculating = drafter is not None
    repetitions = result["repetitions"]
    report = {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_kind": "hardware",
        "runner": "mlx-vlm-control",
        "model": {"repo": args.model_repo, "revision": snapshot.name, "local_snapshot": str(snapshot),
                  "artifact": _artifact_metadata(snapshot)},
        "hardware": _hardware(),
        "software": {"python": platform.python_version(),
                     **{name: metadata.version(name) for name in ("mlx", "mlx-vlm")}},
        "workload": {
            "prompt_tokens_per_sequence": len(prompt),
            "output_tokens_per_sequence": args.output_tokens,
            "batch_size": 1,
            "sampling": "greedy",
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "speculation": {"max_draft": args.speculate, "drafter": kind, "draft_block_size": args.speculate + 1,
                            "draft_model": {"local_snapshot": str(drafter), "revision": drafter.name,
                                            "artifact": _artifact_metadata(drafter)}} if speculating else None,
        },
        "correctness": {
            "repetition_token_invariance": result["repetition_token_invariance"],
            "speculative_tokens_match_plain_greedy": result["matches_plain_greedy"],
        },
        "measurement": {
            "units": {"latency": "milliseconds", "throughput": "tokens per second",
                      "memory": "decimal gigabytes of MLX peak memory"},
            "timing_boundaries": {
                "ttft_ms": "host: stream_generate call to the first token observed",
                "inter_token_latency_ms": ("host: decode time per token, the mean" if speculating
                                           else "host: median interval between tokens"),
                "decode_tokens_per_second": "(output tokens - 1) divided by first-to-last token time",
            },
            "model_load_timed": False,
        },
        "summary": summarize_repetitions(repetitions),
        "speculation_summary": summarize_speculation(repetitions) if speculating else None,
        "repetitions": repetitions,
        "claim_boundary": "Warm MLX-VLM control on one local snapshot; comparable to Lōkahi only for the same "
                          "hardware, snapshot, prompt tokens and output length.",
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.expanduser().resolve().write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    ok = result["repetition_token_invariance"] and result["matches_plain_greedy"] is not False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
