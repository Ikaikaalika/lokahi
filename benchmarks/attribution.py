#!/usr/bin/env python3
"""Rank where a forward pass spends its time, from ``lokahi profile`` output.

Each section of a phase (prefill or decode) gets its share of the phase's
section time, the memory bandwidth it achieved, and, when the chip's peak is
known, the time it would save running at that peak. Decode is bound by memory
bandwidth, so ``recoverable_seconds`` is the improvement loop's ranking: work
goes to the section with the most time above its bandwidth floor. This is
attribution, not throughput evidence; fast-path timing comes from ``bench``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

BANDWIDTH_TABLE = Path(__file__).parent / "targets" / "apple_silicon_memory_bandwidth_v1.json"


class AttributionError(ValueError):
    """Raised for profiles that cannot be attributed."""


def peak_bandwidth(chip: str, gpu_cores: int | None = None, table: Mapping[str, Any] | None = None) -> float | None:
    """Published peak in bytes per second, or None when it is unknown or ambiguous."""

    if table is None:
        table = json.loads(BANDWIDTH_TABLE.read_text(encoding="utf-8"))
    matches = [entry for entry in table["chips"] if entry["chip"] == chip.strip()]
    if len(matches) > 1:
        matches = [entry for entry in matches if entry.get("gpu_cores") == gpu_cores]
    if len(matches) != 1:
        return None
    return float(matches[0]["gb_per_second"]) * 1e9


def attribute_phase(phase: Mapping[str, Any], peak_bytes_per_second: float | None, per: int = 1) -> dict[str, Any]:
    """Attribution rows for one phase, ranked by recoverable (else total) time.

    ``per`` divides totals into per-token values (decode steps).
    """

    sections = phase.get("sections")
    if not isinstance(sections, list) or not sections:
        raise AttributionError("phase has no sections")
    if per <= 0:
        raise AttributionError("per must be positive")
    total = sum(float(section["seconds"]) for section in sections)
    if total <= 0:
        raise AttributionError("phase sections took no time")
    rows = []
    for section in sections:
        seconds = float(section["seconds"])
        traffic = float(section["bytes"])
        if seconds < 0 or traffic < 0:
            raise AttributionError(f"section {section.get('name')!r} has negative time or traffic")
        achieved = traffic / seconds if seconds > 0 else None
        row: dict[str, Any] = {
            "name": section["name"],
            "calls": int(section["calls"]),
            "seconds": seconds / per,
            "share": seconds / total,
            "bytes": traffic / per,
            "achieved_gb_per_second": achieved / 1e9 if achieved is not None else None,
            "peak_fraction": None,
            "recoverable_seconds": None,
        }
        if peak_bytes_per_second:
            floor = traffic / peak_bytes_per_second
            row["peak_fraction"] = (achieved / peak_bytes_per_second) if achieved is not None else None
            row["recoverable_seconds"] = max(0.0, seconds - floor) / per
        rows.append(row)
    key = "recoverable_seconds" if peak_bytes_per_second else "seconds"
    rows.sort(key=lambda row: row[key], reverse=True)
    phase_bytes = sum(float(section["bytes"]) for section in sections)
    return {
        "section_seconds": total / per,
        "bytes": phase_bytes / per,
        "achieved_gb_per_second": phase_bytes / total / 1e9,
        "peak_fraction": (phase_bytes / total / peak_bytes_per_second) if peak_bytes_per_second else None,
        "ranked_by": key,
        "sections": rows,
    }


def attribute(profile: Mapping[str, Any], peak_bytes_per_second: float | None) -> dict[str, Any]:
    """Prefill (whole prompt) and decode (per token) attribution for a profile."""

    steps = int(profile.get("decode_steps", 0))
    if steps <= 0:
        raise AttributionError("profile has no decode steps")
    return {
        "backend": profile.get("backend"),
        "architecture": profile.get("architecture"),
        "peak_gb_per_second": peak_bytes_per_second / 1e9 if peak_bytes_per_second else None,
        "prefill": attribute_phase(profile["prefill"], peak_bytes_per_second),
        "decode_per_token": attribute_phase(profile["decode"], peak_bytes_per_second, per=steps),
    }


def render_markdown(attribution: Mapping[str, Any], limit: int = 6) -> str:
    """Top decode sections as a Markdown table."""

    decode = attribution["decode_per_token"]
    peak = attribution["peak_gb_per_second"]
    lines = [
        f"Decode per token: {1000 * decode['section_seconds']:.3f} ms of section time, "
        f"{decode['achieved_gb_per_second']:.1f} GB/s"
        + (f" ({100 * decode['peak_fraction']:.0f}% of the {peak:.0f} GB/s peak)" if peak else " (peak unknown)"),
        "",
        "| Section | Share | ms/token | GB/s | Peak % | Recoverable ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in decode["sections"][:limit]:
        gbps = "–" if row["achieved_gb_per_second"] is None else f"{row['achieved_gb_per_second']:.1f}"
        peak_pct = "–" if row["peak_fraction"] is None else f"{100 * row['peak_fraction']:.0f}%"
        recoverable = "–" if row["recoverable_seconds"] is None else f"{1000 * row['recoverable_seconds']:.3f}"
        lines.append(
            f"| {row['name']} | {100 * row['share']:.1f}% | {1000 * row['seconds']:.3f} | {gbps} | "
            f"{peak_pct} | {recoverable} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path, help="JSON written by `lokahi profile`")
    parser.add_argument("--chip", help="e.g. 'Apple M1'; omit to skip the bandwidth ceiling")
    parser.add_argument("--gpu-cores", type=int)
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    peak = peak_bandwidth(args.chip, args.gpu_cores) if args.chip else None
    result = attribute(profile, peak)
    print(render_markdown(result) if args.markdown else json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
