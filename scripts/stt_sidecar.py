#!/usr/bin/env python3
"""Parakeet-MLX sidecar — run under .venv-stt (Python 3.12 + Metal).

Usage:
  .venv-stt/bin/python scripts/stt_sidecar.py <audio> <out.json> [model_id]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: stt_sidecar.py <audio> <out.json> [model]", file=sys.stderr)
        return 2
    audio = Path(sys.argv[1])
    out = Path(sys.argv[2])
    model_id = sys.argv[3] if len(sys.argv) > 3 else "mlx-community/parakeet-tdt-0.6b-v3"

    from parakeet_mlx import from_pretrained

    model = from_pretrained(model_id)
    # Chunk long podcasts; overlap keeps sentence boundaries sane.
    result = model.transcribe(
        str(audio),
        chunk_duration=120.0,
        overlap_duration=15.0,
    )
    sentences = []
    for s in result.sentences:
        sentences.append(
            {
                "text": (s.text or "").strip(),
                "start": float(s.start),
                "end": float(s.end),
                "confidence": float(getattr(s, "confidence", 0.0) or 0.0),
            }
        )
    payload = {
        "model": model_id,
        "text": result.text or "",
        "sentences": sentences,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {len(sentences)} sentences -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
