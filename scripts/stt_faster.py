"""Transcribe with faster-whisper (Linux / N100-friendly CPU backend).

Same CLI contract as stt_sidecar.py (Parakeet/Mac):
    python stt_faster.py <audio> <dest.json> [model]

Writes {"sentences": [{"start": s, "end": e, "text": t}, ...]}.
Model is a whisper size (tiny/base/small/...) or a CTranslate2 path.
Default `base` balances accuracy and N100 speed (~5-15 min per hour of audio).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def words_to_sentences(words: list[dict]) -> list[dict]:
    """Group word timestamps into sentence spans on end punctuation."""
    sentences: list[dict] = []
    buf: list[dict] = []

    def flush() -> None:
        if not buf:
            return
        text = " ".join(w["word"] for w in buf).strip()
        if text:
            sentences.append({"start": buf[0]["start"], "end": buf[-1]["end"], "text": text})
        buf.clear()

    for w in words:
        buf.append(w)
        if w["word"].rstrip().endswith((".", "?", "!")) and len(buf) >= 3:
            flush()
    flush()
    return sentences


def main() -> None:
    if len(sys.argv) < 3:
        print("usage: stt_faster.py <audio> <dest.json> [model]", file=sys.stderr)
        raise SystemExit(2)
    audio = Path(sys.argv[1])
    dest = Path(sys.argv[2])
    model_name = sys.argv[3] if len(sys.argv) > 3 else "base"

    from faster_whisper import WhisperModel

    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments, _info = model.transcribe(
        str(audio),
        beam_size=1,  # greedy: fastest, plenty for ad-boundary timestamps
        word_timestamps=True,
        vad_filter=True,
    )
    words: list[dict] = []
    for seg in segments:
        if not seg.words:
            text = (seg.text or "").strip()
            if text:
                words.append({"start": seg.start, "end": seg.end, "word": text})
            continue
        for w in seg.words:
            words.append({"start": w.start, "end": w.end, "word": w.word.strip()})
    words = [w for w in words if w["word"]]

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        json.dumps(
            {"sentences": words_to_sentences(words), "backend": "faster-whisper", "model": model_name},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"faster-whisper({model_name}): {len(words)} words -> {dest}")


if __name__ == "__main__":
    main()
