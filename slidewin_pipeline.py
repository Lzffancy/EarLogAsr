# -*- coding: utf-8 -*-
"""
No-VAD sliding-window pipeline: every sample of audio is transcribed, nothing cut.

  30s windows, 20s stride (10s overlap) -> SenseVoice per window
  -> overlap text alignment/dedup -> degenerate-output filter -> merged transcript

Usage:
  .venv/Scripts/python.exe slidewin_pipeline.py <wav> <out_prefix>
"""
import sys
import json
import time
import os
import difflib

os.environ.setdefault("MODELSCOPE_CACHE", os.path.expanduser("~/.cache/modelscope"))

import numpy as np
import soundfile as sf
import torch
import torchaudio
from funasr import AutoModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sensevoice_transcribe import parse_rich, fmt_ts

WIN_S = 30        # window length
STRIDE_S = 20     # stride (10s overlap)
MIN_ALIGN = 6     # min chars for overlap alignment
PUNCT = "。，、！？…—,.!?·~ "


def clean_text(parsed):
    return "".join(p["text"] for p in parsed).strip()


def is_degenerate(text):
    """punct-only or single-char loop ('看看看看...') => hallucination on noise."""
    body = text.strip(PUNCT)
    if not body:
        return True
    if len(body) >= 6:
        top = max(body.count(c) for c in set(body))
        if top / len(body) > 0.7:
            return True
    return False


def align_merge(accum, new_text):
    """Append only the non-overlapping part of new_text.
    Fuzzy suffix-prefix match: the same audio transcribed twice differs by a
    few chars, so accept a ratio threshold instead of exact match.
    Returns (new_part, overlap_len_in_new_text)."""
    if not accum or not new_text:
        return new_text, 0
    tail = accum[-120:]
    head = new_text[:180]
    best_k = 0
    max_k = min(len(tail), len(head))
    for k in range(max_k, MIN_ALIGN - 1, -1):
        sm = difflib.SequenceMatcher(None, tail[-k:], head[:k], autojunk=False)
        if sm.real_quick_ratio() < 0.75 or sm.quick_ratio() < 0.75:
            continue
        if sm.ratio() >= 0.70:
            best_k = k
            break
    if best_k:
        return new_text[best_k:], best_k
    return new_text, 0


def main():
    wav_path = sys.argv[1]
    out_prefix = sys.argv[2]

    print("loading audio...")
    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    wav16 = torchaudio.functional.resample(
        torch.from_numpy(audio).unsqueeze(0), sr, 16000)[0].numpy()
    total_s = len(wav16) / 16000

    print("loading model...")
    asr = AutoModel(model="iic/SenseVoiceSmall", disable_update=True, device="cpu")

    win, stride = WIN_S * 16000, STRIDE_S * 16000
    starts = list(range(0, max(1, len(wav16) - win + 1), stride))
    if len(wav16) - starts[-1] > 8 * 16000:   # tail worth transcribing
        starts.append(len(wav16) - win)
    print(f"audio {total_s:.0f}s, windows: {len(starts)}")

    t0 = time.time()
    records = []          # per-window kept pieces: {start_s, end_s, text}
    accum = ""
    n_blank = 0
    for i, st in enumerate(starts):
        seg = wav16[st: st + win]
        res = asr.generate(input=seg, cache={}, language="zh", use_itn=True)
        parsed = parse_rich(res[0]["text"])
        text = clean_text(parsed)
        emotion = next((p["emotion"] for p in parsed if p.get("emotion")), None)

        if is_degenerate(text):
            n_blank += 1
            if (i + 1) % 10 == 0:
                print(f"  win {i + 1}/{len(starts)} (blank={n_blank})")
            continue

        new_part, used = align_merge(accum, text)
        accum += new_part
        if new_part.strip(PUNCT):
            # time estimate: unconsumed part of this window maps linearly
            frac = used / max(1, len(text))
            est_start = st / 16000 + frac * (len(seg) / 16000)
            records.append({
                "start_s": round(est_start, 2),
                "end_s": round((st + len(seg)) / 16000, 2),
                "start": fmt_ts(est_start * 1000),
                "end": fmt_ts((st + len(seg)) / 16.0),
                "text": new_part.strip(),
                "emotion": emotion,
            })
        if (i + 1) % 10 == 0:
            print(f"  win {i + 1}/{len(starts)} (blank={n_blank})")

    elapsed = time.time() - t0

    with open(out_prefix + ".json", "w", encoding="utf-8") as f:
        json.dump({"file": wav_path, "elapsed_sec": elapsed,
                   "windows": len(starts), "blank_windows": n_blank,
                   "win_s": WIN_S, "stride_s": STRIDE_S,
                   "sentences": records}, f, ensure_ascii=False, indent=1)

    with open(out_prefix + ".txt", "w", encoding="utf-8") as f:
        f.write(f"# {os.path.basename(wav_path)}  no-VAD sliding window "
                f"({WIN_S}s/{STRIDE_S}s)\n")
        f.write(f"# windows={len(starts)} blank={n_blank} elapsed={elapsed:.0f}s\n\n")
        for r in records:
            emo = f" ({r['emotion']})" if r["emotion"] and r["emotion"] != "NEUTRAL" else ""
            f.write(f"[{r['start']}] {r['text']}{emo}\n")

    n_chars = sum(len(r["text"]) for r in records)
    print(f"DONE {wav_path} pieces={len(records)} chars={n_chars} "
          f"blank={n_blank} elapsed={elapsed:.0f}s "
          f"(rtf={elapsed / total_s:.3f})")


if __name__ == "__main__":
    main()
