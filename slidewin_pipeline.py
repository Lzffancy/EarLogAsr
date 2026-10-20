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
from device_config import funasr_device, on_gpu, describe

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


# ---------------- parallel workers ----------------
# windows are independent: ASR runs in worker processes, merge stays serial

_G = {}


def _worker_init(npy_path, n_threads):
    torch.set_num_threads(n_threads)
    from funasr import AutoModel as AM
    _G["wav"] = np.load(npy_path, mmap_mode="r")
    _G["asr"] = AM(model="iic/SenseVoiceSmall", disable_update=True,
                   device=funasr_device())


def _worker_transcribe(task):
    i, st = task
    win = WIN_S * 16000
    seg = np.asarray(_G["wav"][st: st + win])
    res = _G["asr"].generate(input=seg, cache={}, language="zh", use_itn=True)
    return i, res[0]["text"]


def parallel_transcribe(wav16, starts, jobs):
    """Returns dict {window_index: raw_text}. Each worker loads its own model
    and reads audio via memmap."""
    import tempfile
    from multiprocessing import Pool
    n_threads = max(1, (os.cpu_count() or 8) // jobs)
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        tmp = f.name
    try:
        np.save(tmp, wav16)
        with Pool(jobs, initializer=_worker_init,
                  initargs=(tmp, n_threads)) as pool:
            out = {}
            done = 0
            for i, text in pool.imap_unordered(
                    _worker_transcribe, list(enumerate(starts)), chunksize=1):
                out[i] = text
                done += 1
                if done % 10 == 0:
                    print(f"  asr {done}/{len(starts)} (jobs={jobs})")
        return out
    finally:
        os.unlink(tmp)


def main():
    wav_path = sys.argv[1]
    out_prefix = sys.argv[2]
    jobs = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    if on_gpu() and jobs > 1:
        print("GPU mode: forcing jobs=1 (multi-process would contend for one GPU)")
        jobs = 1
    print(f"device: {describe()}")

    print("loading audio...")
    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    wav16 = torchaudio.functional.resample(
        torch.from_numpy(audio).unsqueeze(0), sr, 16000)[0].numpy()
    total_s = len(wav16) / 16000

    win, stride = WIN_S * 16000, STRIDE_S * 16000
    starts = list(range(0, max(1, len(wav16) - win + 1), stride))
    if len(wav16) - starts[-1] > 8 * 16000:   # tail worth transcribing
        starts.append(len(wav16) - win)
    print(f"audio {total_s:.0f}s, windows: {len(starts)}, jobs={jobs}")

    # energy gate: skip windows that are true silence (not VAD — raw RMS only)
    RMS_SKIP = 1e-3
    active, n_silence = [], 0
    for i, st in enumerate(starts):
        seg = wav16[st: st + win]
        if float(np.sqrt(np.mean(seg ** 2))) < RMS_SKIP:
            n_silence += 1
        else:
            active.append((i, st))
    starts_active = [st for _, st in active]
    print(f"energy gate: skipped {n_silence} silent windows")

    t0 = time.time()
    if jobs > 1:
        raw = parallel_transcribe(wav16, starts_active, jobs)
    else:
        print("loading model...")
        asr = AutoModel(model="iic/SenseVoiceSmall", disable_update=True,
                        device=funasr_device())
        raw = {}
        for k, (i, st) in enumerate(active):
            res = asr.generate(input=wav16[st: st + win], cache={},
                               language="zh", use_itn=True)
            raw[i] = res[0]["text"]
            if (k + 1) % 10 == 0:
                print(f"  asr {k + 1}/{len(active)}")

    # serial merge, window order preserved
    records = []
    accum = ""
    n_blank = 0
    for i, st in active:
        seg = wav16[st: st + win]
        parsed = parse_rich(raw[i])
        text = clean_text(parsed)
        emotion = next((p["emotion"] for p in parsed if p.get("emotion")), None)

        if is_degenerate(text):
            n_blank += 1
            continue

        new_part, used = align_merge(accum, text)
        accum += new_part
        if new_part.strip(PUNCT):
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

    elapsed = time.time() - t0

    with open(out_prefix + ".json", "w", encoding="utf-8") as f:
        json.dump({"file": wav_path, "elapsed_sec": elapsed,
                   "windows": len(starts), "blank_windows": n_blank,
                   "silent_windows_skipped": n_silence, "jobs": jobs,
                   "win_s": WIN_S, "stride_s": STRIDE_S,
                   "sentences": records}, f, ensure_ascii=False, indent=1)

    with open(out_prefix + ".txt", "w", encoding="utf-8") as f:
        f.write(f"# {os.path.basename(wav_path)}  no-VAD sliding window "
                f"({WIN_S}s/{STRIDE_S}s, jobs={jobs})\n")
        f.write(f"# windows={len(starts)} silent_skipped={n_silence} "
                f"blank={n_blank} elapsed={elapsed:.0f}s\n\n")
        for r in records:
            emo = f" ({r['emotion']})" if r["emotion"] and r["emotion"] != "NEUTRAL" else ""
            f.write(f"[{r['start']}] {r['text']}{emo}\n")

    n_chars = sum(len(r["text"]) for r in records)
    print(f"DONE {wav_path} pieces={len(records)} chars={n_chars} "
          f"blank={n_blank} elapsed={elapsed:.0f}s "
          f"(rtf={elapsed / total_s:.3f})")


if __name__ == "__main__":
    main()
