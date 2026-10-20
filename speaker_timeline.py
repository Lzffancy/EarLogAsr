# -*- coding: utf-8 -*-
"""
Speaker timeline: CAM++ embeddings on small sliding windows over the whole
recording -> cluster -> smooth -> speaker turns.

  python speaker_timeline.py <wav> <out_prefix> [dist_threshold]
  EARLOG_N_SPK=k forces spectral clustering with k speakers.

Output:
  <out_prefix>_emb.npy   embeddings per hop (for reclustering)
  <out_prefix>.json      {"hop_s", "win_s", "turns": [[st_ms, ed_ms, "SPK00"], ...]}
"""
import sys
import json
import time
import os

os.environ.setdefault("MODELSCOPE_CACHE", os.path.expanduser("~/.cache/modelscope"))

import numpy as np
import soundfile as sf
import torch
import torchaudio
from funasr import AutoModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from earlog_pipeline import cluster
from sensevoice_transcribe import fmt_ts

WIN_S = 4.0      # embedding window
HOP_S = 2.0      # hop
SMOOTH_K = 2     # majority filter half-width (2 -> 5 hops = 10s)
MIN_TURN_S = 1.0


def majority_smooth(labels, k):
    n = len(labels)
    out = labels.copy()
    for i in range(n):
        lo, hi = max(0, i - k), min(n, i + k + 1)
        vals, cnt = np.unique(labels[lo:hi], return_counts=True)
        if cnt.max() >= (hi - lo) // 2 + 1:
            out[i] = vals[cnt.argmax()]
    return out


def labels_to_turns(labels, hop_s, total_s):
    turns = []
    for i, lab in enumerate(labels):
        st, ed = i * hop_s, min(total_s, i * hop_s + WIN_S)
        if turns and turns[-1][2] == lab and st - turns[-1][1] <= hop_s:
            turns[-1][1] = ed
        else:
            turns.append([st, ed, int(lab)])
    return [t for t in turns if t[1] - t[0] >= MIN_TURN_S]


def write_turns(labels, total_s, out_prefix, threshold, elapsed, wav_path):
    labels = majority_smooth(labels, SMOOTH_K)
    turns = labels_to_turns(labels, HOP_S, total_s)
    dur = {}
    for st, ed, lab in turns:
        dur[lab] = dur.get(lab, 0) + (ed - st)
    order = {lab: rank for rank, (lab, _) in enumerate(
        sorted(dur.items(), key=lambda kv: -kv[1]))}
    turns_out = [[int(st * 1000), int(ed * 1000), f"SPK{order[lab]:02d}"]
                 for st, ed, lab in turns]
    stats = {f"SPK{order[l]:02d}": round(d / 60, 1) for l, d in dur.items()}
    with open(out_prefix + ".json", "w", encoding="utf-8") as f:
        json.dump({"file": wav_path, "elapsed_sec": elapsed,
                   "win_s": WIN_S, "hop_s": HOP_S, "threshold": threshold,
                   "n_speakers": len(dur), "speaker_minutes": stats,
                   "turns": turns_out}, f, ensure_ascii=False, indent=1)
    print(f"DONE speakers={len(dur)} turns={len(turns_out)} "
          f"elapsed={elapsed:.0f}s minutes={stats}")


def recluster(emb_path, out_prefix, threshold):
    n_speakers = int(os.environ.get("EARLOG_N_SPK", "0")) or None
    X = np.load(emb_path)
    total_s = len(X) * HOP_S + WIN_S
    labels = cluster(X, threshold, n_speakers)
    write_turns(labels, total_s, out_prefix, threshold, 0, emb_path)


def main():
    if sys.argv[1] == "--recluster":
        recluster(sys.argv[2], sys.argv[3],
                  float(sys.argv[4]) if len(sys.argv) > 4 else 0.8)
        return

    wav_path = sys.argv[1]
    out_prefix = sys.argv[2]
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.8
    n_speakers = int(os.environ.get("EARLOG_N_SPK", "0")) or None

    print("loading audio...")
    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    wav16 = torchaudio.functional.resample(
        torch.from_numpy(audio).unsqueeze(0), sr, 16000)[0].numpy()
    total_s = len(wav16) / 16000

    print("loading cam++...")
    spk = AutoModel(model="cam++", disable_update=True, device="cpu")

    win, hop = int(WIN_S * 16000), int(HOP_S * 16000)
    starts = list(range(0, max(1, len(wav16) - win + 1), hop))
    print(f"audio {total_s:.0f}s, hops: {len(starts)}")

    t0 = time.time()
    embs = []
    for i, st in enumerate(starts):
        seg = wav16[st: st + win]
        emb = spk.generate(input=seg)[0]["spk_embedding"]
        emb = np.asarray(emb, dtype=np.float32).flatten()
        embs.append(emb / (np.linalg.norm(emb) + 1e-8))
        if (i + 1) % 100 == 0:
            print(f"  emb {i + 1}/{len(starts)}")
    X = np.stack(embs)
    np.save(out_prefix + "_emb.npy", X)

    labels = cluster(X, threshold, n_speakers)
    elapsed = time.time() - t0
    print(f"  emb done, clustering... ({elapsed:.0f}s)")
    write_turns(labels, total_s, out_prefix, threshold, elapsed, wav_path)


if __name__ == "__main__":
    main()
