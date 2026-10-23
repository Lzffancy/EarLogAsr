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
from device_config import speaker_device

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
            turns.append([st, ed, lab])
    return [t for t in turns if t[1] - t[0] >= MIN_TURN_S]


def verify_assign(X, vp_path, thr):
    """Match each hop embedding against enrolled voiceprints (cosine)."""
    vp = json.load(open(vp_path, encoding="utf-8"))
    names = list(vp.keys())
    C = np.stack([vp[n]["centroid"] for n in names])
    sims = X @ C.T
    best = sims.argmax(axis=1)
    labels = np.array([names[b] if sims[i, b] >= thr else ""
                       for i, b in enumerate(best)], dtype=object)
    return labels, sims, names


def hybrid_labels(X, vp_path, vthr, cthr, n_speakers):
    """Enrolled speakers by verification; the rest fall back to clustering."""
    labels, sims, names = verify_assign(X, vp_path, vthr)
    mask = np.array([l == "" for l in labels])
    print(f"  verify: {(~mask).sum()}/{len(labels)} hops matched (thr={vthr})")
    for i, n in enumerate(names):
        s = sims[:, i]
        print(f"    {n}: sim p50={np.percentile(s, 50):.2f} "
              f"p90={np.percentile(s, 90):.2f} max={s.max():.2f} "
              f"hits={(labels == n).sum()}")
    if mask.any():
        sub = cluster(X[mask], cthr, n_speakers)
        for j, lab in zip(np.where(mask)[0], sub):
            labels[j] = f"C{int(lab):02d}"
    return labels


def write_turns(labels, total_s, out_prefix, threshold, elapsed, wav_path):
    labels = majority_smooth(labels, SMOOTH_K)
    turns = labels_to_turns(labels, HOP_S, total_s)
    dur = {}
    for st, ed, lab in turns:
        dur[lab] = dur.get(lab, 0) + (ed - st)
    # enrolled names (str) pass through; cluster ids get ranked.
    # cluster() returns ints; hybrid_labels() returns "Cxx" -- handle both,
    # otherwise single-speaker runs label turns "0" instead of "SPK00".
    def is_cluster(lab):
        return isinstance(lab, (int, np.integer)) or str(lab).startswith("C")
    cid_order = {lab: rank for rank, (lab, _) in enumerate(
        sorted(((l, d) for l, d in dur.items() if is_cluster(l)),
               key=lambda kv: -kv[1]))}
    def disp(lab):
        return f"SPK{cid_order[lab]:02d}" if is_cluster(lab) else str(lab)
    turns_out = [[int(st * 1000), int(ed * 1000), disp(lab)]
                 for st, ed, lab in turns]
    stats = {disp(l): round(d / 60, 1) for l, d in
             sorted(dur.items(), key=lambda kv: -kv[1])}
    os.makedirs(os.path.dirname(os.path.abspath(out_prefix)), exist_ok=True)
    with open(out_prefix + ".json", "w", encoding="utf-8") as f:
        json.dump({"file": wav_path, "elapsed_sec": elapsed,
                   "win_s": WIN_S, "hop_s": HOP_S, "threshold": threshold,
                   "n_speakers": len(dur), "speaker_minutes": stats,
                   "turns": turns_out}, f, ensure_ascii=False, indent=1)
    print(f"DONE speakers={len(dur)} turns={len(turns_out)} "
          f"elapsed={elapsed:.0f}s minutes={stats}")


def parse_verify(argv):
    """--verify voiceprints.json [vthr] -> (vp_path, vthr) or (None, None)."""
    if "--verify" not in argv:
        return None, None
    i = argv.index("--verify")
    vp = argv[i + 1]
    vthr = float(argv[i + 2]) if len(argv) > i + 2 and \
        not argv[i + 2].startswith("--") else 0.45
    return vp, vthr


def recluster(emb_path, out_prefix, threshold, vp_path=None, vthr=0.45):
    n_speakers = int(os.environ.get("EARLOG_N_SPK", "0")) or None
    X = np.load(emb_path)
    total_s = len(X) * HOP_S + WIN_S
    if vp_path:
        labels = hybrid_labels(X, vp_path, vthr, threshold, n_speakers)
    else:
        labels = cluster(X, threshold, n_speakers)
    write_turns(labels, total_s, out_prefix, threshold, 0, emb_path)


def _is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def positionals(argv):
    """argv[3:] minus flags and their values, in order. --verify eats one value,
    so `speaker_timeline.py a.wav out --verify vp.json` does not read vp.json as
    the clustering threshold."""
    out, skip = [], False
    for a in argv[3:]:
        if skip:
            skip = False
            continue
        if a.startswith("--"):
            skip = a == "--verify"
            continue
        out.append(a)
    return out


# ---------------- reusable core ----------------
# run_day.py imports these so a day of files shares one interpreter and one
# loaded CAM++ instance.


def load_wav16(wav_path):
    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return torchaudio.functional.resample(
        torch.from_numpy(audio).unsqueeze(0), sr, 16000)[0].numpy()


def embed(wav16, spk, verbose=True):
    """CAM++ embedding per hop. Returns X (n_hops, D), L2-normalised."""
    win, hop = int(WIN_S * 16000), int(HOP_S * 16000)
    starts = list(range(0, max(1, len(wav16) - win + 1), hop))
    if verbose:
        print(f"audio {len(wav16) / 16000:.0f}s, hops: {len(starts)}")
    embs = []
    for i, st in enumerate(starts):
        emb = spk.generate(input=wav16[st: st + win])[0]["spk_embedding"]
        if hasattr(emb, "detach"):
            emb = emb.detach().cpu()
        emb = np.asarray(emb, dtype=np.float32).flatten()
        embs.append(emb / (np.linalg.norm(emb) + 1e-8))
        if verbose and (i + 1) % 100 == 0:
            print(f"  emb {i + 1}/{len(starts)}")
    return np.stack(embs)


def embed_file(wav_path, spk, out_prefix, threshold, vp_path=None, vthr=0.45,
               n_speakers=None, verbose=True):
    """Speaker stage for one file: embed -> save npy -> label -> write json."""
    wav16 = load_wav16(wav_path)
    total_s = len(wav16) / 16000
    t0 = time.time()
    X = embed(wav16, spk, verbose)
    np.save(out_prefix + "_emb.npy", X)
    if vp_path:
        labels = hybrid_labels(X, vp_path, vthr, threshold, n_speakers)
    else:
        labels = cluster(X, threshold, n_speakers)
    elapsed = time.time() - t0
    if verbose:
        print(f"  emb done, clustering... ({elapsed:.0f}s)")
    write_turns(labels, total_s, out_prefix, threshold, elapsed, wav_path)
    return X, labels, elapsed


def main():
    vp_path, vthr = parse_verify(sys.argv)
    if sys.argv[1] == "--recluster":
        recluster(sys.argv[2], sys.argv[3],
                  float(sys.argv[4]) if len(sys.argv) > 4 and
                  not sys.argv[4].startswith("--") else 0.8,
                  vp_path, vthr or 0.45)
        return

    wav_path = sys.argv[1]
    out_prefix = sys.argv[2]
    pos = positionals(sys.argv)
    if "--threshold" in sys.argv:
        threshold = float(sys.argv[sys.argv.index("--threshold") + 1])
    else:
        threshold = float(pos[0]) if pos and _is_float(pos[0]) else 0.8
    n_speakers = int(os.environ.get("EARLOG_N_SPK", "0")) or None

    print(f"loading cam++... (device={speaker_device()})")
    spk = AutoModel(model="cam++", disable_update=True, device=speaker_device())
    embed_file(wav_path, spk, out_prefix, threshold, vp_path, vthr or 0.45,
               n_speakers)


if __name__ == "__main__":
    main()
