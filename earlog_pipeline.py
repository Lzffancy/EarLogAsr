# -*- coding: utf-8 -*-
"""
EarLog pipeline: VAD -> merge to chunks -> SenseVoice (text+emotion) + CAM++ embedding -> cluster.
Usage:
  python earlog_pipeline.py <wav> <out_prefix> [dist_threshold]   # full run, saves embeddings
  python earlog_pipeline.py --sweep <emb.npy>                     # threshold sweep on saved embeddings
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
from sensevoice_transcribe import parse_rich, fmt_ts

MERGE_GAP_MS = 400      # merge VAD segments with gap below this
MERGE_MAX_MS = 15000    # chunk max duration


def merge_segments(segments):
    chunks = []
    for st, ed in segments:
        if chunks and st - chunks[-1][1] < MERGE_GAP_MS and ed - chunks[-1][0] <= MERGE_MAX_MS:
            chunks[-1][1] = ed
        else:
            chunks.append([st, ed])
    return chunks


def cluster(X, threshold, n_speakers=None):
    from sklearn.cluster import AgglomerativeClustering, SpectralClustering
    if n_speakers:
        return SpectralClustering(
            n_clusters=n_speakers, affinity="cosine",
            random_state=0, assign_labels="kmeans",
        ).fit_predict(X)
    return AgglomerativeClustering(
        n_clusters=None, metric="cosine", linkage="average",
        distance_threshold=threshold,
    ).fit_predict(X)


def sweep(emb_path):
    X = np.load(emb_path)
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        labels = cluster(X, t)
        print(f"threshold={t:.1f}  n_speakers={labels.max()+1}")


def main():
    if sys.argv[1] == "--sweep":
        sweep(sys.argv[2])
        return

    wav_path = sys.argv[1]
    out_prefix = sys.argv[2]
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.8
    n_speakers = int(os.environ.get("EARLOG_N_SPK", "0")) or None

    print("loading audio...")
    audio, sr = sf.read(wav_path, dtype="float32")
    wav16 = torchaudio.functional.resample(torch.from_numpy(audio).unsqueeze(0), sr, 16000)[0].numpy()

    print("loading models...")
    vad = AutoModel(model="fsmn-vad", disable_update=True, device="cpu")
    asr = AutoModel(model="iic/SenseVoiceSmall", disable_update=True, device="cpu")
    spk = AutoModel(model="cam++", disable_update=True, device="cpu")

    t0 = time.time()
    segments = vad.generate(input=wav16)[0]["value"]
    chunks = merge_segments(segments)
    print(f"VAD segments: {len(segments)} -> chunks: {len(chunks)}")

    records = []
    embeddings = []
    for i, (st, ed) in enumerate(chunks):
        seg_audio = wav16[int(st * 16):int(ed * 16)]
        if len(seg_audio) < 1600:
            continue
        res = asr.generate(input=seg_audio, cache={}, language="zh", use_itn=True)
        parsed = parse_rich(res[0]["text"])
        text = "".join(p["text"] for p in parsed).strip()
        emotion = next((p["emotion"] for p in parsed if p.get("emotion")), None)
        event = next((p["event"] for p in parsed if p.get("event")), None)

        emb = spk.generate(input=seg_audio)[0]["spk_embedding"]
        emb = np.asarray(emb, dtype=np.float32).flatten()
        embeddings.append(emb / (np.linalg.norm(emb) + 1e-8))

        records.append({
            "start_ms": int(st), "end_ms": int(ed),
            "start": fmt_ts(st), "end": fmt_ts(ed),
            "text": text, "emotion": emotion, "event": event,
        })
        if (i + 1) % 20 == 0:
            print(f"  asr+spk {i+1}/{len(chunks)}")

    X = np.stack(embeddings)
    np.save(out_prefix + "_emb.npy", X)

    print("clustering speakers...")
    labels = cluster(X, threshold, n_speakers)
    dur = {}
    for lab, r in zip(labels, records):
        dur[lab] = dur.get(lab, 0) + (r["end_ms"] - r["start_ms"])
    order = {lab: rank for rank, (lab, _) in enumerate(
        sorted(dur.items(), key=lambda kv: -kv[1]))}
    for lab, r in zip(labels, records):
        r["speaker"] = f"SPK{order[lab]:02d}"

    elapsed = time.time() - t0
    n_spk = len(dur)
    stats = {f"SPK{order[l]:02d}": round(d / 60000, 1) for l, d in dur.items()}

    with open(out_prefix + ".json", "w", encoding="utf-8") as f:
        json.dump({"file": wav_path, "elapsed_sec": elapsed, "n_speakers": n_spk,
                   "threshold": threshold, "speaker_minutes": stats, "sentences": records},
                  f, ensure_ascii=False, indent=1)

    with open(out_prefix + ".txt", "w", encoding="utf-8") as f:
        f.write(f"# {os.path.basename(wav_path)}\n")
        f.write(f"# speakers={n_spk} minutes={stats} elapsed={elapsed:.0f}s\n\n")
        prev = None
        for r in records:
            if r["speaker"] != prev:
                f.write("\n")
            emo = f" ({r['emotion']})" if r["emotion"] and r["emotion"] != "NEUTRAL" else ""
            f.write(f"[{r['start']}] {r['speaker']}: {r['text']}{emo}\n")
            prev = r["speaker"]

    print(f"DONE {wav_path} chunks={len(records)} speakers={n_spk} elapsed={elapsed:.0f}s")
    print("speaker minutes:", stats)


if __name__ == "__main__":
    main()
