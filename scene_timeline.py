# -*- coding: utf-8 -*-
"""
Scene timeline: PANNs CNN14 audio tagging on sliding windows.

  python scene_timeline.py <wav> <out_prefix>

Output: <out_prefix>.json {"windows": [{"start_s","end_s","labels":[...],"probs":[...]}]}
"""
import sys
import json
import time
import os

import numpy as np
import soundfile as sf
import torch
import torchaudio

WIN_S = 10.0
HOP_S = 10.0
TOP_K = 3
MIN_PROB = 0.05

# AudioSet label ids we care about for life-logging scenes
KEEP = {
    "Silence", "Speech", "Conversation", "Music", "Vehicle", "Car", "Bus",
    "Train", "Subway, metro, underground", "Wind", "Rain", "Thunderstorm",
    "Keyboard", "Typing", "Footsteps", "Applause", "Laughter", "Dishes, cutlery",
    "Restaurant", "Water", "Bird", "Dog", "Alarm", "Ringing", "Telephone",
    "Door", "Street music", "Traffic noise, roadway noise", "Crowd", "Hubbub",
}


def main():
    wav_path = sys.argv[1]
    out_prefix = sys.argv[2]

    from panns_inference import AudioTagging, labels
    at = AudioTagging(checkpoint_path=None, device="cpu")

    print("loading audio...")
    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    wav32 = torchaudio.functional.resample(
        torch.from_numpy(audio).unsqueeze(0), sr, 32000)[0].numpy()
    total_s = len(wav32) / 32000

    win, hop = int(WIN_S * 32000), int(HOP_S * 32000)
    starts = list(range(0, max(1, len(wav32) - win + 1), hop))
    print(f"audio {total_s:.0f}s, windows: {len(starts)}")

    t0 = time.time()
    windows = []
    for i, st in enumerate(starts):
        seg = wav32[st: st + win]
        if len(seg) < 3200:
            continue
        clipwise, _ = at.inference(seg[None, :])
        probs = clipwise[0]
        top = probs.argsort()[::-1][:TOP_K]
        labs = [labels[j] for j in top if probs[j] >= MIN_PROB]
        labs = [l for l in labs if l in KEEP] or [labels[top[0]]]
        windows.append({
            "start_s": round(st / 32000, 2),
            "end_s": round((st + win) / 32000, 2),
            "labels": labs,
            "probs": [round(float(probs[labels.index(l)]), 3) for l in labs],
        })
        if (i + 1) % 30 == 0:
            print(f"  scene {i + 1}/{len(starts)}")

    elapsed = time.time() - t0
    with open(out_prefix + ".json", "w", encoding="utf-8") as f:
        json.dump({"file": wav_path, "elapsed_sec": elapsed,
                   "win_s": WIN_S, "hop_s": HOP_S, "windows": windows},
                  f, ensure_ascii=False, indent=1)

    from collections import Counter
    c = Counter(l for w in windows for l in w["labels"])
    print(f"DONE {wav_path} windows={len(windows)} elapsed={elapsed:.0f}s")
    print("label counts:", dict(c.most_common(10)))


if __name__ == "__main__":
    main()
