# -*- coding: utf-8 -*-
"""Enroll a speaker voiceprint from segments of a fused transcript.

  python enroll_speaker.py <name> <fused.jsonl> <SPKxx>
      average CAM++ embeddings of all rows labeled SPKxx -> voiceprints.json

  python enroll_speaker.py <name> <wav> <start_s> <end_s> [more ranges...]
      manual ranges, e.g. python enroll_speaker.py 李卓凡 ../x.wav 1080 1200

Voiceprints are L2-normalized centroid vectors stored in voiceprints.json
(next to where you run it). speaker_timeline.py --verify consumes them.
"""
import sys
import json
import os

os.environ.setdefault("MODELSCOPE_CACHE", os.path.expanduser("~/.cache/modelscope"))

import numpy as np
import soundfile as sf
import torch
import torchaudio
from funasr import AutoModel

MAX_SEG_S = 15.0     # cap per segment; long rows get truncated (centroid is stable)
MIN_SEG_S = 1.5      # skip fragments too short for a reliable embedding
VP_PATH = "voiceprints.json"


def load_wav16(wav_path):
    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return torchaudio.functional.resample(
        torch.from_numpy(audio).unsqueeze(0), sr, 16000)[0].numpy()


def segments_from_jsonl(jsonl_path, spk_label):
    rows = [json.loads(l) for l in open(jsonl_path, encoding="utf-8")]
    picked = [(r["start_ms"] / 1000, r["end_ms"] / 1000)
              for r in rows if (r.get("spk") or "").split("·")[0] == spk_label]
    if not picked:
        sys.exit(f"no rows with spk={spk_label} in {jsonl_path}")
    return picked


def main():
    name = sys.argv[1]
    if sys.argv[2].endswith(".jsonl"):
        jsonl_path, spk_label = sys.argv[2], sys.argv[3]
        # wav path lives in the sibling fused txt header; older runs wrote only
        # the basename there, so fall back to the sw json when that does not open
        import re
        head = open(jsonl_path.replace(".jsonl", ".txt"),
                    encoding="utf-8").readline()
        m = re.match(r"# (.+?)\s\s+fused timeline", head)
        wav_path = m.group(1) if m else sys.exit("wav not found in fused txt")
        if not os.path.exists(wav_path):
            sw = jsonl_path.replace("_fused.jsonl", "_sw.json")
            if os.path.exists(sw):
                wav_path = json.load(open(sw, encoding="utf-8"))["file"]
            else:
                sys.exit(f"wav not found: {wav_path}")
        ranges = segments_from_jsonl(jsonl_path, spk_label)
    else:
        wav_path = sys.argv[2]
        nums = [float(x) for x in sys.argv[3:]]
        ranges = list(zip(nums[::2], nums[1::2]))

    print(f"enrolling {name}: {len(ranges)} segments from {wav_path}")
    wav16 = load_wav16(wav_path)
    spk = AutoModel(model="cam++", disable_update=True, device="cpu")

    embs, used = [], 0.0
    for st, ed in ranges:
        ed = min(ed, st + MAX_SEG_S)
        if ed - st < MIN_SEG_S:
            continue
        seg = wav16[int(st * 16000):int(ed * 16000)]
        emb = spk.generate(input=seg)[0]["spk_embedding"]
        if hasattr(emb, "detach"):          # GPU: funasr returns a cuda tensor
            emb = emb.detach().cpu()
        emb = np.asarray(emb, dtype=np.float32).flatten()
        embs.append(emb / (np.linalg.norm(emb) + 1e-8))
        used += ed - st
    if not embs:
        sys.exit("no usable segments")
    centroid = np.mean(np.stack(embs), axis=0)
    centroid = centroid / np.linalg.norm(centroid)

    vp = {}
    if os.path.exists(VP_PATH):
        vp = json.load(open(VP_PATH, encoding="utf-8"))
    vp[name] = {"centroid": centroid.tolist(), "n_segments": len(embs),
                "seconds": round(used, 1), "source": wav_path}
    with open(VP_PATH, "w", encoding="utf-8") as f:
        json.dump(vp, f, ensure_ascii=False)
    print(f"DONE {name}: {len(embs)} segments, {used:.0f}s audio -> {VP_PATH}")


if __name__ == "__main__":
    main()
