# -*- coding: utf-8 -*-
"""Second-channel review: re-transcribe low-confidence SenseVoice segments
with FireRedASR2-AED (better sentence quality + native timestamps).

Usage:
  python firered_review.py <wav> <sw_prefix> [out_prefix] [--ranges 24:00-30:00,...] [--all]

Low-confidence heuristics (on sw_xxx.json sentences):
  - top single-char ratio > 0.30   (stutter / repetition debris)
  - punctuation ratio     > 0.35   (fragmented output)
  - chars/sec < 1.0 over a >=8s span (sparse pickup on active speech)

FireRedASR2S is NOT vendored into this repo. Point to it via env:
  FIRERED_REPO     default ../FireRedASR2S
  FIRERED_AED_DIR  default ../models/models/xukaituo--FireRedASR2-AED/snapshots/master
  FIRERED_VAD_DIR  default ../models/models/xukaituo--FireRedVAD/snapshots/master/VAD
GPU is used automatically (fp16) when available, CPU (fp32) otherwise.
"""
import sys
import json
import time
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from device_config import on_gpu, describe

PUNCT = "。，、！？…—,.!?·~ "
GAP_MERGE_S = 15      # merge flagged records closer than this
MAX_SEG_S = 90        # split review segments longer than this
PAD_S = 1.0           # audio padding around each segment

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_ts(s):
    parts = [float(p) for p in s.split(":")]
    return sum(p * 60 ** i for i, p in enumerate(reversed(parts)))


def flag_low_conf(sent):
    text = sent["text"]
    body = text.strip(PUNCT)
    if not body:
        return False  # blank/degenerate windows carry no info worth re-doing
    punct_ratio = sum(text.count(c) for c in PUNCT) / max(1, len(text))
    top_ratio = max(body.count(c) for c in set(body)) / len(body)
    dur = max(0.1, sent["end_s"] - sent["start_s"])
    cps = len(body) / dur
    reasons = []
    if top_ratio > 0.30:
        reasons.append(f"repeat={top_ratio:.2f}")
    if punct_ratio > 0.35:
        reasons.append(f"punct={punct_ratio:.2f}")
    if cps < 1.0 and dur >= 8:
        reasons.append(f"sparse={cps:.1f}cps")
    return reasons


def merge_segments(flagged):
    """flagged: list of (start_s, end_s, [reasons]) -> merged segments."""
    segs = []
    for st, ed, reasons in flagged:
        if segs and st - segs[-1][1] <= GAP_MERGE_S:
            segs[-1][1] = ed
            segs[-1][2] |= set(reasons)
        else:
            segs.append([st, ed, set(reasons)])
    out = []
    for st, ed, reasons in segs:
        while ed - st > MAX_SEG_S:
            out.append((st, st + MAX_SEG_S, reasons))
            st += MAX_SEG_S
        out.append((st, ed, reasons))
    return [(max(0, st - PAD_S), ed + PAD_S, sorted(r)) for st, ed, r in out]


def load_firered():
    repo = os.environ.get("FIRERED_REPO", os.path.join(HERE, "..", "FireRedASR2S"))
    aed = os.environ.get("FIRERED_AED_DIR", os.path.join(
        HERE, "..", "models/models/xukaituo--FireRedASR2-AED/snapshots/master"))
    vad = os.environ.get("FIRERED_VAD_DIR", os.path.join(
        HERE, "..", "models/models/xukaituo--FireRedVAD/snapshots/master/VAD"))
    for p, name in [(repo, "FIRERED_REPO"), (aed, "FIRERED_AED_DIR")]:
        if not os.path.exists(p):
            sys.exit(f"missing {name}: {p}\n"
                     "setup: git clone https://github.com/FireRedTeam/FireRedASR2S.git\n"
                     "       modelscope download --model xukaituo/FireRedASR2-AED")
    # FireRedVAD default speech_threshold=0.4 drops the low-volume speech we
    # review; lower it (FIRERED_VAD_THRESHOLD, default 0.25). Segments where
    # AED still returns <sil> stay empty — that is a real recall limit, and
    # the SV text in the report is the fallback. FIRERED_REVIEW_VAD=0 turns
    # VAD off (better recall on borderline segments, but no sub-segment
    # timestamps).
    use_vad = os.environ.get("FIRERED_REVIEW_VAD", "1") == "1"
    vad_threshold = float(os.environ.get("FIRERED_VAD_THRESHOLD", "0.25"))
    if use_vad and not os.path.exists(vad):
        sys.exit(f"missing FIRERED_VAD_DIR: {vad}")
    sys.path.insert(0, os.path.abspath(repo))
    from fireredasr2s import FireRedAsr2System, FireRedAsr2SystemConfig
    from fireredasr2s.fireredasr2 import FireRedAsr2Config
    from fireredasr2s.fireredvad import FireRedVadConfig
    gpu = on_gpu()
    cfg = FireRedAsr2SystemConfig(
        vad_model_dir=vad,
        asr_type="aed",
        asr_model_dir=aed,
        asr_config=FireRedAsr2Config(use_gpu=gpu, use_half=gpu, beam_size=3),
        vad_config=FireRedVadConfig(use_gpu=gpu, speech_threshold=vad_threshold),
        asr_batch_size=1,
        enable_vad=use_vad,
        enable_lid=False,
        enable_punc=False,
    )
    return FireRedAsr2System(cfg)


def fmt(s):
    m, sec = divmod(s, 60)
    h, m = divmod(int(m), 60)
    return f"{h:02d}:{m:02d}:{sec:05.2f}"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    wav_path, sw_prefix = args[0], args[1]
    out_prefix = args[2] if len(args) > 2 else f"{sw_prefix}_review"
    review_all = "--all" in sys.argv
    manual = []
    for a in sys.argv:
        if a.startswith("--ranges="):
            for r in a.split("=", 1)[1].split(","):
                st, ed = r.split("-")
                manual.append((parse_ts(st), parse_ts(ed), ["manual"]))

    sw = json.load(open(f"{sw_prefix}.json", encoding="utf-8"))
    sentences = sw["sentences"]

    flagged = []
    for s in sentences:
        reasons = flag_low_conf(s)
        if reasons:
            flagged.append((s["start_s"], s["end_s"], reasons))
    segments = [] if review_all else merge_segments(flagged)
    if review_all:
        segments = merge_segments([(s["start_s"], s["end_s"], ["all"]) for s in sentences])
    # budget cap: review at most EARLOG_REVIEW_MAX_FRAC (default 20%) of audio;
    # keep highest-severity segments first (repeat > punct > sparse)
    if not review_all and segments:
        total_audio = sentences[-1]["end_s"]
        budget = total_audio * float(os.environ.get("EARLOG_REVIEW_MAX_FRAC", "0.2"))
        weight = {"repeat": 3, "punct": 2, "sparse": 1}
        scored = sorted(segments,
                        key=lambda s: -sum(weight.get(r.split("=")[0], 0) for r in s[2]))
        kept, used = [], 0.0
        for seg in scored:
            dur = seg[1] - seg[0]
            if used + dur > budget:
                continue
            kept.append(seg)
            used += dur
        if len(kept) < len(segments):
            print(f"budget cap {budget:.0f}s: kept {len(kept)}/{len(segments)} flagged segments")
        segments = kept
    segments += manual
    segments.sort()
    if not segments:
        print("no low-confidence segments flagged; nothing to review")
        return
    total_review = sum(ed - st for st, ed, _ in segments)
    print(f"device: {describe()}")
    print(f"flagged {len(flagged)} records -> {len(segments)} review segments "
          f"({total_review:.0f}s of {sw['sentences'][-1]['end_s']:.0f}s audio)")

    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    wav16 = torchaudio.functional.resample(
        torch.from_numpy(audio).unsqueeze(0), sr, 16000)[0].numpy()

    print("loading FireRedASR2-AED...")
    system = load_firered()
    import logging
    logging.getLogger("fireredasr2s.asr_system").setLevel(logging.WARNING)

    results = []
    t0_all = time.time()
    for k, (st, ed, reasons) in enumerate(segments):
        seg = wav16[int(st * 16000):int(ed * 16000)]
        seg_i16 = (np.clip(seg, -1, 1) * 32767).astype("int16")
        tmp = os.path.abspath(f"_review_tmp_{k}.wav")
        sf.write(tmp, seg_i16, 16000, subtype="PCM_16")
        t0 = time.time()
        res = system.process(tmp)
        os.unlink(tmp)
        sv_text = "".join(s["text"] for s in sentences
                          if s["end_s"] > st and s["start_s"] < ed)
        fr_sents = [{"start_s": round(st + s["start_ms"] / 1000, 2),
                     "end_s": round(st + s["end_ms"] / 1000, 2),
                     "text": s["text"]} for s in res["sentences"]]
        results.append({"start_s": st, "end_s": ed, "reasons": reasons,
                        "sv_text": sv_text, "fr_sentences": fr_sents,
                        "elapsed_s": round(time.time() - t0, 1)})
        print(f"  [{k + 1}/{len(segments)}] {fmt(st)}-{fmt(ed)} "
              f"({ed - st:.0f}s, {time.time() - t0:.0f}s, {','.join(reasons)})")

    with open(f"{out_prefix}.json", "w", encoding="utf-8") as f:
        json.dump({"wav": wav_path, "sw": sw_prefix, "segments": results},
                  f, ensure_ascii=False, indent=1)
    with open(f"{out_prefix}.txt", "w", encoding="utf-8") as f:
        for r in results:
            f.write(f"\n===== {fmt(r['start_s'])}-{fmt(r['end_s'])} "
                    f"[{','.join(r['reasons'])}] =====\nSV: {r['sv_text']}\nFR:\n")
            for s in r["fr_sentences"]:
                f.write(f"  [{fmt(s['start_s'])}-{fmt(s['end_s'])}] {s['text']}\n")
    print(f"DONE {len(results)} segments in {time.time() - t0_all:.0f}s -> {out_prefix}.json/.txt")


if __name__ == "__main__":
    main()
