# -*- coding: utf-8 -*-
"""SenseVoice-Small: ASR + emotion + audio event detection in one pass."""
import sys
import json
import re
import time
import os

os.environ.setdefault("MODELSCOPE_CACHE", os.path.expanduser("~/.cache/modelscope"))

from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from device_config import funasr_device

LANG_TAGS = {"zh", "en", "yue", "ja", "ko"}
EMOTION_TAGS = {"HAPPY", "SAD", "ANGRY", "NEUTRAL", "FEARFUL", "DISGUSTED", "SURPRISED", "EMO_UNKNOWN"}
EVENT_TAGS = {"Speech", "Applause", "Laughter", "Cry", "Sneeze", "Breath", "Cough", "Music",
              "Noise", "Silence", "BGM", "Dial", "Keyboard", "Door", "Alarm", "Ringing",
              "Car", "Engine", "Footsteps", "Water", "Wind", "Rain", "Bird", "Dog", "Cat",
              "Mosquito", "Babycry", "Break", "Glass", "Bell", "Heartbeat", "Typing",
              "Machine", "Thunder", "Explosion", "Gunshot", "Screen", "Silence", "Woof",
              "Meow", "Health", "Event_UNK", "Speech nosiy", "Speech noisy"}
SKIP_TAGS = {"withitn", "woitn", "itn"}


def parse_rich(text):
    """Parse SenseVoice rich tags: <|zh|><|HAPPY|><|Speech|><|withitn|>text ..."""
    segs = []
    cur = {"lang": None, "emotion": None, "event": None, "text": ""}
    pos = 0
    for m in re.finditer(r"<\|([^|]+)\|>", text):
        cur["text"] += text[pos:m.start()]
        tag = m.group(1)
        if cur["text"].strip():
            cur["text"] = cur["text"].strip()
            segs.append(cur)
            cur = {"lang": None, "emotion": None, "event": None, "text": ""}
        if tag in LANG_TAGS:
            cur["lang"] = tag
        elif tag in EMOTION_TAGS:
            cur["emotion"] = tag
        elif tag in SKIP_TAGS:
            pass
        else:
            cur["event"] = tag
        pos = m.end()
    cur["text"] += text[pos:]
    if cur["text"].strip():
        cur["text"] = cur["text"].strip()
        segs.append(cur)
    return segs


def fmt_ts(ms):
    s = ms / 1000.0
    m, s = divmod(s, 60)
    h, m = divmod(int(m), 60)
    return f"{h:02d}:{m:02d}:{s:05.2f}"


def main():
    wav_path = sys.argv[1]
    out_prefix = sys.argv[2]

    model = AutoModel(
        model="iic/SenseVoiceSmall",
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        device=funasr_device(),
        disable_update=True,
    )

    t0 = time.time()
    res = model.generate(
        input=wav_path,
        cache={},
        language="zh",
        use_itn=True,
        batch_size_s=60,
        merge_vad=True,
        merge_length_s=15,
    )
    elapsed = time.time() - t0

    item = res[0]
    with open(out_prefix + "_raw.json", "w", encoding="utf-8") as f:
        json.dump(item, f, ensure_ascii=False, indent=1, default=str)

    sentence_info = item.get("sentence_info") or []
    sentences = []
    if sentence_info:
        for s in sentence_info:
            for seg in parse_rich(s.get("text", "")):
                sentences.append({
                    "start_ms": s.get("start"), "end_ms": s.get("end"),
                    "start": fmt_ts(s["start"]), "end": fmt_ts(s["end"]),
                    **seg,
                })
    else:
        for seg in parse_rich(item.get("text", "")):
            sentences.append(seg)

    with open(out_prefix + ".json", "w", encoding="utf-8") as f:
        json.dump({"file": wav_path, "elapsed_sec": elapsed, "sentences": sentences},
                  f, ensure_ascii=False, indent=1)

    with open(out_prefix + ".txt", "w", encoding="utf-8") as f:
        f.write(f"# {os.path.basename(wav_path)}  SenseVoice-Small (inference {elapsed:.0f}s)\n\n")
        for s in sentences:
            ts = f"[{s['start']} --> {s['end']}] " if "start" in s else ""
            emo = f" ({s['emotion']})" if s.get("emotion") and s["emotion"] != "NEUTRAL" else ""
            evt = f" {{{s['event']}}}" if s.get("event") and s["event"] != "Speech" else ""
            lang = f" <{s['lang']}>" if s.get("lang") and s["lang"] != "zh" else ""
            f.write(f"{ts}{lang}{s['text']}{emo}{evt}\n")

    # stats
    from collections import Counter
    emos = Counter(s.get("emotion") for s in sentences if s.get("emotion"))
    evts = Counter(s.get("event") for s in sentences if s.get("event"))
    print(f"DONE {wav_path} segs={len(sentences)} elapsed={elapsed:.0f}s")
    print("emotions:", dict(emos))
    print("events:", dict(evts))


if __name__ == "__main__":
    main()
