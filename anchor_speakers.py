# -*- coding: utf-8 -*-
"""Anchor speaker labels to real names via self-introduction sentences.

Scans fused JSONL for patterns like "大家好我是X" / "我是那个X" / "我叫X",
writes speakers_registry.json: {file_basename: {SPKxx: {name, evidence, start, conf}}}.

  python anchor_speakers.py fused_a.jsonl [fused_b.jsonl ...]
  python anchor_speakers.py --apply   # also rewrite spk labels in the jsonl files

conf=high when the intro follows 大家好/各位老师 within a few chars;
conf=low otherwise (reported for manual confirmation, not auto-applied).
"""
import sys
import json
import re
import os

# two-pass: greeted pattern first (high conf), plain fallback (low conf).
# single regex with optional greeting would eat the greeting into the middle
# (leftmost match wins), so keep them separate.
GREETED = re.compile(
    r"(?:大家好|各位老师好?|老师们好?)[^。！？]{0,8}?"
    r"我(?:是|叫)(?:那个|这边是)?([一-龥]{2,4})(?=[，。,.!！呢啊])")
PLAIN = re.compile(
    r"我(?:是|叫)(?:那个|这边是)?([一-龥]{2,4})(?=[，。,.!！呢啊])")
STOPWORDS = {"第一个", "第二个", "第三个", "最后", "一名", "一名学", "学生",
             "选手", "汇报人"}


def source_wav(jsonl_path):
    """fused txt 的 header 第一行带原始 wav 名，用它做 registry key，
    和 fuse.py --names 的查找键保持一致。"""
    txt = jsonl_path.replace(".jsonl", ".txt")
    if os.path.exists(txt):
        head = open(txt, encoding="utf-8").readline()
        m = re.match(r"# (\S+\.wav)", head)
        if m:
            return m.group(1)
    return os.path.basename(jsonl_path)


def scan(jsonl_path):
    rows = [json.loads(l) for l in open(jsonl_path, encoding="utf-8")]
    hits = []
    for r in rows:
        text = r.get("text") or ""
        found = False
        for m in GREETED.finditer(text):
            name = m.group(1)
            if name in STOPWORDS:
                continue
            hits.append({"spk": r.get("spk"), "name": name, "conf": "high",
                         "start": r.get("start"),
                         "evidence": text[max(0, m.start() - 12):m.end() + 12]})
            found = True
        if found:
            continue
        for m in PLAIN.finditer(text):
            name = m.group(1)
            if name in STOPWORDS:
                continue
            hits.append({"spk": r.get("spk"), "name": name, "conf": "low",
                         "start": r.get("start"),
                         "evidence": text[max(0, m.start() - 12):m.end() + 12]})
    return hits


def main():
    apply = "--apply" in sys.argv
    paths = [a for a in sys.argv[1:] if a.endswith(".jsonl")]
    registry = {}
    if os.path.exists("speakers_registry.json"):
        registry = json.load(open("speakers_registry.json", encoding="utf-8"))

    for path in paths:
        file_key = source_wav(path)
        hits = scan(path)
        entry = registry.setdefault(file_key, {})
        for h in hits:
            if not h["spk"] or h["spk"] == "SPK??":
                continue
            cur = entry.get(h["spk"])
            # high conf wins; first high-conf hit sticks
            if cur and cur.get("conf") == "high" and h["conf"] == "low":
                continue
            if cur and cur.get("conf") == "high" and h["conf"] == "high":
                continue
            entry[h["spk"]] = {"name": h["name"], "conf": h["conf"],
                               "evidence": h["evidence"], "start": h["start"]}
        for spk, info in sorted(entry.items()):
            print(f"{file_key} {spk} -> {info['name']} ({info['conf']}) "
                  f"@ {info['start']}  …{info['evidence']}…")

    with open("speakers_registry.json", "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=1)
    print("-> speakers_registry.json")

    if apply:
        for path in paths:
            file_key = source_wav(path)
            fmap = registry.get(file_key, {})
            rows = [json.loads(l) for l in open(path, encoding="utf-8")]
            n = 0
            for r in rows:
                info = fmap.get(r.get("spk") or "")
                if info and info["conf"] == "high":
                    r["spk"] = f"{r['spk']}·{info['name']}"
                    n += 1
            with open(path, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"applied: {path} ({n} rows relabeled)")


if __name__ == "__main__":
    main()
