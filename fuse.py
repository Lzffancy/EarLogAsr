# -*- coding: utf-8 -*-
"""
Fuse ASR layer streams into the unified timeline JSONL.

  python fuse.py <out_prefix> --text sw_xxx.json [--spk spk_xxx.json] [--scene sc_xxx.json]

JSONL row: {"start_ms","end_ms","spk","scene","text","emotion","conf"}
conf: "low" when text body < 4 chars (likely hallucination on noise).
Also writes a readable <out_prefix>.txt grouped by speaker turns.
"""
import sys
import json
import argparse
import os

PUNCT = "。，、！？…—,.!?·~ "


def overlap_s(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def assign_speaker(st_s, ed_s, turns):
    best, best_ov = None, 0.0
    for t0, t1, lab in turns:
        ov = overlap_s(st_s, ed_s, t0 / 1000, t1 / 1000)
        if ov > best_ov:
            best, best_ov = lab, ov
    # require the turn to cover at least 30% of the text span
    if best and best_ov >= 0.3 * (ed_s - st_s):
        return best
    return "SPK??"


def assign_scene(st_s, ed_s, scene_windows):
    if not scene_windows:
        return None
    votes = {}
    for w in scene_windows:
        ov = overlap_s(st_s, ed_s, w["start_s"], w["end_s"])
        if ov <= 0:
            continue
        for lab in w["labels"][:2]:
            votes[lab] = votes.get(lab, 0) + ov
    if not votes:
        return None
    return max(votes, key=votes.get)


def apply_review(rows, review, turns, scene_windows):
    """Splice FireRed review results back into the timeline.

    For each reviewed segment with FR output: rows fully inside the segment
    are replaced by FR sentences (precise timestamps, conf="reviewed").
    Segments where FR came back empty keep their original SV rows.
    Returns (rows, n_replaced_segments).
    """
    segs = [s for s in review["segments"] if s["fr_sentences"]]
    if not segs:
        return rows, 0
    out = []
    for r in rows:
        rs, re_ = r["start_ms"] / 1000, r["end_ms"] / 1000
        if any(s["start_s"] <= rs and re_ <= s["end_s"] for s in segs):
            continue          # swallowed by a reviewed segment
        out.append(r)
    for s in segs:
        for fs in s["fr_sentences"]:
            st_s, ed_s = fs["start_s"], fs["end_s"]
            out.append({
                "start_ms": int(st_s * 1000), "end_ms": int(ed_s * 1000),
                "start": fmt_ms(st_s * 1000), "end": fmt_ms(ed_s * 1000),
                "spk": assign_speaker(st_s, ed_s, turns) if turns else None,
                "scene": assign_scene(st_s, ed_s, scene_windows),
                "text": fs["text"],
                "emotion": None,
                "conf": "reviewed",
            })
    out.sort(key=lambda r: r["start_ms"])
    return out, len(segs)


def fmt_ms(ms):
    s = ms / 1000.0
    m, s = divmod(s, 60)
    h, m = divmod(int(m), 60)
    return f"{h:02d}:{m:02d}:{s:05.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_prefix")
    ap.add_argument("--text", required=True)
    ap.add_argument("--spk")
    ap.add_argument("--scene")
    ap.add_argument("--review", help="firered_review 产物 json，回灌低置信段")
    ap.add_argument("--names", help="speakers_registry.json，SPK→姓名")
    args = ap.parse_args()

    content = json.load(open(args.text, encoding="utf-8"))
    sentences = content["sentences"]

    turns = []
    if args.spk:
        turns = json.load(open(args.spk, encoding="utf-8"))["turns"]
    scene_windows = None
    if args.scene:
        scene_windows = json.load(open(args.scene, encoding="utf-8"))["windows"]

    rows = []
    for s in sentences:
        st_s, ed_s = s["start_s"], s["end_s"]
        body = s["text"].strip(PUNCT)
        rows.append({
            "start_ms": int(st_s * 1000),
            "end_ms": int(ed_s * 1000),
            "start": s["start"], "end": s["end"],
            "spk": assign_speaker(st_s, ed_s, turns) if turns else None,
            "scene": assign_scene(st_s, ed_s, scene_windows),
            "text": s["text"],
            "emotion": s.get("emotion"),
            "conf": "low" if len(body) < 4 else "normal",
        })

    if args.review:
        review = json.load(open(args.review, encoding="utf-8"))
        rows, n_rep = apply_review(rows, review, turns, scene_windows)
        print(f"review applied: {n_rep} segments spliced in")

    if args.names:
        reg = json.load(open(args.names, encoding="utf-8"))
        fmap = reg.get(os.path.basename(content["file"]), {})
        for r in rows:
            info = fmap.get(r["spk"] or "")
            if info and info.get("name") and info.get("conf") == "high":
                r["spk"] = f"{r['spk']}·{info['name']}"

    with open(args.out_prefix + ".jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # readable view: group consecutive rows by speaker
    with open(args.out_prefix + ".txt", "w", encoding="utf-8") as f:
        src = os.path.basename(content["file"])
        f.write(f"# {src}  fused timeline\n")
        f.write(f"# rows={len(rows)} "
                f"low_conf={sum(1 for r in rows if r['conf'] == 'low')}\n\n")
        cur_spk, buf, buf_start = None, [], None
        def flush():
            if not buf:
                return
            tag = f" {{{cur_scene}}} " if cur_scene else " "
            f.write(f"\n[{buf_start}]{tag}{cur_spk or ''}\n")
            f.write("".join(buf) + "\n")
        cur_scene = None
        for r in rows:
            if r["spk"] != cur_spk or r["scene"] != cur_scene:
                flush()
                cur_spk, cur_scene = r["spk"], r["scene"]
                buf, buf_start = [], r["start"]
            mark = "〈?〉" if r["conf"] == "low" else \
                   ("〈核〉" if r["conf"] == "reviewed" else "")
            buf.append(r["text"] + mark)
        flush()

    n_low = sum(1 for r in rows if r["conf"] == "low")
    spk_min = {}
    for r in rows:
        if r["spk"]:
            spk_min[r["spk"]] = spk_min.get(r["spk"], 0) + (r["end_ms"] - r["start_ms"]) / 60000
    spk_min = {k: round(v, 1) for k, v in sorted(spk_min.items(), key=lambda kv: -kv[1])}
    print(f"DONE rows={len(rows)} low_conf={n_low} speakers={spk_min}")


if __name__ == "__main__":
    main()
