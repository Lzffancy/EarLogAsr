# -*- coding: utf-8 -*-
"""
Batch runner: three streams + fuse, one recording after another.

  python run_batch.py <wav_dir|wav_file> <out_dir> [options]

Options:
  --jobs N          slidewin window parallelism (default 1; GPU forces 1 anyway)
  --n-spk K         EARLOG_N_SPK passed to speaker_timeline (spectral clustering)
  --no-scene        skip the PANNs scene stream (saves ~40-100s / 30min audio)
  --no-spk          skip speaker stream
  --verify PATH     voiceprints.json -> verification mode (RECOMMENDED: pure
                    clustering splits one person into two on far-field audio)
  --vthr X          cosine threshold for --verify (default 0.45)
  --redo ASR|SPK|SCENE|FUSE   force re-run of a stage even if its output exists
  --glob PATTERN    file pattern inside the dir (default *.wav)

Each stage is skipped when its output already exists, so re-running after a
failure in the scene stream does not re-transcribe. Per-stage logs go to
<out_dir>/logs/<name>/<stage>.log, and a machine-readable summary to
<out_dir>/batch_summary.json.

Stage outputs (all under <out_dir>/):
  <name>_sw.json/.txt     content stream
  <name>_spk.json/_emb.npy speaker stream (npy kept for --recluster)
  <name>_sc.json          scene stream
  <name>_fused.jsonl/.txt fused timeline -- this is what the LLM layer reads
"""
import sys
import os
import json
import time
import glob
import argparse
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def run(cmd, log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as lf:
        lf.write("$ " + " ".join(cmd) + "\n\n")
        lf.flush()
        p = subprocess.run(cmd, cwd=HERE, stdout=lf, stderr=subprocess.STDOUT,
                           env=os.environ.copy(), text=True)
    return p.returncode, time.time() - t0, log_path


def tail(path, n=6):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = [l.rstrip() for l in f if l.strip()]
        return " | ".join(lines[-n:])[:400]
    except OSError:
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("out_dir")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--n-spk", type=int, default=0)
    ap.add_argument("--no-scene", action="store_true")
    ap.add_argument("--no-spk", action="store_true")
    ap.add_argument("--verify", help="voiceprints.json -> speaker verification mode")
    ap.add_argument("--vthr", type=float, default=None)
    ap.add_argument("--redo", default="", help="comma list: ASR,SPK,SCENE,FUSE")
    ap.add_argument("--glob", default="*.wav")
    args = ap.parse_args()

    redo = {s.strip().upper() for s in args.redo.split(",") if s.strip()}

    if os.path.isdir(args.source):
        wavs = sorted(glob.glob(os.path.join(args.source, args.glob)))
    else:
        wavs = [args.source]
    if not wavs:
        print(f"no wav matched: {args.source}/{args.glob}")
        return 2

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    if args.n_spk:
        os.environ["EARLOG_N_SPK"] = str(args.n_spk)

    print(f"{len(wavs)} recording(s) -> {out_dir}")
    summary = []
    t_all = time.time()

    for idx, wav in enumerate(wavs, 1):
        name = os.path.splitext(os.path.basename(wav))[0]
        base = os.path.join(out_dir, name)
        sw, spk, sc, fu = base + "_sw", base + "_spk", base + "_sc", base + "_fused"
        rec = {"file": wav, "name": name, "stages": {}}
        print(f"\n[{idx}/{len(wavs)}] {name}")

        # 1. content stream
        if "ASR" in redo or not os.path.exists(sw + ".json"):
            rc, dt, lp = run([PY, "slidewin_pipeline.py", wav, sw, str(args.jobs)],
                             os.path.join(out_dir, "logs", name, "asr.log"))
            rec["stages"]["asr"] = {"rc": rc, "sec": round(dt), "log": lp}
            print(f"  asr   {'OK ' if rc == 0 else 'FAIL'} {dt:.0f}s  {tail(lp)}")
            if rc != 0:
                summary.append(rec)
                continue
        else:
            print("  asr   cached")

        # 2. speaker stream
        if not args.no_spk:
            if "SPK" in redo or not os.path.exists(spk + ".json"):
                cmd = [PY, "speaker_timeline.py", wav, spk]
                if args.verify:
                    cmd += ["--verify", os.path.abspath(args.verify)]
                    if args.vthr is not None:
                        cmd.append(str(args.vthr))
                rc, dt, lp = run(cmd,
                                 os.path.join(out_dir, "logs", name, "spk.log"))
                rec["stages"]["spk"] = {"rc": rc, "sec": round(dt), "log": lp}
                print(f"  spk   {'OK ' if rc == 0 else 'FAIL'} {dt:.0f}s  {tail(lp)}")
            else:
                print("  spk   cached")

        # 3. scene stream
        if not args.no_scene:
            if "SCENE" in redo or not os.path.exists(sc + ".json"):
                rc, dt, lp = run([PY, "scene_timeline.py", wav, sc],
                                 os.path.join(out_dir, "logs", name, "scene.log"))
                rec["stages"]["scene"] = {"rc": rc, "sec": round(dt), "log": lp}
                print(f"  scene {'OK ' if rc == 0 else 'FAIL'} {dt:.0f}s  {tail(lp)}")
            else:
                print("  scene cached")

        # 4. fuse
        if "FUSE" in redo or not os.path.exists(fu + ".jsonl"):
            cmd = [PY, "fuse.py", fu, "--text", sw + ".json"]
            if not args.no_spk and os.path.exists(spk + ".json"):
                cmd += ["--spk", spk + ".json"]
            if not args.no_scene and os.path.exists(sc + ".json"):
                cmd += ["--scene", sc + ".json"]
            rc, dt, lp = run(cmd, os.path.join(out_dir, "logs", name, "fuse.log"))
            rec["stages"]["fuse"] = {"rc": rc, "sec": round(dt), "log": lp}
            print(f"  fuse  {'OK ' if rc == 0 else 'FAIL'} {dt:.0f}s  {tail(lp)}")
        else:
            print("  fuse  cached")

        # roll up stats from the fused jsonl
        try:
            rows = [json.loads(l) for l in
                    open(fu + ".jsonl", encoding="utf-8") if l.strip()]
            rec["rows"] = len(rows)
            rec["chars"] = sum(len(r["text"]) for r in rows)
            rec["low_conf"] = sum(1 for r in rows if r["conf"] == "low")
            rec["speakers"] = sorted({r["spk"] for r in rows if r["spk"]})
            rec["scenes"] = sorted({r["scene"] for r in rows if r["scene"]})
        except OSError:
            pass
        summary.append(rec)

    with open(os.path.join(out_dir, "batch_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"ran_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "elapsed_sec": round(time.time() - t_all),
                   "recordings": summary}, f, ensure_ascii=False, indent=1)

    print(f"\n=== total {time.time() - t_all:.0f}s, {len(summary)} done ===")
    for r in summary:
        print(f"{r['name']}  rows={r.get('rows', '?')} chars={r.get('chars', '?')} "
              f"low={r.get('low_conf', '?')} spk={len(r.get('speakers', []))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
