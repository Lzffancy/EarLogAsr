# -*- coding: utf-8 -*-
"""Whole-day runner: one interpreter per stage, GPU and CPU stages in parallel.

  python run_day.py <wav_dir|wav> <out_dir> [options]

  --verify PATH     voiceprints.json -> speaker verification mode (recommended)
  --vthr X          cosine threshold for --verify (default 0.45)
  --glob PATTERN    file pattern inside the dir (default *.wav)
  --redo LIST       comma list of ASR,SPK,SCENE,FUSE to force re-run
  --no-spk          skip the speaker stream
  --no-scene        skip the PANNs scene stream
  --serial          run stages one after another (no parallel processes)

Why this exists: run_batch.py spawns 3-4 fresh interpreters per file, and each
pays ~28s to import torch plus ~12s to import funasr. Measured 2026-10-23:
a 30min file takes 68s asr wall time but only 16-18s of real inference, so a
16-file day burned ~36 min on repeated init alone.

Here each stage imports once and loads its model once, then loops the day's
files. The GPU stage (SenseVoice then PANNs) and the CPU stage (CAM++) run as
two concurrent processes, so the CPU-bound speaker stream overlaps GPU work.

Outputs match run_batch.py exactly (same suffixes, same fuse call), so the two
are interchangeable; run_batch.py stays for single-file debugging.
"""
import sys
import os
import json
import time
import glob
import argparse
import tempfile
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def collect(src, pattern):
    if os.path.isdir(src):
        return sorted(glob.glob(os.path.join(src, pattern)))
    return [src]


def stem(wav):
    return os.path.splitext(os.path.basename(wav))[0]


def base_of(out_dir, wav):
    return os.path.join(out_dir, stem(wav))


# ---------------- stage workers ----------------

def stage_gpu(cfg):
    """SenseVoice for every file, then PANNs for every file. One import."""
    import torch
    # Deliberately leave torch's thread count alone. Changing it changes float
    # reduction order and shifts transcript characters (measured 2026-10-23:
    # 4 threads vs 8 gave 48/55 identical segments).
    #
    # Note on reproducibility: the 2nd and later files in a batch differ from
    # a cold-start run of the same file by ~0.7% of characters (48/55 segments
    # identical). This is stable, not random -- three runs of this script agreed
    # 55/55 with each other. torch.use_deterministic_algorithms and
    # CUBLAS_WORKSPACE_CONFIG do NOT remove it, so they are not set here.

    from device_config import funasr_device, panns_device, describe
    from funasr import AutoModel
    import slidewin_pipeline as sw

    wavs = cfg["wavs"]
    redo = set(cfg["redo"])
    print(f"[gpu] device: {describe()}  files={len(wavs)}", flush=True)

    t_load = time.time()
    asr = AutoModel(model="iic/SenseVoiceSmall", disable_update=True,
                    device=funasr_device())
    print(f"[gpu] SenseVoice loaded in {time.time()-t_load:.1f}s", flush=True)

    for k, wav in enumerate(wavs, 1):
        op = base_of(cfg["out_dir"], wav) + "_sw"
        if os.path.exists(op + ".json") and "ASR" not in redo:
            print(f"[gpu] {k}/{len(wavs)} asr cached  {stem(wav)}", flush=True)
            continue
        t0 = time.time()
        wav16 = sw.load_wav16(wav)
        records, stats = sw.transcribe(wav16, asr, verbose=False)
        sw.write_outputs(op, wav, records, stats)
        print(f"[gpu] {k}/{len(wavs)} asr {time.time()-t0:5.1f}s "
              f"pieces={len(records)}  {stem(wav)}", flush=True)

    del asr
    torch.cuda.empty_cache()

    if cfg["no_scene"]:
        return

    from panns_inference import AudioTagging
    import scene_timeline as sc

    t_load = time.time()
    at = AudioTagging(checkpoint_path=None, device=panns_device())
    print(f"[gpu] PANNs loaded in {time.time()-t_load:.1f}s", flush=True)

    for k, wav in enumerate(wavs, 1):
        op = base_of(cfg["out_dir"], wav) + "_sc"
        if os.path.exists(op + ".json") and "SCENE" not in redo:
            print(f"[gpu] {k}/{len(wavs)} scene cached  {stem(wav)}", flush=True)
            continue
        t0 = time.time()
        sc.tag_file(wav, at, op, verbose=False)
        print(f"[gpu] {k}/{len(wavs)} scene {time.time()-t0:5.1f}s  {stem(wav)}",
              flush=True)


def stage_cpu(cfg):
    """CAM++ for every file. One import, one model load."""
    import torch
    # Same reason as stage_gpu: keep the default thread count.

    from device_config import speaker_device
    from funasr import AutoModel
    import speaker_timeline as spk

    wavs = cfg["wavs"]
    redo = set(cfg["redo"])
    n_speakers = int(os.environ.get("EARLOG_N_SPK", "0")) or None
    print(f"[cpu] device: {speaker_device()}  files={len(wavs)}", flush=True)

    t_load = time.time()
    model = AutoModel(model="cam++", disable_update=True,
                      device=speaker_device())
    print(f"[cpu] CAM++ loaded in {time.time()-t_load:.1f}s", flush=True)

    for k, wav in enumerate(wavs, 1):
        op = base_of(cfg["out_dir"], wav) + "_spk"
        if os.path.exists(op + ".json") and "SPK" not in redo:
            print(f"[cpu] {k}/{len(wavs)} spk cached  {stem(wav)}", flush=True)
            continue
        t0 = time.time()
        spk.embed_file(wav, model, op, cfg["threshold"], cfg["verify"],
                       cfg["vthr"], n_speakers, verbose=False)
        print(f"[cpu] {k}/{len(wavs)} spk {time.time()-t0:5.1f}s  {stem(wav)}",
              flush=True)


# ---------------- orchestrator ----------------

def run_fuse(wavs, out_dir, no_spk, no_scene, redo):
    for wav in wavs:
        b = base_of(out_dir, wav)
        fu = b + "_fused"
        if os.path.exists(fu + ".jsonl") and "FUSE" not in redo:
            print(f"  fuse cached  {stem(wav)}")
            continue
        cmd = [PY, "fuse.py", fu, "--text", b + "_sw.json"]
        if not no_spk and os.path.exists(b + "_spk.json"):
            cmd += ["--spk", b + "_spk.json"]
        if not no_scene and os.path.exists(b + "_sc.json"):
            cmd += ["--scene", b + "_sc.json"]
        p = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
        tag = "OK " if p.returncode == 0 else "FAIL"
        last = (p.stdout or p.stderr).strip().splitlines()[-1:] or [""]
        print(f"  fuse {tag} {stem(wav)}  {last[0][:110]}")
        if p.returncode != 0:
            print(p.stderr[-600:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", nargs="?", help="wav file or directory")
    ap.add_argument("out_dir", nargs="?")
    ap.add_argument("--verify", help="voiceprints.json -> verification mode")
    ap.add_argument("--vthr", type=float, default=0.45)
    ap.add_argument("--threshold", type=float, default=0.8,
                    help="clustering threshold for unverified hops")
    ap.add_argument("--glob", default="*.wav")
    ap.add_argument("--redo", default="")
    ap.add_argument("--no-spk", action="store_true")
    ap.add_argument("--no-scene", action="store_true")
    ap.add_argument("--serial", action="store_true")
    ap.add_argument("--stage", choices=["gpu", "cpu"], help=argparse.SUPPRESS)
    ap.add_argument("--manifest", help=argparse.SUPPRESS)
    args = ap.parse_args()

    # worker mode: read the manifest the parent wrote
    if args.stage:
        with open(args.manifest, encoding="utf-8") as f:
            cfg = json.load(f)
        cfg["verify"] = (os.path.abspath(cfg["verify"])
                         if cfg["verify"] else None)
        (stage_gpu if args.stage == "gpu" else stage_cpu)(cfg)
        return 0

    if not args.source or not args.out_dir:
        ap.error("need <source> and <out_dir>")

    wavs = collect(args.source, args.glob)
    if not wavs:
        print(f"no wav matched: {args.source}/{args.glob}")
        return 2
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    redo = {s.strip().upper() for s in args.redo.split(",") if s.strip()}

    print(f"{len(wavs)} recording(s) -> {out_dir}")
    print(f"redo={sorted(redo) or '-'} parallel={not args.serial}")
    t_all = time.time()

    cfg = {"wavs": wavs, "out_dir": out_dir, "verify": args.verify,
           "vthr": args.vthr, "threshold": args.threshold,
           "redo": sorted(redo), "no_spk": args.no_spk,
           "no_scene": args.no_scene}
    mf = os.path.join(tempfile.gettempdir(), f"earlog_manifest_{os.getpid()}.json")
    with open(mf, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False)

    stages = []
    if not args.serial:
        if not args.no_spk:
            stages.append(("cpu", ["--stage", "cpu", "--manifest", mf]))
        stages.append(("gpu", ["--stage", "gpu", "--manifest", mf]))
        procs = [(name, subprocess.Popen([PY, __file__] + extra, cwd=HERE))
                 for name, extra in stages]
        print(f"launched: {', '.join(n for n, _ in procs)}")
        for name, p in procs:
            p.wait()
            print(f"  stage {name} exit={p.returncode}")
    else:
        stage_gpu(cfg)
        if not args.no_spk:
            stage_cpu(cfg)

    print(f"\nstages done in {time.time() - t_all:.0f}s, fusing...")
    try:
        os.unlink(mf)
    except OSError:
        pass
    run_fuse(wavs, out_dir, args.no_spk, args.no_scene, redo)

    summary = []
    for wav in wavs:
        fu = base_of(out_dir, wav) + "_fused"
        rec = {"file": wav, "name": stem(wav)}
        try:
            rows = [json.loads(l) for l in open(fu + ".jsonl", encoding="utf-8")
                    if l.strip()]
            rec.update(rows=len(rows),
                       chars=sum(len(r["text"]) for r in rows),
                       low_conf=sum(1 for r in rows if r["conf"] == "low"),
                       speakers=sorted({r["spk"] for r in rows if r["spk"]}))
        except OSError:
            pass
        summary.append(rec)

    with open(os.path.join(out_dir, "day_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump({"ran_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "elapsed_sec": round(time.time() - t_all),
                   "parallel": not args.serial,
                   "recordings": summary}, f, ensure_ascii=False, indent=1)

    print(f"\n=== total {time.time() - t_all:.0f}s, {len(summary)} done ===")
    for r in summary:
        print(f"{r['name'][:40]:42s} rows={r.get('rows', '?')} "
              f"chars={r.get('chars', '?')} low={r.get('low_conf', '?')} "
              f"spk={len(r.get('speakers', []))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
