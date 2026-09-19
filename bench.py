#!/usr/bin/env python3
import glob, json, os, platform, re, sys, time
import numpy as np

def get_interpreter():
    for mod, pkg in (("ai_edge_litert.interpreter", "ai-edge-litert"),
                     ("tflite_runtime.interpreter", "tflite-runtime"),
                     ("tensorflow.lite",            "tensorflow")):
        try:
            return __import__(mod, fromlist=["Interpreter"]).Interpreter, pkg
        except Exception:
            continue
    sys.exit("не знайдено інтерпретатора TFLite")

FLAGS = ["avx2", "avx512f", "avx512_vnni", "amx_int8",
         "asimddp", "i8mm", "sve", "sve2", "bf16"]
MODES = ["f32", "dynrange", "fp16", "int8"]

def cpu_info():
    d = {"machine": platform.machine(), "system": platform.system(),
         "cores": os.cpu_count(), "model": platform.processor() or "?", "flags": []}
    nz = lambda s: s.replace("_", "").lower()
    want = {nz(f): f for f in FLAGS}
    try:
        import cpuinfo
        i = cpuinfo.get_cpu_info()
        d["model"] = i.get("brand_raw") or d["model"]
        have = {nz(x) for x in i.get("flags", [])}
        d["flags"] = [want[k] for k in want if k in have]
        return d
    except Exception:
        pass
    try:
        txt = open("/proc/cpuinfo").read()
        m = re.search(r"^(?:model name|Model name)\s*:\s*(.+)$", txt, re.M)
        if m: d["model"] = m.group(1).strip()
        f = re.search(r"^(?:flags|Features)\s*:\s*(.+)$", txt, re.M)
        if f:
            have = {nz(x) for x in f.group(1).split()}
            d["flags"] = [want[k] for k in want if k in have]
    except FileNotFoundError:
        pass
    return d

def run(Interp, path, X, threads, warm=30, reps=400):
    it = Interp(model_path=path, num_threads=threads); it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    dt, (sc, zp) = i["dtype"], i["quantization"]
    if dt in (np.int8, np.uint8):
        lo, hi = np.iinfo(dt).min, np.iinfo(dt).max
        Xq = [np.clip(np.round(x/sc + zp), lo, hi).astype(dt) for x in X]
    else:
        Xq = [x.astype(dt) for x in X]
    for k in range(warm):
        it.set_tensor(i["index"], Xq[k % len(Xq)]); it.invoke()
    ts, preds = [], []
    for k in range(reps):
        x = Xq[k % len(Xq)]
        t0 = time.perf_counter()
        it.set_tensor(i["index"], x); it.invoke(); out = it.get_tensor(o["index"])
        ts.append(time.perf_counter() - t0)
        preds.append(int(np.argmax(out)))
    ts = 1000 * np.array(ts)
    return (dict(median=round(float(np.median(ts)), 4), mean=round(float(ts.mean()), 4),
                 p10=round(float(np.percentile(ts, 10)), 4),
                 p90=round(float(np.percentile(ts, 90)), 4),
                 kB=round(os.path.getsize(path)/1024, 1),
                 in_dtype=str(np.dtype(dt))),
            np.array(preds[:len(X)]))

if __name__ == "__main__":
    Interp, pkg = get_interpreter()
    X = [x[None] for x in np.load("bench/samples.npy").astype("float32")]
    lab = np.load("bench/labels.npy")

    found = {}                                      # {модель: {варіант: шлях}}
    for p in sorted(glob.glob("bench/*.tflite")):
        stem = os.path.basename(p)[:-7]
        if "_" not in stem: continue
        name, mode = stem.rsplit("_", 1)
        found.setdefault(name, {})[mode] = p
    if not found: sys.exit("не знайдено bench/*.tflite")

    ci = cpu_info()
    print(f"{ci['model']} | {ci['machine']} | ядер {ci['cores']} | "
          f"прапорці: {ci['flags'] or '—'}\nінтерпретатор: {pkg}")
    print(f"моделей: {len(found)}, файлів: {sum(len(v) for v in found.values())}\n")

    threads = list(range(1, (os.cpu_count() or 1) + 1))
    res = {"cpu": ci, "runtime": pkg, "runs": {}}

    for th in threads:
        print(f"--- потоків = {th} ---")
        for name in sorted(found):
            ref = None
            for mode in [m for m in MODES if m in found[name]]:
                r, pr = run(Interp, found[name][mode], X, th)
                if mode == "f32": ref = pr
                r.update(model=name, mode=mode, threads=th,
                         acc=round(float((pr == lab).mean()), 4),
                         agree_f32=round(float((pr == ref).mean()), 4)
                                   if ref is not None else None)
                base = res["runs"].get(f"{name}_f32_t{th}")
                r["vs_f32"] = round(base["median"]/r["median"], 3) if base else 1.0
                res["runs"][f"{name}_{mode}_t{th}"] = r
                print(f"{name:8} {mode:9} {r['in_dtype']:>7} "
                      f"{r['median']:8.4f} мс  {r['kB']:7.1f} КБ  "
                      f"acc={r['acc']:.4f}  згода={r['agree_f32']}  "
                      f"×{r['vs_f32']:.2f}")
        print()

    name = re.sub(r"[^A-Za-z0-9]+", "-", ci["model"])[:40]
    out = sys.argv[sys.argv.index("--out")+1] if "--out" in sys.argv else f"result-{name}.json"
    json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
    print("записано:", out)
