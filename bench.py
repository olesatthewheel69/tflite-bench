#!/usr/bin/env python3
import glob, json, os, platform, re, subprocess, sys, time
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

def cpu_info():
    d = {"machine": platform.machine(), "system": platform.system(),
         "cores": os.cpu_count(), "model": platform.processor() or "?", "flags": []}
    nz = lambda s: s.replace("_", "").lower()
    want = {nz(f): f for f in FLAGS}
    try:
        import cpuinfo                                  # працює і на Windows, і на Linux
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
    if dt in (np.int8, np.uint8):                      # вхід квантований
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
    models = {os.path.basename(p)[6:-7]: p                       # final_<tag>.tflite
              for p in sorted(glob.glob("bench/final_*.tflite"))}
    if not models: sys.exit("не знайдено bench/final_*.tflite")

    ci = cpu_info()
    print(f"{ci['model']} | {ci['machine']} | ядер {ci['cores']} | "
          f"прапорці: {ci['flags'] or '—'}\nінтерпретатор: {pkg}")
    print(f"моделі: {', '.join(models)}\n")

    res, ref = {"cpu": ci, "runtime": pkg, "runs": {}}, None
    for th in (1, os.cpu_count()):
        print(f"--- потоків = {th} ---")
        for tag, path in models.items():
            r, pr = run(Interp, path, X, th)
            if tag == "f32": ref = pr
            r["acc"] = round(float((pr == lab).mean()), 4)
            r["agree_f32"] = round(float((pr == ref).mean()), 4) if ref is not None else None
            res["runs"][f"{tag}_t{th}"] = r
            print(f"{tag:10} {r['in_dtype']:>7}  медіана {r['median']:7.4f} мс "
                  f"[{r['p10']:.4f}–{r['p90']:.4f}]  {r['kB']:6.1f} КБ  "
                  f"acc={r['acc']:.4f}  згода={r['agree_f32']}")
        b = res["runs"][f"f32_t{th}"]["median"]
        for tag in models:
            v = res["runs"][f"{tag}_t{th}"]["median"]
            print(f"    {tag:10} / f32 = {v/b:.3f}" + (f"  (×{b/v:.2f})" if v < b else ""))
        print()

    name = re.sub(r"[^A-Za-z0-9]+", "-", ci["model"])[:40]
    out = sys.argv[sys.argv.index("--out")+1] if "--out" in sys.argv else f"result-{name}.json"
    json.dump(res, open(out, "w"), ensure_ascii=False, indent=1); print("записано:", out)
