#!/usr/bin/env python3
import json, os, platform, re, subprocess, sys, time
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

FLAGS = ["avx2", "avx512f", "avx512_vnni", "amx_int8",      # x86
         "asimddp", "i8mm", "sve", "sve2", "bf16"]          # arm

def cpu_info():
    info = {"machine": platform.machine(), "system": platform.system(),
            "python": platform.python_version(), "cores": os.cpu_count(),
            "model": platform.processor() or "?", "flags": []}
    try:
        txt = open("/proc/cpuinfo").read()
        m = re.search(r"^(?:model name|Model name)\s*:\s*(.+)$", txt, re.M)
        if m: info["model"] = m.group(1).strip()
        f = re.search(r"^(?:flags|Features)\s*:\s*(.+)$", txt, re.M)
        if f:
            have = set(f.group(1).split())
            info["flags"] = [x for x in FLAGS if x in have]
    except FileNotFoundError:
        try:                                                # Windows
            info["model"] = subprocess.check_output(
                ["wmic", "cpu", "get", "name"], text=True).split("\n")[1].strip()
        except Exception:
            pass
    return info

def bench(Interp, path, X, threads, warm=30, reps=400):
    it = Interp(model_path=path, num_threads=threads); it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    for k in range(warm):
        it.set_tensor(i["index"], X[k % len(X)]); it.invoke()
    ts, preds = [], []
    for k in range(reps):
        x = X[k % len(X)]
        t0 = time.perf_counter()
        it.set_tensor(i["index"], x); it.invoke(); out = it.get_tensor(o["index"])
        ts.append(time.perf_counter() - t0)
        preds.append(int(np.argmax(out)))
    ts = 1000 * np.array(ts)
    return {"median": round(float(np.median(ts)), 4),
            "mean":   round(float(ts.mean()), 4),
            "p10":    round(float(np.percentile(ts, 10)), 4),
            "p90":    round(float(np.percentile(ts, 90)), 4),
            "kB":     round(os.path.getsize(path) / 1024, 1),
            "preds":  preds[:len(X)]}

if __name__ == "__main__":
    Interp, pkg = get_interpreter()
    X = np.load("bench/samples.npy").astype("float32")[:, None][:, 0]
    X = [x[None] for x in X]                                # batch=1
    lab = np.load("bench/labels.npy")

    res = {"cpu": cpu_info(), "runtime": pkg, "n": len(X), "runs": {}}
    print(f"{res['cpu']['model']} | {res['cpu']['machine']} | "
          f"ядер {res['cpu']['cores']} | прапорці: {res['cpu']['flags'] or '—'}")
    print(f"інтерпретатор: {pkg}\n")

    for th in (1, os.cpu_count()):
        for tag, f in (("f32", "bench/final_f32.tflite"),
                       ("int8", "bench/final_int8.tflite")):
            r = bench(Interp, f, X, th)
            acc = float((np.array(r.pop("preds")) == lab).mean())
            r["acc"] = round(acc, 4)
            res["runs"][f"{tag}_t{th}"] = r
            print(f"{tag:4} потоків={th}  медіана {r['median']:7.4f} мс  "
                  f"[{r['p10']:.4f}–{r['p90']:.4f}]  {r['kB']:6.1f} КБ  acc={acc:.4f}")

    a, b = res["runs"]["f32_t1"]["median"], res["runs"]["int8_t1"]["median"]
    print(f"\nint8 / f32 при 1 потоці: {b/a:.3f}"
          f"  ({'квантування ПРИСКОРЮЄ' if b < a else 'квантування СПОВІЛЬНЮЄ'})")

    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "result.json"
    json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
    print("записано:", out)
