import csv
import glob
import os
import platform
import re
import time
from datetime import datetime, timezone
from importlib.metadata import version

import numpy as np

try:
    from ai_edge_litert.interpreter import Interpreter
    runtime = "ai-edge-litert " + version("ai-edge-litert")
except ImportError:
    import tensorflow
    from tensorflow.lite import Interpreter
    runtime = "tensorflow " + tensorflow.__version__

FLAGS = ["avx2", "avx512f", "avx512_vnni", "amx_int8",
         "asimddp", "i8mm", "sve", "sve2", "bf16"]
WARMUP = 30
REPEATS = 400

cpu = platform.processor() or "unknown"
flags = []
try:
    import cpuinfo
    info = cpuinfo.get_cpu_info()
    cpu = info.get("brand_raw") or cpu
    have = {flag.replace("_", "").lower() for flag in info.get("flags", [])}
    flags = [flag for flag in FLAGS if flag.replace("_", "").lower() in have]
except Exception:
    pass

samples = np.load("bench/samples.npy").astype("float32")
labels = np.load("bench/labels.npy")
run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
name = re.sub(r"[^A-Za-z0-9]+", "-", cpu).strip("-")[:40]
paths = sorted(glob.glob("bench/*.tflite"))
print(cpu, platform.machine(), flags, runtime, len(paths), "files")

rows = []
accuracy = {}
for threads in range(1, (os.cpu_count() or 1) + 1):
    for path in paths:
        model, mode = os.path.basename(path)[:-len(".tflite")].rsplit("_", 1)
        interpreter = Interpreter(model_path=path, num_threads=threads)
        interpreter.allocate_tensors()
        details = interpreter.get_input_details()[0]
        output = interpreter.get_output_details()[0]
        scale, zero = details["quantization"]
        data = samples
        if details["dtype"] == np.int8:
            data = np.round(samples / scale + zero)
            data = np.clip(data, -128, 127).astype(np.int8)

        if path not in accuracy:
            correct = 0
            for k in range(len(data)):
                interpreter.set_tensor(details["index"], data[k][None])
                interpreter.invoke()
                predicted = interpreter.get_tensor(output["index"]).argmax()
                correct += int(predicted == labels[k])
            accuracy[path] = correct / len(data)

        times = []
        for k in range(WARMUP + REPEATS):
            interpreter.set_tensor(details["index"], data[k % len(data)][None])
            begin = time.perf_counter()
            interpreter.invoke()
            if k >= WARMUP:
                times.append(time.perf_counter() - begin)

        rows.append({"run": run, "platform": name, "cpu": cpu,
                     "arch": platform.machine(), "flags": ";".join(flags),
                     "runtime": runtime, "threads": threads, "model": model,
                     "mode": mode, "ms": round(1000 * np.median(times), 4),
                     "accuracy": round(accuracy[path], 4)})
        print(threads, model, mode, rows[-1]["ms"], rows[-1]["accuracy"])

filename = f"result-{name}-{run}.csv"
with open(filename, "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print("saved", filename)
