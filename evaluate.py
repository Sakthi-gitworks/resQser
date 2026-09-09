"""
evaluate.py — does the new model actually separate crashes from ordinary photos?

Runs the exact preprocessing the server uses, then reports the one number that
matters: the margin between the lowest-scoring accident and the highest-scoring
non-accident. Your current model has a margin of +0.001 (desk 99.90%, crash
100.00%), which is why no threshold can work.

Usage
-----
    python evaluate.py --model model.onnx
    python evaluate.py --model model.onnx --data dataset --limit 60
"""

import argparse
import io
import os
import random

import numpy as np
import onnxruntime as ort
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def preprocess(path):
    img = Image.open(path).convert("RGB").resize((224, 224))
    d = np.array(img).astype(np.float32) / 255.0
    d = (d - MEAN) / STD
    return np.expand_dims(np.transpose(d, (2, 0, 1)), 0)


def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def score_dir(sess, name, folder, limit):
    if not os.path.isdir(folder):
        return []
    files = [os.path.join(folder, f) for f in os.listdir(folder)
             if f.lower().endswith(EXT)]
    random.Random(0).shuffle(files)
    out = []
    for p in files[:limit]:
        try:
            logits = sess.run(None, {name: preprocess(p)})[0]
            out.append(float(softmax(np.array(logits, np.float32))[0][1]) * 100)
        except Exception:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model.onnx")
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--limit", type=int, default=80)
    ap.add_argument("--threshold", type=float, default=75.0)
    args = ap.parse_args()

    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    print("model  :", args.model,
          "(%.1f MB)" % (os.path.getsize(args.model) / 1e6))
    print("input  :", sess.get_inputs()[0].shape,
          "-> output", sess.get_outputs()[0].shape)

    acc = score_dir(sess, name, os.path.join(args.data, "accident"), args.limit)
    neg = score_dir(sess, name, os.path.join(args.data, "non_accident"),
                    args.limit)
    if not acc or not neg:
        raise SystemExit("need images in both dataset/accident and "
                         "dataset/non_accident")

    t = args.threshold
    tp = sum(1 for v in acc if v >= t)
    fn = len(acc) - tp
    fp = sum(1 for v in neg if v >= t)
    tn = len(neg) - fp

    print("\n=== accident score distribution (class 1 %%) ===")
    print("  accidents     n=%-4d  min %6.2f  mean %6.2f  max %6.2f"
          % (len(acc), min(acc), sum(acc) / len(acc), max(acc)))
    print("  non-accidents n=%-4d  min %6.2f  mean %6.2f  max %6.2f"
          % (len(neg), min(neg), sum(neg) / len(neg), max(neg)))

    margin = min(acc) - max(neg)
    print("\n=== separation ===")
    print("  worst accident %.2f%%   best non-accident %.2f%%"
          % (min(acc), max(neg)))
    print("  margin %+.2f points" % margin)
    if margin > 10:
        print("  GOOD - a threshold sits comfortably between the two groups")
    elif margin > 0:
        print("  TIGHT - separable but fragile; more negatives would help")
    else:
        print("  BROKEN - the groups overlap, no threshold can separate them")
        print("           add more hard negatives and retrain")

    print("\n=== at threshold %.0f%% ===" % t)
    print("  caught accidents      %d/%d" % (tp, len(acc)))
    print("  MISSED accidents      %d      <- dangerous" % fn)
    print("  false alarms          %d/%d   <- wasted dispatches"
          % (fp, len(neg)))
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    print("  precision %.3f   recall %.3f   accuracy %.3f"
          % (prec, rec, (tp + tn) / max(len(acc) + len(neg), 1)))

    # a threshold that admits no false alarms at all
    safe = max(neg)
    print("\n  lowest threshold with zero false alarms: %.2f%% "
          "(would still catch %d/%d accidents)"
          % (safe, sum(1 for v in acc if v >= safe), len(acc)))


if __name__ == "__main__":
    main()
