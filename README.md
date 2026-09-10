# Retraining the accident classifier

Your current model scores a **photo of a desk at 99.90% "accident"** and a real
motorcycle crash at 100.00%. Nothing separates those two numbers, so no
threshold and no server setting can fix it. The model needs retraining.

## Why it fails

Control images show what it actually learned:

| Input | Predicted |
|---|---|
| Solid black / white / grey / blue / green | class 0 |
| Random noise | class 0 |
| Office photo, desk photo, real crash | **class 1** |

It learned *"is this a real photograph?"*, not *"is this a crash?"* — the
classic symptom of a non-accident class with too little variety. It has
apparently never seen an ordinary indoor scene or a parked car, so anything
photographic lands on the accident side of the boundary.

## Fix it in three commands

```bash
pip install requests pillow torch torchvision onnx onnxruntime

python build_dataset.py          # downloads ~1200 hard negatives
# copy your crash photos into dataset/accident/
python train.py                  # retrains ResNet50, writes model.onnx
python evaluate.py               # tells you whether it actually worked
```

## What each script does

**`build_dataset.py`** pulls Creative Commons images from Wikimedia Commons
(no API key, no account) into `dataset/non_accident/`. The search terms target
exactly what your model gets wrong — office desks, classrooms, living rooms,
parked cars, traffic jams, roadworks, broken-down vehicles, traffic police.
Roadside non-crashes matter most: a model that has never seen a parked car
will call one a collision.

Add `--positives` if you need more crash images too, but your own are better.

**`train.py`** fine-tunes ResNet50 — the same backbone as your current model
(53 convs, 2048→2 classifier, 23.5M params), so **the server needs no
changes**. It uses heavy augmentation because real submissions are phone
photos in poor light, sometimes photographed off a screen. Class weighting
handles the usual imbalance. It reports false alarms and missed accidents
separately each epoch, because those two errors cost very different things.

**`evaluate.py`** reports the number that matters: the margin between the
worst-scoring accident and the best-scoring non-accident.

```
margin +32.40 points   GOOD - a threshold sits comfortably between
margin  +0.001 points  BROKEN - your current model
```

## Two things the export fixes

Your current `model.onnx` is **235 KB of graph** with the 94 MB of weights in a
separate `model.onnx.data`. onnxruntime creates a session even when that file
is missing, so `model_loaded: true` is not proof the weights are there —
a silent, dangerous failure.

`train.py` exports **one self-contained file**. Deploy it and that whole class
of problem disappears.

It also pins class order so `accident` is index 1, matching the server's
`ACCIDENT_CLASS_INDEX=1`. PyTorch's `ImageFolder` sorts alphabetically by
default, which would silently make `accident` index 0 and invert every verdict.

## Deploying the new model

Drop `model.onnx` into your server repo, push, let Render redeploy. Then:

```bash
curl -F "photo=@desk.jpg"  https://resqser.onrender.com/api/debug/classify
curl -F "photo=@crash.jpg" https://resqser.onrender.com/api/debug/classify
```

Desk should score low, crash high. If both sit near 100%, it needs more
negatives — go back to `build_dataset.py --per-term 120`.

## The highest-value thing you can add

A few hundred photos taken on the phones your users will actually submit
from. Same cameras, same lighting, same motion blur, same habit of
photographing a screen. Downloaded stock images are a decent floor, but they
do not look like what arrives at 2am from a roadside.

Split them roughly evenly: for every crash photo, one ordinary photo of
something that is emphatically not a crash.
