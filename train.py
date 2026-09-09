"""
train.py — retrain the accident classifier and export ONNX for the server.

Produces a model byte-compatible with what main.py already expects:

    input   "input"   [1, 3, 224, 224]   NCHW, ImageNet normalised
    output  "output"  [1, 2]             raw logits (server applies softmax)
    class 0 = non_accident,  class 1 = accident   (ACCIDENT_CLASS_INDEX=1)

Same ResNet50 backbone as your current model, so nothing on the server needs
changing — drop the new model.onnx in and redeploy.

Unlike your current export this writes ONE file with the weights embedded,
not a 235 KB graph plus a separate 94 MB .data file. That split is what made
`model_loaded: true` misleading when the weights were absent.

Usage
-----
    pip install torch torchvision onnx onnxruntime pillow
    python train.py                      # sensible defaults
    python train.py --epochs 12 --batch 16
"""

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

# Must match the server: class 1 is the accident class. ImageFolder assigns
# indices alphabetically, so "accident" would become 0 — we force the order.
CLASSES = ["non_accident", "accident"]

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class OrderedImageFolder(datasets.ImageFolder):
    """ImageFolder that pins class indices instead of sorting alphabetically."""

    def find_classes(self, directory):
        missing = [c for c in CLASSES
                   if not os.path.isdir(os.path.join(directory, c))]
        if missing:
            raise FileNotFoundError(
                "missing folder(s) %s under %s" % (missing, directory))
        return CLASSES, {c: i for i, c in enumerate(CLASSES)}


def build_loaders(root, batch, workers, val_split):
    # Aggressive augmentation matters here: submissions are phone photos in
    # bad light, often photographed off a screen.
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    full = OrderedImageFolder(root, transform=train_tf)
    n_val = max(1, int(len(full) * val_split))
    n_train = len(full) - n_val
    train_ds, val_ds = random_split(
        full, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    val_ds.dataset = OrderedImageFolder(root, transform=val_tf)

    counts = [0, 0]
    for _, label in full.samples:
        counts[label] += 1
    print("  non_accident=%d  accident=%d  (train %d / val %d)"
          % (counts[0], counts[1], n_train, n_val))
    if min(counts) == 0:
        raise SystemExit("both classes need images - see build_dataset.py")

    return (DataLoader(train_ds, batch_size=batch, shuffle=True,
                       num_workers=workers, pin_memory=True),
            DataLoader(val_ds, batch_size=batch, shuffle=False,
                       num_workers=workers, pin_memory=True),
            counts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--out", default="model.onnx")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", dev)

    print("=== dataset ===")
    train_dl, val_dl, counts = build_loaders(args.data, args.batch,
                                             args.workers, args.val_split)

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, 2)     # 2048 -> 2, same as yours
    model = model.to(dev)

    # Imbalance is normal here (usually far more negatives); weight the loss
    # so the accident class is not drowned out.
    total = sum(counts)
    weights = torch.tensor([total / (2.0 * max(c, 1)) for c in counts],
                           dtype=torch.float32, device=dev)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_acc = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss = 0.0
        for x, y in train_dl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            run_loss += loss.item() * x.size(0)
        sched.step()

        model.eval()
        correct = total_n = 0
        fp = fn = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(dev), y.to(dev)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total_n += y.size(0)
                fp += ((pred == 1) & (y == 0)).sum().item()   # desk called crash
                fn += ((pred == 0) & (y == 1)).sum().item()   # crash missed
        acc = correct / max(total_n, 1)
        print("  epoch %2d/%d  loss %.4f  val acc %.3f  false-alarms %d  missed %d"
              % (epoch, args.epochs, run_loss / max(len(train_dl.dataset), 1),
                 acc, fp, fn))

        if acc >= best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    print("best val accuracy: %.3f" % best_acc)

    # ---- export ONNX, weights embedded in a single file ----
    model.eval().cpu()
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model, dummy, args.out,
        input_names=["input"], output_names=["output"],
        opset_version=13,
        do_constant_folding=True,
        dynamic_axes=None,                      # fixed [1,3,224,224]
    )
    size_mb = os.path.getsize(args.out) / 1e6
    print("\nwrote %s  (%.1f MB, single file - no .data companion)"
          % (args.out, size_mb))
    print("next: python evaluate.py --model %s" % args.out)


if __name__ == "__main__":
    main()
