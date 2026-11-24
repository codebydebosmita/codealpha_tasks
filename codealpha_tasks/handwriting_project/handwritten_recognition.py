"""
Handwritten Character Recognition
---------------------------------
CNN trained on EMNIST (digits + letters) with PyTorch, plus a CRNN model
for extending to words/sentences.

Install:   pip install torch torchvision pillow numpy
Train:     python handwritten_recognition.py train --epochs 10
Predict:   python handwritten_recognition.py predict my_char.png
"""
import argparse
import glob
import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# EMNIST "balanced" split: 47 classes (merges look-alike upper/lowercase letters)
LABELS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"
NUM_CLASSES = len(LABELS)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "emnist_cnn.pt"


# ---------------------------------------------------------------- Model ----
class CharCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                                   # 14x14
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                                   # 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.4),
            nn.Linear(64 * 7 * 7, 256), nn.ReLU(),
            nn.Dropout(0.4), nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ----------------------------------------------------------------- Data ----
def fix_orientation(im):
    return im.transpose(Image.TRANSPOSE)


def get_loaders(batch_size=128):
    # torchvision's EMNIST images are transposed; transpose back to upright.
    fix = transforms.Lambda(fix_orientation)
    train_tf = transforms.Compose([
        fix,
        transforms.RandomAffine(10, translate=(0.08, 0.08), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1751,), (0.3332,)),
    ])
    test_tf = transforms.Compose([
        fix, transforms.ToTensor(), transforms.Normalize((0.1751,), (0.3332,)),
    ])
    train = datasets.EMNIST("data", split="balanced", train=True, download=True, transform=train_tf)
    test = datasets.EMNIST("data", split="balanced", train=False, download=True, transform=test_tf)
    return (DataLoader(train, batch_size, shuffle=True, num_workers=0),
            DataLoader(test, 512, num_workers=0))


# ------------------------------------------------------------- Training ----
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


def train(epochs=10, lr=1e-3, resume=False):
    train_loader, test_loader = get_loaders()
    model = CharCNN().to(DEVICE)
    best = 0.0
    if resume:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        best = evaluate(model, test_loader)
        print(f"Resumed from {MODEL_PATH} (test acc {best:.4f})")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr * 5,
                                                steps_per_epoch=len(train_loader), epochs=epochs)
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            sched.step()
            running += loss.item()
        acc = evaluate(model, test_loader)
        print(f"epoch {epoch:2d} | loss {running / len(train_loader):.4f} | test acc {acc:.4f}")
        if acc > best:
            best = acc
            torch.save(model.state_dict(), MODEL_PATH)
    print(f"Best test accuracy: {best:.4f}  (saved to {MODEL_PATH})")


# ----------------------------------------------------------- Prediction ----
NORM = transforms.Normalize((0.1751,), (0.3332,))


def preprocess_pil(path, debug=True):
    """Photo/scan of ONE character -> 28x28 PIL image styled like EMNIST
    (white bold glyph on black, softly blurred, tightly centered)."""
    arr = np.array(Image.open(path).convert("L"))
    if arr.mean() > 127:                                   # dark ink on light paper -> invert
        arr = 255 - arr
    blur = max(1.0, min(arr.shape) / 150)
    arr = np.array(Image.fromarray(arr).filter(ImageFilter.GaussianBlur(blur)))
    arr = np.where(arr > 0.25 * arr.max(), arr, 0).astype(np.uint8)   # drop background/noise
    ys, xs = np.where(arr > 0)
    if len(xs) == 0:
        raise ValueError("No ink found in image.")
    arr = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]           # crop to glyph
    h, w = arr.shape
    k = max(3, int(max(h, w) * 0.08) // 2 * 2 + 1)                    # thicken thin strokes
    arr = np.array(Image.fromarray(np.pad(arr, k)).filter(ImageFilter.MaxFilter(k)))
    ys, xs = np.where(arr > 0)
    arr = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = arr.shape
    side = int(max(h, w) * 1.15)                                      # EMNIST-like small border
    canvas = np.zeros((side, side), np.uint8)
    y0, x0 = (side - h) // 2, (side - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = arr
    img = Image.fromarray(canvas).resize((28, 28), Image.LANCZOS)
    a = np.array(img).astype(np.float32)
    img = Image.fromarray((a * 255 / max(a.max(), 1)).astype(np.uint8))   # stretch contrast
    if debug:
        img.resize((224, 224), Image.NEAREST).save("debug_input.png")
    return img


def preprocess(path):
    return NORM(transforms.ToTensor()(preprocess_pil(path))).unsqueeze(0)


def predict(path, topk=3):
    model = CharCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    with torch.no_grad():
        probs = F.softmax(model(preprocess(path).to(DEVICE)), dim=1)[0]
    for p, i in zip(*probs.topk(topk)):
        print(f"{LABELS[i]!r}: {p.item() * 100:.1f}%")
    print("Saved what the model sees to debug_input.png")


# ------------------------------------------- Fine-tune on YOUR handwriting ----
def finetune(folder, epochs=40, lr=1e-4):
    """folder/<label>/*.png  e.g. samples/7/, samples/A/, samples/lower_a/ (lowercase)."""
    model = CharCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    shutil.copy(MODEL_PATH, "emnist_cnn_base.pt")          # backup of the original model
    for prm in model.features.parameters():                 # only retrain the classifier head
        prm.requires_grad = False
    imgs, ys = [], []
    for d in sorted(os.listdir(folder)):
        name = d[6:] if d.startswith("lower_") else d
        if name not in LABELS:
            print(f"skipping folder '{d}' (not a known label)")
            continue
        for f in glob.glob(os.path.join(folder, d, "*")):
            try:
                imgs.append(preprocess_pil(f, debug=False))
                ys.append(LABELS.index(name))
            except Exception as e:
                print(f"skipping {f}: {e}")
    if not imgs:
        raise SystemExit("No usable images found.")
    print(f"Fine-tuning on {len(imgs)} images, {len(set(ys))} classes")
    aug = transforms.RandomAffine(8, translate=(0.08, 0.08), scale=(0.9, 1.1))
    opt = torch.optim.Adam(model.classifier.parameters(), lr=lr)
    for ep in range(1, epochs + 1):
        model.train()
        model.features.eval()                               # keep BatchNorm stats fixed
        order = np.random.permutation(len(imgs))
        total = 0.0
        for i in range(0, len(order), 16):
            idx = order[i:i + 16]
            x = torch.stack([NORM(transforms.ToTensor()(aug(imgs[j]))) for j in idx]).to(DEVICE)
            y = torch.tensor([ys[j] for j in idx]).to(DEVICE)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        if ep % 5 == 0 or ep == 1:
            print(f"epoch {ep:3d} | loss {total / len(imgs):.4f}")
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved fine-tuned model to {MODEL_PATH} (original backed up as emnist_cnn_base.pt)")


# -------------------------------------- Extension: CRNN for words/lines ----
class CRNN(nn.Module):
    """
    Input : (B, 1, 32, W) grayscale text-line image
    Output: (T, B, num_classes+1) log-probs for nn.CTCLoss (class 0 = blank)
    Train with an IAM / synthetic word dataset:
        loss = nn.CTCLoss(blank=0, zero_infinity=True)
        loss(log_probs, targets, input_lengths, target_lengths)
    """
    def __init__(self, num_classes=len(LABELS), hidden=128):
        super().__init__()
        def block(i, o, pool):
            return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(pool)]
        self.cnn = nn.Sequential(
            *block(1, 64, (2, 2)),      # H 16
            *block(64, 128, (2, 2)),    # H 8
            *block(128, 256, (2, 1)),   # H 4  (keep width = time steps)
            *block(256, 256, (2, 1)),   # H 2
            nn.Conv2d(256, 256, (2, 1)), nn.ReLU(),   # H 1
        )
        self.rnn = nn.LSTM(256, hidden, num_layers=2, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, num_classes + 1)

    def forward(self, x):
        f = self.cnn(x).squeeze(2).permute(2, 0, 1)   # (T, B, C)
        out, _ = self.rnn(f)
        return F.log_softmax(self.fc(out), dim=2)

    @staticmethod
    def greedy_decode(log_probs):
        """Collapse repeats and blanks -> list of strings."""
        best = log_probs.argmax(2).T.tolist()
        results = []
        for seq in best:
            prev, s = 0, []
            for k in seq:
                if k != prev and k != 0:
                    s.append(LABELS[k - 1])
                prev = k
            results.append("".join(s))
        return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("--epochs", type=int, default=10)
    t.add_argument("--resume", action="store_true", help="continue from saved model")
    t.add_argument("--lr", type=float, default=1e-3)
    p = sub.add_parser("predict"); p.add_argument("image")
    f = sub.add_parser("finetune"); f.add_argument("folder")
    f.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()
    if args.cmd == "train":
        train(args.epochs, args.lr, args.resume)
    elif args.cmd == "predict":
        predict(args.image)
    else:
        finetune(args.folder, args.epochs)
