# Honest training + overfitting check for the mango leaf model (runs on your laptop).
#
#   python train_check.py --quick        # 1-minute smoke test first (tiny data, 1 epoch)
#   python train_check.py                # real run: up to 15 epochs, early stopping
#   python train_check.py --epochs 20    # more epochs if sir wants
#
# Why this exists: the old script split the 4000 images RANDOMLY. MangoLeafBD has
#   * ~300 exact copies of the same photo (e.g. "IMG_x (Custom).jpg" and "IMG_x (Custom) (Custom).jpg"),
#     277 of them in Cutting Weevil
#   * burst photos taken seconds apart of the same leaf
# so a random split puts near-identical photos in train AND test -> test accuracy is inflated (100%).
#
# This script:
#   1. removes duplicate copies of the same shot
#   2. splits each class BY TIME: oldest 70% train, next 10% validation, newest 20% test
#      (test = later photo sessions the model never saw anything close to)
#   3. picks the best epoch on VALIDATION (not test), stops early if validation stops improving
#   4. reports train vs test accuracy (the overfitting gap), per-class accuracy, confusion matrix
#   5. if a folder real_photos/<Disease>/ or feedback_data/<Disease>/ exists, tests on those real photos too
#   6. saves mango_resnet_model.pth + anchors.npz next to this file (old model kept as *_prev.pth)
import argparse
import os
import re
import shutil
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=15)
ap.add_argument("--patience", type=int, default=4, help="stop after this many epochs without val improvement")
ap.add_argument("--quick", action="store_true", help="tiny smoke test, saves nothing")
args = ap.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "MangoLeafBD Dataset")
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
torch.manual_seed(42); np.random.seed(42); random.seed(42)

norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.05),
    transforms.RandomApply([transforms.GaussianBlur(5, (0.1, 2.0))], p=0.3),
    transforms.ToTensor(), norm])
test_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), norm])

ds_train = datasets.ImageFolder(DATA, train_tf)
ds_eval = datasets.ImageFolder(DATA, test_tf)
classes = ds_eval.classes
print(classes)


def shot_key(path):
    """'IMG_20211011_140211 (Custom) (Custom).jpg' -> '20211011140211' (date+time of the photo)."""
    name = os.path.basename(path).split("(")[0]
    digits = re.sub(r"\D", "", name)
    return digits[:14] if len(digits) >= 14 else os.path.basename(path)


# 1. de-duplicate copies of the same shot
seen, keep = set(), []
for i, (p, y) in enumerate(ds_eval.samples):
    k = (y, shot_key(p))
    if k not in seen:
        seen.add(k); keep.append(i)
print(f"images: {len(ds_eval.samples)}  after removing duplicate copies: {len(keep)}")

# 2. time-based split per class
tr_idx, va_idx, te_idx = [], [], []
for c in range(len(classes)):
    idx = sorted((i for i in keep if ds_eval.samples[i][1] == c), key=lambda i: shot_key(ds_eval.samples[i][0]))
    if args.quick:
        idx = idx[:: max(1, len(idx) // 20)][:20]
    a, b = int(0.7 * len(idx)), int(0.8 * len(idx))
    tr_idx += idx[:a]; va_idx += idx[a:b]; te_idx += idx[b:]
print(f"train {len(tr_idx)} | val {len(va_idx)} | test {len(te_idx)}  (test = newest photos of each class)")

nw = 0 if os.name == "nt" else 2
train_loader = DataLoader(Subset(ds_train, tr_idx), batch_size=32, shuffle=True, num_workers=nw)
train_eval_loader = DataLoader(Subset(ds_eval, tr_idx), batch_size=64, num_workers=nw)
val_loader = DataLoader(Subset(ds_eval, va_idx), batch_size=64, num_workers=nw)
test_loader = DataLoader(Subset(ds_eval, te_idx), batch_size=64, num_workers=nw)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(512, len(classes))
for name, p in model.named_parameters():
    p.requires_grad = name.startswith("layer4") or name.startswith("fc")
model = model.to(device)
opt = torch.optim.Adam([{"params": model.layer4.parameters(), "lr": 1e-4},
                        {"params": model.fc.parameters(), "lr": 1e-3}], weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
crit = nn.CrossEntropyLoss(label_smoothing=0.1)


def set_train():
    model.train()
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d) and not name.startswith("layer4"):
            m.eval()


def predict(loader):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            ps.append(F.softmax(model(x.to(device)), 1).cpu()); ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def acc(loader):
    y, p = predict(loader)
    return float((p.argmax(1) == y).mean()), float(p.max(1).mean())


epochs = 1 if args.quick else args.epochs
best_val, best_state, bad = -1.0, None, 0
print("\nepoch | train loss | train acc | val acc | gap(train-val)")
for ep in range(epochs):
    set_train()
    tot, n = 0.0, 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        loss = crit(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item() * len(y); n += len(y)
    sched.step()
    tr_acc, _ = acc(train_eval_loader)
    va_acc, _ = acc(val_loader)
    print(f"{ep + 1:5d} | {tot / n:10.4f} | {tr_acc:8.2%} | {va_acc:7.2%} | {tr_acc - va_acc:+.2%}")
    if va_acc > best_val:
        best_val, bad = va_acc, 0
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    else:
        bad += 1
        if bad >= args.patience:
            print(f"early stop: val accuracy did not improve for {args.patience} epochs")
            break

model.load_state_dict(best_state)
tr_acc, _ = acc(train_eval_loader)
y, p = predict(test_loader)
pred = p.argmax(1)
te_acc = float((pred == y).mean())

print("\n================ RESULT (best epoch by validation) ================")
print(f"train accuracy : {tr_acc:.2%}")
print(f"val accuracy   : {best_val:.2%}")
print(f"TEST accuracy  : {te_acc:.2%}   (newest, unseen photo sessions)")
print(f"mean confidence on test: {p.max(1).mean():.2%}")
gap = tr_acc - te_acc
if gap > 0.10:
    print(f"-> OVERFITTING: train is {gap:.0%} higher than test.")
elif te_acc > 0.97:
    print("-> No train/test gap. High test accuracy on later sessions = it learned the disease, "
          "not just the photo. Still check real field photos below.")
else:
    print(f"-> Small gap ({gap:+.1%}); test accuracy {te_acc:.1%} is the honest number to report.")

print("\nper-class test accuracy:")
for c, name in enumerate(classes):
    m = y == c
    if m.any():
        print(f"  {name:18s} {(pred[m] == c).mean():6.1%}  ({m.sum()} images)")
cm = np.zeros((len(classes), len(classes)), int)
for t, q in zip(y, pred):
    cm[t, q] += 1
print("\nconfusion matrix (rows = true, cols = predicted):")
print("      " + " ".join(f"{n[:4]:>5s}" for n in classes))
for c, name in enumerate(classes):
    print(f"{name[:5]:>5s} " + " ".join(f"{v:5d}" for v in cm[c]))

# 5. real photos (the number that actually matters)
for folder in ("real_photos", "feedback_data"):
    root = os.path.join(HERE, folder)
    if not os.path.isdir(root):
        continue
    ok = tot = 0
    model.eval()
    for c, name in enumerate(classes):
        d = os.path.join(root, name)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if not f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                continue
            try:
                img = Image.open(os.path.join(d, f)).convert("RGB")
            except OSError:
                continue
            with torch.no_grad():
                q = F.softmax(model(test_tf(img).unsqueeze(0).to(device)), 1)[0].cpu()
            tot += 1; ok += int(q.argmax() == c)
            print(f"  [{folder}] {name:16s} -> {classes[int(q.argmax())]:16s} {q.max():.0%}  {f}")
    if tot:
        print(f"REAL PHOTO accuracy ({folder}): {ok}/{tot} = {ok / tot:.0%}")

if args.quick:
    print("\nquick test OK - now run:  python train_check.py")
    raise SystemExit

# 6. save
mp = os.path.join(HERE, "mango_resnet_model.pth")
if os.path.exists(mp):
    shutil.copy2(mp, os.path.join(HERE, "mango_resnet_model_prev.pth"))
model = model.cpu()
torch.save(model.state_dict(), mp)
model.fc = nn.Identity(); model.eval()
X, Y = [], []
with torch.no_grad():
    for c in range(len(classes)):
        for i in [i for i in tr_idx if ds_eval.samples[i][1] == c][:30]:
            img, lab = ds_eval[i]
            X.append(model(img.unsqueeze(0)).numpy()[0]); Y.append(lab)
np.savez(os.path.join(HERE, "anchors.npz"), x=np.array(X), y=np.array(Y))
print("\nsaved mango_resnet_model.pth + anchors.npz (old model -> mango_resnet_model_prev.pth)")
print("NOW delete agent_state.npz and rl_memory.json (keep feedback_data), then: streamlit run app.py")
