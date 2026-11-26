# Paste this whole thing into ONE cell of a fresh Kaggle notebook (dataset attached).
# Turn on the GPU first: Settings -> Accelerator -> GPU. Takes a few minutes.
# It creates two files in /kaggle/working: mango_resnet_model.pth and anchors.npz
# Download BOTH from the Output panel and put them next to app.py.
import random
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

path = r"C:\Users\kvrmu\OneDrive\Desktop\ADDITIONAL PROJECT\MangoLeafBD Dataset"
output_dir = "/kaggle/working" if os.name != "nt" else os.path.dirname(os.path.abspath(__file__))
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
torch.manual_seed(42); np.random.seed(42); random.seed(42)

norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# Augmentation imitates phone photos: different crops, angles, lighting, blur
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.05),
    transforms.RandomApply([transforms.GaussianBlur(5, (0.1, 2.0))], p=0.3),
    transforms.ToTensor(), norm])
test_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), norm])

ds_train = datasets.ImageFolder(path, train_tf)   # augmented view (for training)
ds_eval = datasets.ImageFolder(path, test_tf)     # clean view (for testing / anchors)
class_names = ds_train.classes
print(class_names)
assert class_names == ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back',
                       'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould']

# fixed 80/20 split (same every run), test images never get augmentation
perm = torch.randperm(len(ds_train), generator=torch.Generator().manual_seed(42)).tolist()
n_tr = int(0.8 * len(perm))
tr_idx, te_idx = perm[:n_tr], perm[n_tr:]
num_workers = 0 if os.name == "nt" else 2
train_loader = DataLoader(Subset(ds_train, tr_idx), batch_size=32, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(Subset(ds_eval, te_idx), batch_size=32, shuffle=False, num_workers=num_workers)
print("train:", len(tr_idx), "test:", len(te_idx))

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(512, len(class_names))
for name, p in model.named_parameters():          # train only the last block + classifier
    p.requires_grad = name.startswith("layer4") or name.startswith("fc")
model = model.to(device)

opt = torch.optim.Adam([{"params": model.layer4.parameters(), "lr": 1e-4},
                        {"params": model.fc.parameters(), "lr": 1e-3}])
crit = nn.CrossEntropyLoss(label_smoothing=0.1)   # discourages 100%-sure predictions


def set_train():
    model.train()
    for name, m in model.named_modules():          # keep frozen BatchNorm layers fixed
        if isinstance(m, nn.BatchNorm2d) and not name.startswith("layer4"):
            m.eval()


def evaluate(loader):
    model.eval()
    correct = total = 0
    confs = []
    with torch.no_grad():
        for x, y in loader:
            p = F.softmax(model(x.to(device)), dim=1).cpu()
            c, pred = p.max(1)
            correct += (pred == y).sum().item(); total += len(y); confs += c.tolist()
    return correct / total, float(np.mean(confs))


for epoch in range(8):
    set_train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        loss = crit(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
    acc, conf = evaluate(test_loader)
    print(f"epoch {epoch + 1}/8  test accuracy {acc:.2%}  mean confidence {conf:.2%}")

model = model.cpu()
torch.save(model.state_dict(), os.path.join(output_dir, "mango_resnet_model.pth"))
print("saved mango_resnet_model.pth")

# anchors = 30 clean training images per class (as 512-d features), replayed by the RL agent
model.fc = nn.Identity()
model.eval().to(device)
by_class = {k: [] for k in range(len(class_names))}
for i in tr_idx:
    by_class[ds_eval.targets[i]].append(i)
X, Y = [], []
with torch.no_grad():
    for k, idxs in by_class.items():
        for i in idxs[:30]:
            img, label = ds_eval[i]
            X.append(model(img.unsqueeze(0).to(device)).cpu().numpy()[0]); Y.append(label)
np.savez(os.path.join(output_dir, "anchors.npz"), x=np.array(X), y=np.array(Y))
print("saved anchors.npz", np.array(X).shape, "-> now download both files from the Output panel")
