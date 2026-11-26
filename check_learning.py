"""
check_learning.py - does the RL agent really LEARN from feedback, and does it help on OTHER photos?

Test (uses your trained model, anchors.npz and the MangoLeafBD folder; does not touch the app's saved state):
  * takes the newest 20% of each class (the test sessions) and splits them into two groups:
        feedback photos  - the "user" gives feedback on these
        unseen photos    - never given feedback, only used to measure
  * makes both groups look like field photos (colour cast, low contrast, blur, tighter crop), because on clean
    dataset photos the model is already ~96% right and there is little left to learn
  * gives 0, 8, 24, 48, 96 feedbacks (label feedback, and reward-only feedback) and measures accuracy on the
    UNSEEN photos (does the learning transfer?) and on clean photos (does it forget what it knew?)
  * 5 random repeats, mean and spread

Run:  python check_learning.py            (a few minutes on CPU)
"""
import os, sys, glob
import numpy as np
import torch, torch.nn as nn, torchvision.models as models
from torchvision import transforms
from PIL import Image, ImageFilter, ImageEnhance

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from rl_agent import RLAgent

CLASSES = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 'Gall Midge', 'Healthy',
           'Powdery Mildew', 'Sooty Mould']
data = next((d for d in glob.glob(os.path.join(HERE, "**"), recursive=True)
             if os.path.isdir(d) and all(os.path.isdir(os.path.join(d, c)) for c in CLASSES)), None)
if data is None:
    sys.exit("MangoLeafBD Dataset folder (with the 8 class folders) not found next to this script.")

torch.set_num_threads(max(1, os.cpu_count() or 1))
r = models.resnet18(weights=None); r.fc = nn.Linear(512, 8)
r.load_state_dict(torch.load(os.path.join(HERE, "mango_resnet_model.pth"), map_location="cpu"))
W0, b0 = r.fc.weight.detach().numpy().copy(), r.fc.bias.detach().numpy().copy(); r.fc = nn.Identity(); r.eval()
tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def field_look(im):
    w, h = im.size
    im = im.crop((int(.12 * w), int(.12 * h), int(.88 * w), int(.88 * h)))
    im = ImageEnhance.Contrast(ImageEnhance.Color(im).enhance(0.35)).enhance(0.6)
    a = np.asarray(im).astype(float); a[..., 0] *= 1.15; a[..., 2] *= 0.75
    return Image.fromarray(np.clip(a, 0, 255).astype("uint8")).filter(ImageFilter.GaussianBlur(1.5))


def feats(paths, shifted):
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), 64):
            ims = [Image.open(p).convert("RGB") for p in paths[i:i + 64]]
            out.append(r(torch.stack([tf(field_look(x) if shifted else x) for x in ims])).numpy())
    return np.concatenate(out)


pool, test, anc = [], [], []
for c, name in enumerate(CLASSES):
    fs = sorted(f for f in os.listdir(os.path.join(data, name)) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    n = len(fs); newest = fs[int(.8 * n):]
    pool += [(os.path.join(data, name, f), c) for f in newest[0::2][:40]]
    test += [(os.path.join(data, name, f), c) for f in newest[1::2][:40]]
    anc += [(os.path.join(data, name, f), c) for f in fs[:int(.7 * n)][:30]]
print("computing features (a few minutes) ...", flush=True)
PX, PY = feats([p for p, _ in pool], True), np.array([c for _, c in pool])
TX, TY = feats([p for p, _ in test], True), np.array([c for _, c in test])
CX = feats([p for p, _ in test], False)
if os.path.exists(os.path.join(HERE, "anchors.npz")):
    with np.load(os.path.join(HERE, "anchors.npz")) as a:
        AX, AY = np.array(a["x"]), np.array(a["y"])
else:
    AX, AY = feats([p for p, _ in anc], False), np.array([c for _, c in anc])


def acc(ag, X, y):
    return float((np.array([ag.probs(x).argmax() for x in X]) == y).mean())


def run(mode, n_fb, seed):
    rng = np.random.default_rng(seed)
    ag = RLAgent(W0, b0, CLASSES, anchors=(AX, AY), seed=seed)
    for x in TX:
        ag.observe(x)
    for i in rng.permutation(len(PY))[:n_fb]:
        x, y = PX[i], int(PY[i]); pred = int(ag.probs(x).argmax())
        if mode == "label":
            ag.learn(x, y)
        else:
            ag.learn_from_reward(x, pred, 10.0 if pred == y else -20.0)
    return acc(ag, TX, TY), acc(ag, CX, TY)


base = RLAgent(W0, b0, CLASSES)
print(f"\nbefore feedback: field-look UNSEEN photos {acc(base, TX, TY):.1%} | clean photos {acc(base, CX, TY):.1%}")
print(f"{'feedback type':14s} {'#feedbacks':>10s} {'UNSEEN field-look':>18s} {'clean (forgetting?)':>20s}")
for mode in ("label", "reward-only"):
    for n in (8, 24, 48, 96):
        res = np.array([run("label" if mode == "label" else "reward", n, s) for s in range(5)])
        print(f"{mode:14s} {n:10d} {res[:, 0].mean():12.1%} ±{res[:, 0].std():4.1%} {res[:, 1].mean():14.1%}")
print("\nUNSEEN photos never got feedback, so a rise there means the learning transfers to new photos.")
