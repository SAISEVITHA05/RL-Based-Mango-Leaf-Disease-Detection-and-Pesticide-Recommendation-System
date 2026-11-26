"""
Finds out WHY the app says Anthracnose / Bacterial Canker so often.
Compares: (A) the trained model alone   (B) the model + the RL agent saved in agent_state.npz

Usage (put next to app.py):
    python check_bias.py "MangoLeafBD Dataset" --per-class 20      # dataset photos
    python check_bias.py real_photos                                # your own photos, same folder layout
Folder layout: <folder>/<Disease name>/*.jpg
"""
import os, sys, argparse
import numpy as np
import torch, torch.nn as nn, torchvision.models as models
from torchvision import transforms
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rl_agent import RLAgent

C = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould']
ap = argparse.ArgumentParser()
ap.add_argument("folder")
ap.add_argument("--per-class", type=int, default=20)
a = ap.parse_args()

tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
net = models.resnet18(weights=None)
net.fc = nn.Linear(512, 8)
net.load_state_dict(torch.load(os.path.join(HERE, "mango_resnet_model.pth"), map_location="cpu"))
net.eval()
W0, b0 = net.fc.weight.detach().numpy().copy(), net.fc.bias.detach().numpy().copy()
net.fc = nn.Identity()

anch = None
if os.path.exists(os.path.join(HERE, "anchors.npz")):
    z = np.load(os.path.join(HERE, "anchors.npz")); anch = (z["x"], z["y"])
fresh = RLAgent(W0, b0, C, anchors=anch)                       # model only, no feedback
saved = RLAgent(W0, b0, C, anchors=anch)
has_state = saved.load(os.path.join(HERE, "agent_state.npz"))
print("agent_state.npz found and loaded:", has_state, "| feedback examples in it:", len(saved.fb_y) if has_state else 0)

res = {"model only": [], "model + saved RL": []}
truth = []
for ci, name in enumerate(C):
    d = os.path.join(a.folder, name)
    if not os.path.isdir(d):
        continue
    files = sorted(f for f in os.listdir(d) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    files = files[-a.per_class:]                                # newest photos
    for f in files:
        with torch.no_grad():
            x = net(tf(Image.open(os.path.join(d, f)).convert("RGB")).unsqueeze(0)).numpy()[0]
        truth.append(ci)
        res["model only"].append(int(fresh.decide(x)[0][0]))
        res["model + saved RL"].append(int(saved.decide(x)[0][0]))

truth = np.array(truth)
for k, p in res.items():
    p = np.array(p)
    print(f"\n=== {k}: accuracy {np.mean(p == truth):.1%} on {len(truth)} photos")
    print("   how often each disease is predicted:", {C[i][:8]: int((p == i).sum()) for i in range(8)})
    print("   how often each disease really occurs:", {C[i][:8]: int((truth == i).sum()) for i in range(8)})
    for i in range(8):
        m = truth == i
        if m.any():
            print(f"   {C[i]:17s} recall {np.mean(p[m] == i):5.0%}")
print("\nIf 'model only' is good but 'model + saved RL' is bad -> delete agent_state.npz and rl_memory.json.")
print("If both are bad on the dataset -> the model file / class order is wrong.")
print("If both are good on the dataset but bad on your own photos -> real-photo domain shift (need real training photos).")