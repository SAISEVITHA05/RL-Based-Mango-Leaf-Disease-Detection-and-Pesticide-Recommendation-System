"""
Does the RL agent really learn?  Measure it on REAL labelled photos.

1) Put your own labelled mango photos in folders named exactly like the classes:
       real_photos/Anthracnose/*.jpg
       real_photos/Bacterial Canker/*.jpg
       real_photos/Healthy/*.jpg ...            (a folder "None of these" is also allowed)
2) Run:   python evaluate_learning.py real_photos
   (no photos yet? try the synthetic self-test:  python evaluate_learning.py --demo)

What it does: the photos are split into a feedback stream and a held-out test set. The agent
receives feedback one photo at a time, and after 0, 1, 2, 4, 8 ... feedbacks it is tested on the
held-out photos (never used for feedback). It repeats this with many random splits and reports
mean accuracy and mean reward (+10 right, -20 wrong, -2 'not sure'), for
   label  : feedback with the correct disease      (what your app's "No, it's actually..." does)
   reward : reward only, no label (pure RL)         (what "No, but I don't know..." does)
and saves learning_curve.png / learning_curve.csv for your report.
"""
import os
import sys
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rl_agent import RLAgent

CLASS_NAMES = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back',
               'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould']
R_RIGHT, R_WRONG, R_UNSURE = 10.0, -20.0, -2.0


# ---------------------------------------------------------------- features (needs torch) ----
def extract_features(folder, model_path):
    import torch
    import torch.nn as nn
    import torchvision.models as models
    from torchvision import transforms
    from PIL import Image
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    net = models.resnet18(weights=None)
    net.fc = nn.Linear(net.fc.in_features, len(CLASS_NAMES))
    net.load_state_dict(torch.load(model_path, map_location="cpu"))
    net.eval()
    W0, b0 = net.fc.weight.detach().numpy().copy(), net.fc.bias.detach().numpy().copy()
    net.fc = nn.Identity()
    X, y = [], []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if not os.path.isdir(path):
            continue
        label = CLASS_NAMES.index(name) if name in CLASS_NAMES else len(CLASS_NAMES)
        for f in sorted(os.listdir(path)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                with torch.no_grad():
                    X.append(net(tf(Image.open(os.path.join(path, f)).convert("RGB")).unsqueeze(0)).numpy()[0])
                y.append(label)
    return np.array(X), np.array(y), W0, b0


# ---------------------------------------------------------------- the experiment (numpy only) ----
def test_reward(agent, X, y):
    acc, rew = [], []
    for x, t in zip(X, y):
        a = int(agent.act(x)[0])                     # disease index, or UNKNOWN when it says "not sure"
        if a == t:
            r = R_RIGHT
        elif a == agent.UNKNOWN:
            r = R_UNSURE
        else:
            r = R_WRONG
        acc.append(a == t); rew.append(r)
    return float(np.mean(acc)), float(np.mean(rew))


def run_experiment(X, y, W0, b0, anchors=None, mode="label", repeats=30, test_frac=0.5, seed=0):
    rng = np.random.default_rng(seed)
    n_stream = int(round(len(y) * (1 - test_frac)))
    checkpoints = sorted({0, 1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128} & set(range(n_stream + 1)) | {n_stream})
    acc = np.zeros((repeats, len(checkpoints))); rew = np.zeros_like(acc)
    for r in range(repeats):
        idx = rng.permutation(len(y))
        stream, test = idx[:n_stream], idx[n_stream:]
        agent = RLAgent(W0, b0, CLASS_NAMES, anchors=anchors, seed=r)
        for x in X:                                   # the app keeps a running feature average of every image it sees
            agent.observe(x)
        j = 0
        for k in range(n_stream + 1):
            if k in checkpoints:
                acc[r, j], rew[r, j] = test_reward(agent, X[test], y[test]); j += 1
            if k == n_stream:
                break
            x, t = X[stream[k]], int(y[stream[k]])
            if mode == "label":
                agent.learn(x, t)
            else:                                     # pure RL: only the reward of the action it took
                a = int(agent.decide(x)[0][0])
                agent.learn_from_reward(x, a, R_RIGHT if a == t else R_WRONG)
    return checkpoints, acc, rew


def demo_data(seed=0):
    d, K = 512, len(CLASS_NAMES)
    rng = np.random.default_rng(seed)
    C = rng.normal(size=(K, d)); C = 6 * C / np.linalg.norm(C, axis=1, keepdims=True)
    mu = np.abs(rng.normal(size=d)); mu = 15 * mu / np.linalg.norm(mu)
    X, y = [], []
    for k in range(K):
        for _ in range(12):
            m = (k + 3) % K
            if rng.random() < 0.5:
                X.append(mu + 1.2 * C[k] + 0.2 * C[m] + 0.3 * rng.normal(size=d))
            else:
                X.append(mu + 0.8 * C[k] + 1.2 * C[m] + 0.3 * rng.normal(size=d))
            y.append(k)
    return np.array(X), np.array(y), 0.12 * C, -0.12 * (C ** 2).sum(1) / 2 - 0.12 * (C @ mu)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", nargs="?", help="folder with one sub-folder of photos per class")
    ap.add_argument("--model", default=os.path.join(HERE, "mango_resnet_model.pth"))
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--demo", action="store_true", help="run on synthetic features (no photos / torch needed)")
    args = ap.parse_args()

    if args.demo:
        X, y, W0, b0 = demo_data()
        print("DEMO on synthetic features - this only checks the learning logic, it is NOT a result on real leaves.\n")
    else:
        if not args.folder:
            ap.error("give the photo folder, or use --demo")
        X, y, W0, b0 = extract_features(args.folder, args.model)
    anchors = None
    if os.path.exists(os.path.join(HERE, "anchors.npz")):
        a = np.load(os.path.join(HERE, "anchors.npz")); anchors = (a["x"], a["y"])
    print(f"{len(y)} photos | classes present: {sorted(set(CLASS_NAMES[i] if i < len(CLASS_NAMES) else 'None of these' for i in y))}")
    if len(y) < 24:
        print("WARNING: very few photos - the curves will be noisy. 100+ labelled real photos is much better.")

    curves = {}
    for mode in ("label", "reward"):
        cps, acc, rew = run_experiment(X, y, W0, b0, anchors, mode, args.repeats)
        curves[mode] = (cps, acc, rew)
        print(f"\nfeedback type: {mode}   ({args.repeats} random splits, tested on held-out photos)")
        print(" feedbacks | accuracy (mean +- std) | mean reward")
        for j, c in enumerate(cps):
            print(f" {c:9d} | {acc[:, j].mean():8.1%} +- {acc[:, j].std():5.1%}    | {rew[:, j].mean():7.2f}")

    with open("learning_curve.csv", "w") as f:
        f.write("mode,feedbacks,accuracy_mean,accuracy_std,reward_mean\n")
        for mode, (cps, acc, rew) in curves.items():
            for j, c in enumerate(cps):
                f.write(f"{mode},{c},{acc[:, j].mean():.4f},{acc[:, j].std():.4f},{rew[:, j].mean():.3f}\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for mode, (cps, acc, rew) in curves.items():
            lab = "feedback with label" if mode == "label" else "reward only (pure RL)"
            ax[0].errorbar(cps, acc.mean(0) * 100, acc.std(0) * 100, marker="o", capsize=3, label=lab)
            ax[1].plot(cps, rew.mean(0), marker="o", label=lab)
        ax[0].set_title("Accuracy on held-out photos"); ax[0].set_xlabel("feedbacks given"); ax[0].set_ylabel("accuracy (%)")
        ax[1].set_title("Average reward on held-out photos"); ax[1].set_xlabel("feedbacks given"); ax[1].set_ylabel("reward")
        for a_ in ax:
            a_.grid(True); a_.legend()
        plt.tight_layout(); plt.savefig("learning_curve.png", dpi=150)
        print("\nsaved learning_curve.png and learning_curve.csv")
    except ImportError:
        print("\nsaved learning_curve.csv  (run `pip install matplotlib` to also get the graph)")


if __name__ == "__main__":
    main()
