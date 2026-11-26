"""
train_fast.py - run MANY epochs on a laptop CPU in minutes instead of hours.

Why it is fast: the early ResNet layers never change, so their output is computed ONCE
(for the clean photo and several augmented copies of it) and kept in RAM. Each epoch then
only trains the small last part of the network on those saved outputs.

Put this file next to app.py and run, one line at a time:
    python train_fast.py --quick                 # ~2 minutes, only checks that everything runs (saves nothing)
    python train_fast.py --mode head --epochs 150    # a few minutes in total: 150 epochs, classifier layer only
    python train_fast.py --mode block --epochs 30    # best accuracy: also fine-tunes the last ResNet block

FASTER REPEAT RUNS:
    python train_fast.py --mode block --epochs 50 --views 3 --cache
        the first run pays the one-time setup (about 5 min with 3 views) and saves it; any later run with the same
        mode/views/data (for example --epochs 30 or 80) loads it from disk and starts training in seconds.

MORE DATA (what your sir asked for):
    python train_fast.py --mode block --epochs 40 --real-train
        also trains on HALF of your real photos (real_photos/<Disease>/ and feedback_data/<Disease>/, shown 5x each);
        the other half is kept aside, so the printed real-photo accuracy stays honest.
    python train_fast.py --mode block --epochs 40 --real-train --final
        FINAL model: uses ALL Kaggle photos (train + validation + test) for training, ~43% more data.
        Run this last, after the normal run has told you the honest accuracy. In this mode the validation/test
        numbers are NOT honest any more (those photos are in training); only the real-photo number still is.

What it does (same honest setup as train_check.py):
  * removes duplicate / near-identical copies of a photo
  * splits every class by filename order = photo time: oldest 70% train, next 10% validation, newest 20% test
  * starts from ImageNet weights (NOT from your old mango model, which had already seen many test photos)
  * keeps the best epoch by VALIDATION accuracy; early stopping is off unless you pass --patience N
  * prints the same epoch table as before and also writes it to epochs.txt after EVERY epoch
    (live graphs while it trains: open a second terminal and run  python plot_epochs.py --watch 30
     final graphs afterwards:                                      python plot_epochs.py --test <test accuracy>)
  * prints test accuracy, precision/recall per class, confusion matrix, and accuracy on your real photos
    (folders feedback_data/<Disease>/ and real_photos/<Disease>/)
  * saves mango_resnet_model.pth + anchors.npz (the old model is kept as mango_resnet_model_prev.pth)
"""
import os
import sys
import time
import shutil
import hashlib
import argparse
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
CLASS_NAMES = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back',
               'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould']
EXT = (".jpg", ".jpeg", ".png")
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ data (no torch needed) ----
def _has_classes(d):
    return all(os.path.isdir(os.path.join(d, c)) for c in CLASS_NAMES)


def find_dataset(arg=None):
    if arg:
        if _has_classes(arg):
            return arg
        sys.exit(f"'{arg}' does not contain the 8 class folders ({', '.join(CLASS_NAMES)}).")
    for dirpath, dirnames, _ in os.walk(HERE):
        depth = dirpath[len(HERE):].count(os.sep)
        if depth > 3:
            dirnames[:] = []
            continue
        if _has_classes(dirpath):
            return dirpath
    sys.exit("Could not find the MangoLeafBD folder next to this script. Use:  python train_fast.py --data \"C:\\path\\to\\MangoLeafBD Dataset\"")


def dhash(path, size=16):
    """Tiny fingerprint of a photo: copies and re-saves of the same photo get (almost) the same one."""
    im = Image.open(path).convert("L").resize((size + 1, size), Image.BILINEAR)
    a = np.asarray(im, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).reshape(-1)


def list_and_dedupe(root, threshold=6, max_per_class=None):
    per_class, removed = [], 0
    for name in CLASS_NAMES:
        folder = os.path.join(root, name)
        files = sorted(f for f in os.listdir(folder) if f.lower().endswith(EXT))   # camera numbering = time order
        if max_per_class:
            files = files[:max_per_class]
        kept, H = [], []
        for f in files:
            p = os.path.join(folder, f)
            h = dhash(p)
            if H and (np.array(H) != h).sum(axis=1).min() <= threshold:
                removed += 1
                continue
            kept.append(p); H.append(h)
        per_class.append(kept)
    return per_class, removed


def split_per_class(per_class):
    """Oldest 70% -> train, next 10% -> validation, newest 20% -> test (per class)."""
    tr, va, te = [], [], []
    for ci, files in enumerate(per_class):
        n = len(files)
        if n < 4:
            sys.exit(f"Class '{CLASS_NAMES[ci]}' has only {n} usable images.")
        a = max(1, int(n * 0.7))
        b = min(n - 1, a + max(1, int(n * 0.1)))
        tr += [(p, ci) for p in files[:a]]
        va += [(p, ci) for p in files[a:b]]
        te += [(p, ci) for p in files[b:]]
    return tr, va, te


def confusion(y, p, k):
    M = np.zeros((k, k), dtype=int)
    for t, q in zip(y, p):
        M[int(t), int(q)] += 1
    return M


def print_report(M):
    tp = np.diag(M)
    support, predicted = M.sum(1), M.sum(0)
    prec, rec = tp / np.maximum(predicted, 1), tp / np.maximum(support, 1)
    log("\nper-class test results:")
    log(f"  {'class':18s} precision  recall   images")
    for i, n in enumerate(CLASS_NAMES):
        log(f"  {n:18s} {prec[i]:8.1%} {rec[i]:8.1%}   {support[i]}")
    log(f"  {'macro average':18s} {prec.mean():8.1%} {rec.mean():8.1%}")
    log("\nconfusion matrix (rows = true, cols = predicted):")
    log("       " + " ".join(f"{n[:4]:>4s}" for n in CLASS_NAMES))
    for i, n in enumerate(CLASS_NAMES):
        log(f"{n[:5]:6s} " + " ".join(f"{v:4d}" for v in M[i]))


def collect_real():
    """Your own photos: feedback_data/<Disease>/ and real_photos/<Disease>/ -> (folder, disease name, class index, path)."""
    real = []
    for folder in ("feedback_data", "real_photos"):
        for ci, name in enumerate(CLASS_NAMES):
            d = os.path.join(HERE, folder, name)
            if os.path.isdir(d):
                real += [(folder, name, ci, os.path.join(d, f)) for f in sorted(os.listdir(d)) if f.lower().endswith(EXT)]
    return real


def split_real(real):
    """Every second photo of each folder goes to training, the others stay a held-out test set."""
    groups, train, test = {}, [], []
    for it in real:
        groups.setdefault((it[0], it[1]), []).append(it)
    for items in groups.values():
        for j, it in enumerate(items):
            (train if j % 2 == 0 else test).append(it)
    return train, test


# ------------------------------------------------------------------------------- training ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="MangoLeafBD folder (found automatically if it is next to this script)")
    ap.add_argument("--mode", choices=["head", "block"], default="block",
                    help="head = only the classifier layer (fastest);  block = classifier + last ResNet block (more accurate)")
    ap.add_argument("--epochs", type=int, default=None, help="default: 150 for head, 30 for block")
    ap.add_argument("--views", type=int, default=None, help="augmented copies kept per photo (default: 8 head, 6 block)")
    ap.add_argument("--patience", type=int, default=0, help="stop early after N epochs without improvement (0 = never stop early)")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--dup-threshold", type=int, default=8, help="max differing bits (of 256) to call two photos duplicates")
    ap.add_argument("--real-train", action="store_true", help="also train on half of your real photos (the other half stays a test set)")
    ap.add_argument("--real-weight", type=int, default=5, help="how many times each real training photo is repeated (each repeat gets different augmentations)")
    ap.add_argument("--final", action="store_true", help="train on ALL Kaggle photos (no honest val/test number any more); uses the last epoch")
    ap.add_argument("--cache", action="store_true",
                    help="save the slow one-time setup (frozen-layer outputs, ~0.5-1 GB) to disk and reuse it on the next run with the same settings")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="tiny test run to check everything works (saves nothing)")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    try:
        torchvision = __import__("torchvision", fromlist=["models", "transforms"])
    except ImportError as exc:
        sys.exit("torchvision is required. Install it in this Python environment with: "
                 "python -m pip install torchvision\n" + str(exc))
    models, transforms = torchvision.models, torchvision.transforms

    torch.manual_seed(0); np.random.seed(0)
    if args.threads:
        torch.set_num_threads(args.threads)
    mode = args.mode
    epochs = args.epochs or (150 if mode == "head" else 30)
    views = args.views or (8 if mode == "head" else 6)
    if args.quick:
        epochs, views, args.no_save = 3, 2, True

    root = find_dataset(args.data)
    log(f"dataset: {root}   | mode: {mode} | epochs: {epochs} | augmented views per photo: {views} | threads: {torch.get_num_threads()}")
    t0 = time.time()
    per_class, removed = list_and_dedupe(root, args.dup_threshold, 12 if args.quick else None)
    train_l, val_l, test_l = split_per_class(per_class)
    log(f"images after removing {removed} duplicate copies: {sum(len(c) for c in per_class)}")
    log(f"train {len(train_l)} | val {len(val_l)} | test {len(test_l)}   (test = newest photos of each class)   [{time.time() - t0:.0f}s]")
    real_all = collect_real()
    real_train, real_test = [], real_all
    if args.real_train and real_all:
        real_train, real_test = split_real(real_all)
        train_l = train_l + [(it[3], it[2]) for it in real_train] * args.real_weight
        log(f"real photos: {len(real_train)} used for training (x{args.real_weight}), {len(real_test)} kept aside as an honest test")
    elif args.real_train:
        log("--real-train given but no photos found in real_photos/ or feedback_data/")
    if args.final:
        train_l = train_l + val_l + test_l
        args.patience = 0
        log("FINAL mode: ALL Kaggle photos are used for training -> train size", len(train_l),
            "\n   the validation / test numbers below are NOT honest (those photos are in training).")

    def load(path):
        return Image.open(path).convert("RGB").resize((224, 224), Image.BILINEAR)

    train_img = [load(p) for p, _ in train_l]; ytr = torch.tensor([c for _, c in train_l])
    val_img = [load(p) for p, _ in val_l];     yva = torch.tensor([c for _, c in val_l])
    test_img = [load(p) for p, _ in test_l];   yte = torch.tensor([c for _, c in test_l])

    to_t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    aug = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)), transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(), transforms.RandomRotation(30),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.05),
        transforms.RandomApply([transforms.GaussianBlur(5, (0.1, 2.0))], p=0.3)])

    net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)   # ImageNet start -> honest test
    net.fc = nn.Linear(net.fc.in_features, len(CLASS_NAMES))
    for p in net.parameters():
        p.requires_grad = False
    if mode == "block":
        for p in net.layer4[1].parameters():
            p.requires_grad = True
    for p in net.fc.parameters():
        p.requires_grad = True
    net.eval()

    # the part of the network that never changes (its output is computed once and kept)
    if mode == "block":
        stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2, net.layer3, net.layer4[0])
        top = lambda x: net.fc(torch.flatten(net.avgpool(net.layer4[1](x)), 1))
        shape = (512, 7, 7)
    else:
        stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2, net.layer3, net.layer4)
        top = lambda x: net.fc(x)
        shape = (512,)

    def encode(imgs, augment):
        out = torch.empty((len(imgs),) + shape, dtype=torch.float16)
        with torch.no_grad():
            for i in range(0, len(imgs), 64):
                x = torch.stack([to_t(aug(im) if augment else im) for im in imgs[i:i + 64]])
                f = stem(x)
                if mode == "head":
                    f = torch.flatten(net.avgpool(f), 1)
                out[i:i + len(x)] = f.half()
        return out

    n = len(train_img)
    mem_gb = n * views * int(np.prod(shape)) * 2 / 1e9
    sig = hashlib.md5("|".join([mode, str(views)] + [f"{os.path.basename(p)}:{c}" for p, c in train_l + val_l + test_l]).encode()).hexdigest()[:10]
    cache_file = os.path.join(HERE, f"feature_cache_{mode}_{views}_{sig}.pt")
    if args.cache and os.path.exists(cache_file):
        log(f"\nstep 1/2: loading the saved setup {os.path.basename(cache_file)} (skips the slow part) ...")
        saved = torch.load(cache_file)
        cache, val_f, test_f = saved["cache"], saved["val"], saved["test"]
        log("   loaded.")
    else:
        log(f"\nstep 1/2: computing the frozen layers once (clean + {views - 1} augmented copies per photo, about {mem_gb:.1f} GB of RAM) ...")
        if mem_gb > 3:
            log("   WARNING: that is a lot of RAM. If it crashes, rerun with a smaller --views (e.g. --views 3).")
        t1 = time.time()
        cache = torch.empty((n, views) + shape, dtype=torch.float16)
        for v in range(views):
            cache[:, v] = encode(train_img, augment=(v > 0))
            log(f"   view {v + 1}/{views} done   [{time.time() - t1:.0f}s]")
        val_f, test_f = encode(val_img, False), encode(test_img, False)
        log(f"   finished in {time.time() - t1:.0f}s")
        if args.cache:
            torch.save({"cache": cache, "val": val_f, "test": test_f}, cache_file)
            log(f"   saved to {os.path.basename(cache_file)} - the next run with --cache and the same settings skips this step")

    params = [p for p in net.parameters() if p.requires_grad]
    if mode == "block":
        lr = args.lr or 1e-4
        opt = torch.optim.AdamW([{"params": list(net.layer4[1].parameters()), "lr": lr},
                                 {"params": list(net.fc.parameters()), "lr": lr * 10}], weight_decay=1e-4)
    else:
        opt = torch.optim.AdamW(params, lr=args.lr or 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    def predict(feats):
        net.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(feats), 128):
                outs.append(F.softmax(top(feats[i:i + 128].float()), dim=1))
        return torch.cat(outs)

    log(f"\nstep 2/2: training {epochs} epochs\n")
    header = "epoch | train loss | train acc | val acc | gap(train-val)"
    log(header)
    rows = [header]
    base_lr = args.lr or (1e-4 if mode == "block" else 1e-3)          # same value the optimizer uses (for the graphs)
    meta = f"# mode={mode} planned_epochs={epochs} lr={base_lr:g}"     # first line of epochs.txt: lets plot_epochs.py draw the schedule
    best_val, best_ep, best_state, stale = -1.0, 0, None, 0
    for ep in range(1, epochs + 1):
        t_ep = time.time()
        if mode == "block":
            net.layer4[1].train()
        net.fc.train()
        perm = torch.randperm(n)
        view_of = torch.randint(0, views, (n,))
        loss_sum, correct, seen = 0.0, 0, 0
        for i in range(0, n, 64):
            idx = perm[i:i + 64]
            if len(idx) < 2:
                continue                                   # BatchNorm needs at least 2 samples
            x = cache[idx, view_of[idx]].float()
            y = ytr[idx]
            logits = top(x)
            loss = crit(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item() * len(idx); correct += (logits.argmax(1) == y).sum().item(); seen += len(idx)
        sched.step()
        tr_acc = 100.0 * correct / seen
        va_acc = 100.0 * (predict(val_f).argmax(1) == yva).float().mean().item()
        row = f"{ep:5d} | {loss_sum / seen:10.4f} | {tr_acc:8.2f}% | {va_acc:6.2f}% | {tr_acc - va_acc:+.2f}%"
        rows.append(row); log(row)
        if not args.quick:                                   # a --quick test must not overwrite a real run's table
            with open(os.path.join(HERE, "epochs.txt"), "w") as f:
                f.write(meta + "\n" + "\n".join(rows) + "\n")
        if va_acc > best_val:
            best_val, best_ep, stale = va_acc, ep, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            stale += 1
        if ep == 1:
            log(f"      (epoch 1 took {time.time() - t_ep:.0f}s -> about {(time.time() - t_ep) * (epochs - 1) / 60:.0f} more minutes)")
        if args.patience and stale >= args.patience:
            log(f"early stop: validation did not improve for {args.patience} epochs")
            break

    if args.final:
        log(f"\nFINAL mode: keeping the last epoch ({ep})")
        best_ep, best_val = ep, va_acc
    else:
        net.load_state_dict(best_state)
    probs = predict(test_f)
    pred = probs.argmax(1)
    test_acc = 100.0 * (pred == yte).float().mean().item()
    log("\n================ RESULT (best epoch by validation) ================")
    log(f"best epoch      : {best_ep}   (validation accuracy {best_val:.2f}%)")
    log(f"TEST accuracy   : {test_acc:.2f}%   " + ("(NOT honest: these photos were in training)" if args.final else "(newest, unseen photo sessions)"))
    log(f"mean confidence on test: {100 * probs.max(1).values.mean().item():.2f}%")
    print_report(confusion(yte.numpy(), pred.numpy(), len(CLASS_NAMES)))

    # accuracy on your own real photos (only the ones NOT used for training)
    net.eval()
    real = real_test
    if real:
        ok = 0
        log("\nreal photos that were NOT used for training:" if args.real_train else "\nreal photos (not used for training):")
        for folder, name, ci, path in real:
            with torch.no_grad():
                p = F.softmax(net(to_t(load(path)).unsqueeze(0)), dim=1)[0]
            guess = int(p.argmax()); ok += int(guess == ci)
            log(f"  [{folder}] {name:16s} -> {CLASS_NAMES[guess]:16s} {p[guess].item():4.0%}  {'ok' if guess == ci else 'WRONG'}")
        log(f"REAL PHOTO accuracy: {ok}/{len(real)} = {100 * ok / len(real):.0f}%   (very few photos = only a rough number)")
    else:
        log("\n(no real photos to test on - add some to real_photos/<Disease>/ to see real-photo accuracy)")

    if args.no_save:
        log("\nnothing saved (--quick / --no-save).")
        return
    old = os.path.join(HERE, "mango_resnet_model.pth")
    if os.path.exists(old):
        prev = os.path.join(HERE, "mango_resnet_model_prev.pth")
        if os.path.exists(prev):
            prev = os.path.join(HERE, "mango_resnet_model_prev_" + time.strftime("%Y%m%d_%H%M%S") + ".pth")
        shutil.copy(old, prev)
    torch.save({k: v.cpu() for k, v in net.state_dict().items()}, old)
    # anchors: 30 clean training photos per class as 512-d features (replayed by the RL agent)
    fc_backup, net.fc = net.fc, nn.Identity()
    xs, ys, taken = [], [], {}
    for im, (_, c) in zip(train_img, train_l):
        if taken.get(c, 0) < 30:
            with torch.no_grad():
                xs.append(net(to_t(im).unsqueeze(0)).numpy()[0]); ys.append(c)
            taken[c] = taken.get(c, 0) + 1
    net.fc = fc_backup
    np.savez(os.path.join(HERE, "anchors.npz"), x=np.array(xs), y=np.array(ys))
    log(f"\nsaved mango_resnet_model.pth + anchors.npz {np.array(xs).shape}  (old model kept as mango_resnet_model_prev.pth)")
    log("NOW delete agent_state.npz and rl_memory.json (keep feedback_data), then run the app.")
    log(f"For the graph:  python plot_epochs.py --test {test_acc:.2f}")


if __name__ == "__main__":
    main()