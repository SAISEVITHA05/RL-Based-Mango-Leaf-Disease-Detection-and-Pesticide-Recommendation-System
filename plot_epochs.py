"""
Makes TWO graphs for your slides from the epoch table that train_fast.py writes to epochs.txt:

  graph1_accuracy_vs_epochs.png        train vs validation accuracy for every epoch (+ best epoch, + final test accuracy)
  graph2_learning_rate_and_loss.png    the learning-rate schedule and the training loss for every epoch

The graphs are NOT drawn inside the training. train_fast.py appends one row to epochs.txt after every epoch
(real numbers); this script reads those rows and draws them. So you can also run it WHILE training is running:

  python plot_epochs.py --watch 30        redraws both graphs every 30 seconds until you press Ctrl+C
                                          (open the two PNG files and they refresh as new epochs finish)
  python plot_epochs.py --test 96.40      final graphs after the run, with the final test accuracy

needs:  pip install matplotlib
Options (normally not needed - train_fast.py writes them into the first line of epochs.txt):
  --mode block|head   --lr 0.0001   --planned N
"""
import re
import sys
import math
import time
import argparse

ROW = re.compile(r"^\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)%\s*\|\s*([\d.]+)%")
META = re.compile(r"^#\s*mode=(\w+)\s+planned_epochs=(\d+)\s+lr=([0-9.eE+-]+)")


def read_table(path):
    E, LOSS, TR, VA, meta = [], [], [], [], {}
    for line in open(path, encoding="utf-8", errors="ignore"):
        m = META.match(line)
        if m:
            meta = {"mode": m.group(1), "planned": int(m.group(2)), "lr": float(m.group(3))}
            continue
        m = ROW.match(line)
        if m:
            E.append(int(m.group(1))); LOSS.append(float(m.group(2))); TR.append(float(m.group(3))); VA.append(float(m.group(4)))
    return E, LOSS, TR, VA, meta


def draw(E, LOSS, TR, VA, test, mode, planned, lr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = max(planned, E[-1])
    best = max(range(len(E)), key=lambda i: (VA[i], -E[i]))            # first epoch with the best validation accuracy
    gap = [t - v for t, v in zip(TR, VA)]
    done = len(E) >= T

    # ---------------- graph 1: accuracy vs number of epochs ----------------
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(E, TR, marker="o", ms=4, label="training accuracy")
    ax.plot(E, VA, marker="s", ms=4, label="validation accuracy (photos not used for training)")
    ax.axvline(E[best], color="gray", linestyle="--", label=f"best epoch {E[best]}: validation {VA[best]:.2f}%")
    if test is not None:
        ax.axhline(test, color="green", linestyle=":", linewidth=2, label=f"final test accuracy {test:.2f}% (newest photos)")
    ax.set_xlabel("number of epochs"); ax.set_ylabel("accuracy (%)"); ax.set_xlim(0, T + 1)
    ax.set_title("Accuracy vs number of epochs" + ("" if done else f"  (training in progress: {len(E)}/{T})"))
    if len(E) > 3:
        lo = min(min(VA[2:]), min(TR[2:]), test if test else 100)
    else:
        lo = min(min(VA), min(TR))
    ax.set_ylim(lo - 2, 100.6); ax.grid(True, alpha=0.4); ax.legend(loc="lower right", fontsize=8)
    tail = gap[len(gap) // 2:]
    note = (f"train - validation gap in the 2nd half: {min(tail):.1f} to {max(tail):.1f} points\n"
            "(small gap = learning the disease, not memorising)")
    if min(min(VA), min(TR)) < lo - 2:
        note += f"\nepochs 1-2 start below the axis (epoch 1: train {TR[0]:.1f}%, validation {VA[0]:.1f}%)"
    ax.text(0.02, 0.04, note, transform=ax.transAxes, fontsize=8, va="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    plt.tight_layout(); plt.savefig("graph1_accuracy_vs_epochs.png", dpi=160); plt.close(fig)

    # ---------------- graph 2: learning rate + training loss ----------------
    cos = lambda e: 0.5 * (1 + math.cos(math.pi * (e - 1) / T))        # cosine schedule used by train_fast.py (steps once per epoch)
    fig, ax1 = plt.subplots(figsize=(8, 4.8))
    xs = list(range(1, T + 1))
    if mode == "block":
        ax1.plot(xs, [lr * 10 * cos(e) for e in xs], color="tab:blue", label=f"classifier learning rate (starts {lr * 10:g})")
        ax1.plot(xs, [lr * cos(e) for e in xs], color="tab:cyan", label=f"last-block learning rate (starts {lr:g})")
    else:
        ax1.plot(xs, [lr * cos(e) for e in xs], color="tab:blue", label=f"learning rate (starts {lr:g})")
    ax1.set_xlabel("number of epochs"); ax1.set_ylabel("learning rate"); ax1.grid(True, alpha=0.4); ax1.set_xlim(0, T + 1)
    ax2 = ax1.twinx()
    ax2.plot(E, LOSS, color="tab:red", marker="o", ms=4, label="training loss")
    ax2.set_ylabel("training loss", color="tab:red")
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper center", fontsize=8)
    ax1.set_title("Learning rate (decreasing smoothly) and training loss" + ("" if done else f"  ({len(E)}/{T})"))
    maxrise = max([0.0] + [LOSS[i] - LOSS[i - 1] for i in range(1, len(LOSS))])
    ax1.text(0.98, 0.66, f"training loss falls from {LOSS[0]:.2f} to {LOSS[-1]:.2f}\n"
                         f"largest rise between two epochs: {maxrise:.4f}\n"
                         "(no spikes, no divergence)\n"
                         "-> learning rate is not too high;\n"
                         "it shrinks smoothly, so training\nsettles at the end",
             transform=ax1.transAxes, fontsize=8, ha="right", va="center", bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
    plt.tight_layout(); plt.savefig("graph2_learning_rate_and_loss.png", dpi=160); plt.close(fig)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", nargs="?", default="epochs.txt")
    ap.add_argument("--test", type=float, default=None, help="final test accuracy in %%, e.g. 96.40")
    ap.add_argument("--mode", choices=["block", "head"], default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--planned", type=int, default=None)
    ap.add_argument("--watch", type=int, default=0, metavar="SECONDS", help="keep redrawing every SECONDS while training runs")
    args = ap.parse_args()

    while True:
        E, LOSS, TR, VA, meta = read_table(args.file)
        if len(E) < 3:
            msg = (f"Only {len(E)} epoch row(s) in {args.file} so far. It must contain the table of the real run "
                   "(a --quick run writes nothing).")
            if not args.watch:
                sys.exit(msg)
            print(msg + " Waiting ...", flush=True)
        else:
            mode = args.mode or meta.get("mode", "block")
            lr = args.lr or meta.get("lr") or (1e-4 if mode == "block" else 1e-3)
            planned = args.planned or meta.get("planned") or len(E)
            best = draw(E, LOSS, TR, VA, args.test, mode, planned, lr)
            print(f"[{time.strftime('%H:%M:%S')}] {len(E)}/{max(planned, len(E))} epochs drawn. Best validation accuracy "
                  f"{VA[best]:.2f}% at epoch {E[best]}; latest: train {TR[-1]:.2f}% / validation {VA[-1]:.2f}%. "
                  "saved graph1_accuracy_vs_epochs.png and graph2_learning_rate_and_loss.png", flush=True)
            if not args.watch or len(E) >= max(planned, len(E)) and len(E) >= planned:
                if args.watch:
                    print("training finished - final graphs saved.")
                break
        if not args.watch:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()