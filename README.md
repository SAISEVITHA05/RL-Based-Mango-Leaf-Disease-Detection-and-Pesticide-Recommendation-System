# 🥭 RL-Based Mango Leaf Disease Detection & Pesticide Recommendation System

A web app that detects **7 mango leaf diseases and healthy leaves** from one photo, recommends the
**pesticide, dose and step-by-step treatment**, and uses a **reinforcement learning agent** that
learns from user feedback and says "not sure" when it is uncertain.

**Author:** K. Sai Sevitha (S20240030381) · **Mentor:** Dr. Pavan Kumar B N · IIIT Sri City · Additional Project, 2026

---

## Highlights

| Result | Value |
|---|---|
| Test accuracy (750 unseen photos, time-based split) | **96.3%** |
| Balanced accuracy (8 classes) | **96.49%** |
| Duplicate photos found and removed (data leakage) | **281** |
| 30 vs 50 epochs (test accuracy) | 96.40% vs 96.27%, so no overfitting |
| RL feedback: accuracy on unseen photos | **79.1% → 94.4%** (label), 92.2% (reward-only) |
| Mango photos accepted by the leaf gate | 100% |
| Other plants / non-leaf photos stopped | ~92% / ~87–89% |

**Classes:** Anthracnose, Bacterial Canker, Cutting Weevil, Die Back, Gall Midge, Healthy, Powdery Mildew, Sooty Mould

---

## How it works

```
Photo → input checks → CLIP leaf gate → ResNet18 features → RL agent → result + treatment
                                                                ↑                  │
                                                     agent update ← user feedback = reward
```

1. **Input checks:** fixes rotation, handles PNG/WEBP/HEIC, warns about dark or blurry photos.
2. **Leaf gate (CLIP ViT-B/32):** stops photos that are not leaves, or not mango leaves.
3. **Classifier:** ResNet18 pretrained on ImageNet, last block fine-tuned on MangoLeafBD.
4. **RL agent (contextual bandit):**
   - State = 512 image features; action = one of 8 classes or "not sure"
   - Reward from the user: **+10** right, **−20** wrong, **−2** "not sure" on a known disease
   - Label feedback = reward-weighted update; reward-only feedback = REINFORCE
   - Replays 30 clean photos per class, so it does not forget
   - Learns its own "not sure" threshold; ε-greedy exploration (30% → 5%)
5. **Treatment:** top-3 cards when unsure; a step-by-step plan (remove → spray + dose → repeat → prevent) after confirmation.

### Leakage-safe evaluation
MangoLeafBD contains duplicate and burst photos. A random split put copies of the same photo in train
**and** test, which gave a fake 100%. This project removes 281 near-duplicates and splits **by photo time**
(oldest 70% train, next 10% validation, newest 20% test).

---

## Project structure

```
├── app.py                 # Streamlit web app
├── rl_agent.py            # RL agent (contextual bandit)
├── leaf_gate.py           # CLIP leaf + mango check
├── mango_probe.npz        # small classifier for the mango check
├── mango_resnet_model.pth # trained ResNet18 model
├── anchors.npz            # replay examples for the RL agent
├── train_fast.py          # training (leakage-safe split, many epochs on CPU)
├── plot_epochs.py         # accuracy / learning-rate graphs
├── check_learning.py      # tests whether the RL agent learns from feedback
├── requirements.txt
└── .streamlit/config.toml
```

---

## Setup and run

Requires Python 3.10 or newer.

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The first start downloads the CLIP weights (about 600 MB) once.

**requirements.txt**
```
torch
torchvision
streamlit>=1.40
numpy
pandas
pillow
pillow-heif
open_clip_torch
matplotlib
```

---

## Train and evaluate (optional)

Download the dataset and place the folder `MangoLeafBD Dataset` (with the 8 class folders) next to the scripts.
Dataset: [MangoLeafBD on Kaggle](https://www.kaggle.com/datasets/warcoder/mango-leaf-disease-dataset)

```bash
python train_fast.py --mode block --epochs 30   # trains and saves mango_resnet_model.pth + anchors.npz
python plot_epochs.py --test 96.40              # accuracy and learning-rate graphs
python check_learning.py                        # RL learning test on unseen photos
```

After retraining, the app rebuilds the RL agent automatically.

---

## Tech stack
Python · PyTorch · torchvision · OpenCLIP · Streamlit · NumPy · Pandas · Pillow · Matplotlib

## Limitations
- Accuracy was measured on dataset photos; real field photos have been tested only on a small set.
- Treatment steps are general guidance; **confirm with an agriculture officer or Krishi Vigyan Kendra before spraying** and follow the product label.

## Future work
- Collect 50–100 expert-labelled field photos per class and retrain
- Learn the pesticide dosage as an RL decision
- Cite treatments from ICAR guides
- Deploy online (Hugging Face Spaces)
- Publish a research paper

## References
1. Mohanty, Hughes, Salathé, *Frontiers in Plant Science*, 2016
2. Ferentinos, *Computers and Electronics in Agriculture*, 2018
3. Singh et al., *IEEE Access*, 2019
4. Ahmed et al., MangoLeafBD, *Data in Brief*, 2023
5. Abbigeri & Devanagavi, *ETASR* 16(2), 2026
6. Marathe et al., *Springer Nature*, 2026
