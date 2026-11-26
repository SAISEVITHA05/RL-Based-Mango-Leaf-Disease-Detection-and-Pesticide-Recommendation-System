import io
import os
import json
import hashlib
import shutil
import stat
import sys
import time
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image, ImageOps

try:                                   # iPhone .heic photos (optional: python -m pip install pillow-heif)
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIC_OK = True
except Exception:
    HEIC_OK = False

from rl_agent import RLAgent

class_names = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back',
               'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould']

remedy_table = {
    "Anthracnose": {"cause": "Fungus (Colletotrichum gloeosporioides), spreads in humid/rainy conditions",
                    "pesticide": "Copper Oxychloride 50% WP", "dose": "300g per 100L water"},
    "Bacterial Canker": {"cause": "Bacterium (Xanthomonas campestris), causes fruit drop and yield loss",
                         "pesticide": "Copper Oxychloride 50% WP + Streptocycline", "dose": "300g/100L + 1g/10L water"},
    "Cutting Weevil": {"cause": "Insect larvae (Deporaus marginatus) tunneling into leaves",
                       "pesticide": "Chlorpyrifos 20% EC", "dose": "2.5ml per liter water"},
    "Die Back": {"cause": "Fungal infection (Botryodiplodia), often after pruning wounds/stress",
                 "pesticide": "Copper Oxychloride + pruning of dead twigs", "dose": "300g per 100L water"},
    "Gall Midge": {"cause": "Insect larvae (Procontarinia spp.) feeding inside leaf/flower tissue",
                   "pesticide": "Malathion 50% EC", "dose": "2ml per liter water"},
    "Healthy": {"cause": "-", "pesticide": "No treatment needed", "dose": "0"},
    "Powdery Mildew": {"cause": "Fungus (Oidium mangiferae), favors cool humid weather",
                       "pesticide": "Wettable Sulfur 80% WP", "dose": "2-3g per liter water"},
    "Sooty Mould": {"cause": "Fungus growing on honeydew from sap-sucking insects",
                    "pesticide": "Neem oil", "dose": "5ml per liter water"}
}

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(APP_DIR, "mango_resnet_model.pth")   # frozen feature extractor
ANCHOR_FILE = os.path.join(APP_DIR, "anchors.npz")             # optional replay examples
STATE_FILE = os.path.join(APP_DIR, "agent_state.npz")          # learned weights + feedback features
MEMORY_FILE = os.path.join(APP_DIR, "rl_memory.json")          # image ids -> label the user gave
LEGACY_MEMORY_FILE = os.path.join(APP_DIR, "feedback_memory.json")
FEEDBACK_DIR = os.path.join(APP_DIR, "feedback_data")
UNDO_DIR = os.path.join(APP_DIR, "undo_history")             # snapshots for "Undo last feedback"
UNDO_KEEP = 10
UPLOAD_TYPES = ["jpg", "jpeg", "png", "webp", "bmp", "heic", "heif"]
SAME_IMAGE_SIMILARITY = 0.98            # a re-saved / re-screenshotted copy of an image counts as the same image

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def _folder_label(folder_name):
    if folder_name in class_names:
        return class_names.index(folder_name)
    if folder_name.startswith("Not ") and folder_name[4:] in class_names:
        return -1 - class_names.index(folder_name[4:])     # user said: this guess was wrong
    return len(class_names)   # "none of these" / unknown


def replay_saved_feedback(resnet, agent):
    """Re-apply images stored under feedback_data if the on-disk agent state cannot be loaded."""
    if not os.path.isdir(FEEDBACK_DIR):
        return 0
    n = 0
    for folder in os.listdir(FEEDBACK_DIR):
        folder_path = os.path.join(FEEDBACK_DIR, folder)
        if not os.path.isdir(folder_path):
            continue
        label = _folder_label(folder)
        for fname in os.listdir(folder_path):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            try:
                image = Image.open(os.path.join(folder_path, fname)).convert("RGB")
            except Exception:
                continue                                    # skip unreadable files
            with torch.no_grad():
                feat = resnet(transform(image).unsqueeze(0)).numpy()[0]
            agent.observe(feat)
            if label < 0:
                agent.learn_from_reward(feat, -1 - label, -20.0)
            else:
                agent.learn(feat, label)
            n += 1
    return n


@st.cache_resource
def load_backbone():
    """Frozen ResNet only. The RL agent is reloaded from disk on every page run so a
    browser refresh cannot forget feedback."""
    with open(MODEL_FILE, "rb") as fh:           # fingerprint of the model file: agent state is tied to it
        model_id = hashlib.md5(fh.read()).hexdigest()
    resnet = models.resnet18(weights=None)
    resnet.fc = nn.Linear(resnet.fc.in_features, len(class_names))
    resnet.load_state_dict(torch.load(MODEL_FILE, map_location="cpu"))
    resnet.eval()
    W0 = resnet.fc.weight.detach().numpy().copy()
    b0 = resnet.fc.bias.detach().numpy().copy()
    resnet.fc = nn.Identity()
    anchors = None
    if os.path.exists(ANCHOR_FILE):
        with np.load(ANCHOR_FILE) as a:
            anchors = (np.array(a["x"], copy=True), np.array(a["y"], copy=True))
    return resnet, W0, b0, anchors, model_id


@st.cache_resource
def load_leaf_gate():
    """CLIP-based 'is this a leaf at all?' check (leaf_gate.py). Optional: the app works without it."""
    try:
        from leaf_gate import LeafGate
        return LeafGate(), None
    except Exception as exc:                 # open_clip not installed / no internet for the first download
        return None, f"{type(exc).__name__}: {exc}"


def build_agent(resnet, W0, b0, anchors, model_id):
    """Loads the saved agent only if it was learned on THIS model file. After retraining (new model file)
    the old state is ignored and the agent is rebuilt from the photos in feedback_data."""
    agent = RLAgent(W0, b0, class_names, anchors=anchors)
    if not agent.load(STATE_FILE, model_id):
        if os.path.exists(STATE_FILE):
            st.session_state["flash_info"] = ("The model file changed (or the saved state was unreadable), so the agent "
                                              "was rebuilt from your saved feedback photos.")
        replay_saved_feedback(resnet, agent)
        agent.save(STATE_FILE, model_id)
    return agent


def content_hash(image):
    """Same photo still matches after refresh even if the file bytes changed a little."""
    tiny = image.convert("RGB").resize((32, 32))
    return hashlib.md5(np.asarray(tiny).tobytes()).hexdigest()


def load_memory():
    memory = {}
    for path in (LEGACY_MEMORY_FILE, MEMORY_FILE):
        if os.path.exists(path):
            with open(path, "r") as f:
                memory.update(json.load(f))
    if os.path.isdir(FEEDBACK_DIR):
        for folder in os.listdir(FEEDBACK_DIR):
            folder_path = os.path.join(FEEDBACK_DIR, folder)
            if not os.path.isdir(folder_path):
                continue
            for fname in os.listdir(folder_path):
                stem, ext = os.path.splitext(fname)
                if ext.lower() in {".jpg", ".jpeg", ".png"} and stem:
                    memory.setdefault(stem, folder)       # the saved json (latest answer) wins
    return memory


def save_memory(memory):
    tmp = MEMORY_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(memory, f)
    os.replace(tmp, MEMORY_FILE)


def remember_label(memory, *ids, label_name=None):
    for key in ids:
        if key:
            memory[key] = label_name
    save_memory(memory)


def already_labeled(memory, *ids):
    for key in ids:
        if key in memory:
            return memory[key]
    return None


def save_feedback_image(image, label_name, image_hash):
    folder = os.path.join(FEEDBACK_DIR, label_name)
    os.makedirs(folder, exist_ok=True)
    image.save(os.path.join(folder, f"{image_hash}.jpg"))


# ------------------------------------------------------------------ input checks ----
def read_upload(data, filename):
    """Bytes -> clean RGB image. Fixes phone rotation (EXIF), puts transparent images on white,
    shrinks huge photos. Returns (image, error message)."""
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
    except Exception:
        if filename.lower().endswith((".heic", ".heif")) and not HEIC_OK:
            return None, ("This is an iPhone HEIC photo. Install support once with "
                          "`python -m pip install pillow-heif` and restart the app, or upload a JPG/PNG.")
        return None, "This file could not be read as an image (it may be damaged). Please upload a JPG or PNG photo."
    try:
        im = ImageOps.exif_transpose(im)               # phone photos: turn them the right way up
    except Exception:
        pass
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        im = bg
    else:
        im = im.convert("RGB")
    if max(im.size) > 2000:
        im.thumbnail((2000, 2000))
    return im, None


def photo_quality(im):
    """Returns (fatal problem or None, list of warnings). Limits were set so that none of the
    MangoLeafBD photos are flagged (their sharpness is >= 34, brightness 93-213, size >= 86 px)."""
    if min(im.size) < 64:
        return f"The image is too small ({im.size[0]}x{im.size[1]} pixels). Please upload a photo at least 64 pixels wide.", []
    g = im.convert("L")
    g.thumbnail((512, 512))
    a = np.asarray(g, dtype=np.float32)
    lap = a[1:-1, 1:-1] * -4 + a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:]
    warnings = []
    if a.mean() < 40:
        warnings.append("too dark")                    # (a dark photo also looks 'blurry', so check this first)
    elif a.mean() > 235:
        warnings.append("too bright / washed out")
    elif lap.var() < 15:
        warnings.append("very blurry")
    return None, warnings


def safe_remove(path):
    """Delete a file or folder. On Windows (especially inside OneDrive) a file can be locked for a moment
    by the sync/antivirus, so retry a few times and never crash the app. Returns True if it is gone."""
    def _unlock(func, p, _exc):
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except Exception:
            pass
    for _ in range(5):
        if not os.path.exists(path):
            return True
        try:
            if os.path.isdir(path):
                if sys.version_info >= (3, 12):
                    shutil.rmtree(path, onexc=_unlock)
                else:
                    shutil.rmtree(path, onerror=_unlock)
            else:
                os.chmod(path, stat.S_IWRITE)
                os.remove(path)
        except Exception:
            pass
        if not os.path.exists(path):
            return True
        time.sleep(0.3)
    return not os.path.exists(path)


def clear_folder(folder):
    """Empty a folder but keep the folder itself (OneDrive often refuses to delete a folder it watches)."""
    if not os.path.isdir(folder):
        return True
    for name in os.listdir(folder):
        safe_remove(os.path.join(folder, name))
    # empty sub-folders that Windows refused to delete are harmless: only leftover FILES count as failure
    return not any(files for _, _, files in os.walk(folder))


def move_aside(folder):
    """Move a folder to <folder>_backup_<time>. If Windows refuses the move, copy it and empty the original."""
    if not os.path.isdir(folder):
        return True
    dest = folder + "_backup_" + time.strftime("%Y%m%d_%H%M%S")
    try:
        shutil.move(folder, dest)
        return True
    except Exception:
        try:
            shutil.copytree(folder, dest, dirs_exist_ok=True)
        except Exception:
            return False
        return clear_folder(folder)


def undo_stack():
    if not os.path.isdir(UNDO_DIR):
        return []
    return sorted(int(f[:-5]) for f in os.listdir(UNDO_DIR) if f.endswith(".json"))


def push_undo(memory, ids, label_name, image_hash, description):
    """Called just BEFORE a feedback is applied: remembers how to put everything back."""
    os.makedirs(UNDO_DIR, exist_ok=True)
    stack = undo_stack()
    n = (stack[-1] + 1) if stack else 1
    if os.path.exists(STATE_FILE):
        shutil.copy(STATE_FILE, os.path.join(UNDO_DIR, f"{n}.npz"))
    img_path = os.path.join(FEEDBACK_DIR, label_name, f"{image_hash}.jpg")
    info = {"memory": {k: memory.get(k) for k in ids if k}, "image": img_path,
            "image_existed": os.path.exists(img_path), "description": description}
    with open(os.path.join(UNDO_DIR, f"{n}.json"), "w") as fh:
        json.dump(info, fh)
    for old in stack[:max(0, len(stack) + 1 - UNDO_KEEP)]:       # keep only the last few
        for ext in (".json", ".npz"):
            safe_remove(os.path.join(UNDO_DIR, f"{old}{ext}"))


def undo_last():
    stack = undo_stack()
    if not stack:
        return None
    n = stack[-1]
    with open(os.path.join(UNDO_DIR, f"{n}.json")) as fh:
        info = json.load(fh)
    snap = os.path.join(UNDO_DIR, f"{n}.npz")
    if os.path.exists(snap):
        shutil.copy(snap, STATE_FILE)
        safe_remove(snap)
    else:
        safe_remove(STATE_FILE)
    memory = {}
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE) as fh:
            memory = json.load(fh)
    for k, v in info["memory"].items():
        if v is None:
            memory.pop(k, None)
        else:
            memory[k] = v
    save_memory(memory)
    if not info["image_existed"]:
        safe_remove(info["image"])
        folder = os.path.dirname(info["image"])
        if os.path.isdir(folder) and not os.listdir(folder):
            safe_remove(folder)                      # an empty folder left behind is harmless
    safe_remove(os.path.join(UNDO_DIR, f"{n}.json"))
    return info["description"]


def last_undo_description():
    stack = undo_stack()
    if not stack:
        return None
    with open(os.path.join(UNDO_DIR, f"{stack[-1]}.json")) as fh:
        return json.load(fh)["description"]


# ------------------------------------------------------------------ layout helpers ----
HIGH, MEDIUM = 0.85, 0.70          # confidence levels for the result card

# pathogen type per class (shown as a small tag on each treatment card)
KIND = {"Anthracnose": "Fungus", "Bacterial Canker": "Bacteria", "Cutting Weevil": "Insect pest",
        "Die Back": "Fungus", "Gall Midge": "Insect pest", "Healthy": "Healthy",
        "Powdery Mildew": "Fungus", "Sooty Mould": "Fungus"}

# palette: leaf green = trust/primary, mango yellow = brand accent, ripe orange = caution, blight red = stop
STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Lexend:wght@400;500;600;700&display=swap');
:root { --leaf:#1E5B38; --leaf-dark:#143F27; --mango:#F4B400; --ripe:#D9661F; --blight:#B42318;
        /* surfaces are tints of the page's own text colour, so they work in Streamlit's light AND dark theme */
        --card: color-mix(in srgb, currentColor 5%, transparent);
        --sap: color-mix(in srgb, currentColor 13%, transparent);
        --line: color-mix(in srgb, currentColor 18%, transparent);
        --label:#2F8553; --safety-bg: rgba(244,180,0,0.14); --safety-line: rgba(244,180,0,0.55); }
h1, h2, h3, h4, .lexend { font-family: 'Lexend', sans-serif !important; letter-spacing: -0.01em; }
.block-container { padding-top: 3.4rem; max-width: 1180px; }
.stMarkdown h4 { font-size: 1.15rem !important; line-height: 1.35; }

/* header band */
.hero { background: var(--leaf-dark); border-radius: 18px; padding: 26px 30px 22px; color: #fff;
        border-bottom: 6px solid var(--mango); margin-bottom: 18px; }
.hero h1 { color: #fff !important; font-size: 2.05rem; line-height: 1.15; margin: 0 0 6px 0; padding: 0; font-weight: 600; }
.hero p { color: #DCE8DF; margin: 0 0 14px 0; font-size: 1.02rem; max-width: 70ch; }
.facts { display: flex; flex-wrap: wrap; gap: 8px; }
.fact { background: rgba(255,255,255,0.10); border: 1px solid rgba(255,255,255,0.22); border-radius: 999px;
        padding: 5px 12px; font-size: 0.88rem; color: #fff; }
.fact b { color: var(--mango); font-weight: 600; }

/* section headings that are a real sequence: result -> treatment -> feedback */
.step { display: flex; align-items: center; gap: 10px; margin: 22px 0 8px 0; }
.step .num { font-family: 'Lexend', sans-serif; background: var(--leaf); color: #fff; width: 30px; height: 30px;
             border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; font-weight: 600; }
.step .title { font-family: 'Lexend', sans-serif; font-size: 1.35rem; font-weight: 600; color: inherit; }

/* the diagnosis report */
.result-card { background: var(--card); color: inherit; border-radius: 16px; padding: 18px 22px 16px;
               border: 1px solid var(--line); border-left: 10px solid; margin: 0 0 12px 0; }
.result-card h2 { font-family: 'Lexend', sans-serif; color: inherit !important; margin: 4px 0 6px 0;
                  font-size: 1.85rem; line-height: 1.15; padding: 0; }
.result-card p { margin: 4px 0; font-size: 1rem; color: inherit; }
.result-card .level { display: inline-block; border-radius: 999px; padding: 3px 12px; font-size: 0.85rem;
                      font-weight: 600; color: #fff; }
.lvl-high   { border-left-color: var(--leaf); }   .lvl-high .level   { background: var(--leaf); }
.lvl-medium { border-left-color: var(--mango); }  .lvl-medium .level { background: #9A7200; }
.lvl-low    { border-left-color: var(--ripe); }   .lvl-low .level    { background: var(--ripe); }
.lvl-no     { border-left-color: var(--blight); } .lvl-no .level     { background: var(--blight); }
.meter { height: 12px; background: var(--sap); border-radius: 999px; overflow: hidden; margin: 10px 0 6px 0; }
.meter > span { display: block; height: 100%; background: var(--mango); border-radius: 999px; }
.meter.small { height: 8px; margin: 6px 0 10px 0; }

/* treatment cards */
.tcard { background: var(--card); color: inherit; border: 1px solid var(--line); border-radius: 14px;
         padding: 16px 18px; height: 100%; }
.tcard.top { border: 2px solid var(--label); }
.tcard h3 { color: inherit !important; font-size: 1.3rem; margin: 6px 0 2px 0; padding: 0; }
.tcard .pct { opacity: 0.75; font-size: 0.95rem; }
.tcard dl { margin: 4px 0 0 0; }
.tcard dt { font-weight: 600; color: var(--label); font-size: 0.9rem; margin-top: 8px; }
.tcard dd { margin: 0; font-size: 0.98rem; }
.tag { display: inline-block; border-radius: 6px; padding: 2px 8px; font-size: 0.8rem; font-weight: 600; margin-right: 6px; }
.tag.rank { background: var(--leaf); color: #fff; }
.tag.rank.other { background: var(--sap); color: inherit; }
.tag.Fungus { background: #FFF1CC; color: #7A5600; }
.tag.Bacteria { background: #FDE2D3; color: #8A3B0B; }
.tag.Insect { background: #E8E1F7; color: #4B3A7A; }
.tag.Healthy { background: #DDF0E2; color: #1E5B38; }
.safety { background: var(--safety-bg); color: inherit; border-radius: 12px; padding: 10px 14px;
          font-size: 0.92rem; margin-top: 12px; border: 1px solid var(--safety-line); }

/* treatment plan (a real sequence, so numbered) */
.plan { counter-reset: s; list-style: none; padding: 0; margin: 6px 0 0 0; }
.plan li { counter-increment: s; position: relative; padding: 12px 16px 12px 58px; margin: 0 0 10px 0;
           background: var(--card); border: 1px solid var(--line); border-radius: 14px; }
.plan li::before { content: counter(s); position: absolute; left: 14px; top: 12px; width: 30px; height: 30px;
                   border-radius: 50%; background: var(--mango); color: #3B2A00; font-family: 'Lexend', sans-serif;
                   font-weight: 600; display: flex; align-items: center; justify-content: center; }
.plan li b:first-child { display: block; font-family: 'Lexend', sans-serif; font-size: 1.05rem; margin-bottom: 2px; }
.plan .spray { color: var(--label); }
.lvl-ok { border-left-color: var(--leaf); } .lvl-ok .level { background: var(--leaf); }

/* empty state */
.howto { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-top: 6px; }
.howto div { background: var(--card); color: inherit; border: 1px solid var(--line); border-radius: 14px; padding: 14px 16px; }
.howto b { font-family: 'Lexend', sans-serif; color: var(--label); display: block; margin-bottom: 4px; }
</style>
"""


def stretch(fn, *args, **kwargs):
    """Full-width element on both new (width='stretch') and older (use_container_width) Streamlit."""
    try:
        return fn(*args, width="stretch", **kwargs)
    except Exception:
        return fn(*args, use_container_width=True, **kwargs)


def step(num, title):
    st.markdown(f'<div class="step"><span class="num">{num}</span><span class="title">{title}</span></div>',
                unsafe_allow_html=True)


def meter(p, small=False):
    pct = max(0.0, min(1.0, float(p))) * 100
    return f'<div class="meter{" small" if small else ""}"><span style="width:{pct:.1f}%"></span></div>'


def confidence_level(conf, not_sure):
    if not_sure or conf < MEDIUM:
        return "low", "Low confidence: not sure", "Most likely"
    if conf < HIGH:
        return "medium", "Medium confidence", "Likely"
    return "high", "High confidence", "Diagnosis"


def result_card(name, conf, level, level_text, heading):
    note = {"high": "The model is confident about this result.",
            "medium": "Probably correct. Compare it with the other close matches below.",
            "low": "The model is not sure. Compare the closest matches below and ask an agricultural expert "
                   "before spraying anything."}[level]
    st.markdown(f"""<div class="result-card lvl-{level}">
        <span class="level">{level_text}</span>
        <h2>{heading}: {name}</h2>
        {meter(conf)}
        <p>Confidence <b>{conf:.1%}</b></p>
        <p>{note}</p></div>""", unsafe_allow_html=True)


def treatment_html(disease):
    info = remedy_table.get(disease)
    if info is None:
        return "<p>No treatment listed for this class.</p>"
    if disease == "Healthy":
        return "<dl><dt>Result</dt><dd>No disease found. No spray needed.</dd></dl>"
    return (f"<dl><dt>Cause</dt><dd>{info['cause']}</dd>"
            f"<dt>Spray</dt><dd>{info['pesticide']}</dd>"
            f"<dt>Dose</dt><dd>{info['dose']}</dd></dl>")


def kind_tag(disease):
    k = KIND.get(disease, "")
    return f'<span class="tag {k.split()[0]}">{k}</span>' if k else ""


def match_card(rank, disease, p, is_top):
    rank_tag = '<span class="tag rank">Most likely</span>' if is_top else f'<span class="tag rank other">Match {rank}</span>'
    st.markdown(f"""<div class="tcard{' top' if is_top else ''}">
        {rank_tag}{kind_tag(disease)}
        <h3>{disease}</h3>
        <span class="pct">{p:.1%} probability</span>
        {meter(p, small=True)}
        {treatment_html(disease)}</div>""", unsafe_allow_html=True)


SAFETY = ('<div class="safety"><b>Before you spray:</b> this is a guide, not a prescription. Confirm the disease with '
          'your local agriculture officer or Krishi Vigyan Kendra, follow the dose on the product label, and wear '
          'gloves and a mask.</div>')


# step-by-step plans. Spray name and dose come from remedy_table; the rest is general orchard practice.
PLAN = {
    "Anthracnose": {
        "first": "Cut off and burn the spotted leaves and dried twigs. Pick up fallen leaves under the tree.",
        "when": "Spray at the first spots, and before and after rain, when the fungus spreads fastest.",
        "repeat": "Repeat about every 15 days while the weather stays wet.",
        "prevent": "Prune the canopy so air and sun get in, and avoid wetting the leaves when watering."},
    "Bacterial Canker": {
        "first": "Prune twigs with black, cracked spots and burn them. Clean the pruning tool between cuts.",
        "when": "Spray the whole canopy, especially new leaves and young fruit.",
        "repeat": "Repeat every 10 to 15 days, usually about 3 sprays.",
        "prevent": "Avoid wounding branches and fruit, and plant windbreaks, since wind-blown rain spreads it."},
    "Cutting Weevil": {
        "first": "Collect the cut leaves lying under the tree and burn them, since the eggs and grubs are inside.",
        "when": "Spray when new leaves (the flush) start to come out, which is when the weevil attacks.",
        "repeat": "Repeat only if fresh cut leaves keep appearing, as the product label allows.",
        "prevent": "Rake or dig the soil under the tree to expose grubs, and check every new flush."},
    "Die Back": {
        "first": "Cut each dried twig about 7 to 8 cm below the dead part and burn it. Cover the cut ends with copper paste.",
        "when": "Spray right after pruning.",
        "repeat": "Repeat about every 15 days until no new drying is seen.",
        "prevent": "Prune only in dry weather, feed and water the tree well, and avoid stress and wounds."},
    "Gall Midge": {
        "first": "Remove leaves and flower parts with galls (small bumps) and destroy them.",
        "when": "Spray when new leaves or flower buds start to open.",
        "repeat": "Repeat only if new galls appear, as the product label allows.",
        "prevent": "Dig the soil under the tree to expose the pupae, and keep the orchard free of weeds."},
    "Healthy": None,
    "Powdery Mildew": {
        "first": "Remove badly covered leaves and flower clusters if there are only a few.",
        "when": "Spray when the flower clusters (panicles) appear, the most important time.",
        "repeat": "Repeat about every 15 days, usually 2 to 3 sprays during flowering.",
        "prevent": "Watch closely in cool, cloudy, humid weather, and keep the canopy open by pruning."},
    "Sooty Mould": {
        "first": "Control the sap-sucking insects (hoppers, scales, mealybugs) first: their sticky honeydew feeds the mould.",
        "when": "Spray the leaves once the insects are under control, and wash the black layer off where you can.",
        "repeat": "Repeat after about 2 weeks if the black layer comes back.",
        "prevent": "Check leaf undersides for insects regularly, and control ants that protect them."},
}


def plan_html(disease):
    info, p = remedy_table.get(disease), PLAN.get(disease)
    if disease == "Healthy" or p is None or info is None:
        return ('<ol class="plan"><li><b>No spray needed</b>The leaf looks healthy.</li>'
                '<li><b>Keep watching</b>Check new leaves every week, and come back with a photo if spots or '
                'powder appear.</li></ol>')
    steps = [("Remove the infected parts", p["first"]),
             ("Spray", f"<b class='spray'>{info['pesticide']}</b>, {info['dose']}. {p['when']}"),
             ("Repeat", p["repeat"]),
             ("Prevent it next time", p["prevent"])]
    return '<ol class="plan">' + "".join(f"<li><b>{t}</b>{d}</li>" for t, d in steps) + "</ol>"


# ------------------------------------------------------------------ app ----
st.set_page_config(page_title="Mango Leaf Disease Detector", page_icon="🥭", layout="wide")
st.markdown(STYLE, unsafe_allow_html=True)

backbone, W0, b0, anchors, MODEL_ID = load_backbone()
agent = build_agent(backbone, W0, b0, anchors, MODEL_ID)
UNKNOWN = agent.UNKNOWN
gate, gate_error = load_leaf_gate()

if "observed" not in st.session_state:
    st.session_state.observed = set()

checks = ("Checks first that the photo is a mango leaf" if (gate is not None and gate.probe is not None)
          else "Checks first that the photo is a leaf" if gate is not None else "Leaf check off")
st.markdown(f"""<div class="hero">
    <h1>🥭 Mango Leaf Disease Detector &amp; Pesticide Recommender</h1>
    <p>Upload a photo of one mango leaf. You get the likely disease, how sure the model is, and the spray
       to use, and your feedback teaches the app.</p>
    <div class="facts">
      <span class="fact"><b>96.3%</b> accuracy on 750 test photos the model never saw</span>
      <span class="fact"><b>8</b> classes: 7 diseases and healthy</span>
      <span class="fact">{checks}</span>
    </div></div>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🥭 Your session")
    last = last_undo_description()
    if last:
        st.caption(f"Last feedback: {last}")
        if stretch(st.button, "↩️ Undo last feedback"):
            done = undo_last()
            st.session_state.flash = f"↩️ Undone: {done}. The agent is back to how it was before that feedback."
            st.rerun()
    else:
        st.caption("No feedback given yet.")
    if gate is not None:
        st.caption("🍃 Leaf check is on: photos that are not a leaf"
                   + (", or not a mango leaf," if gate.probe is not None else "") + " are stopped before diagnosis.")
        if gate.probe is None:
            st.warning("Mango check is off: `mango_probe.npz` was not found next to `leaf_gate.py`. "
                       "Copy it into the project folder and restart the app.")
    else:
        st.caption("🍃 Leaf check is off. Run `python -m pip install open_clip_torch` to turn it on.")

    with st.expander("🔬 RL agent details (for demo)"):
        n_fb = len(agent.fb_y)
        eps_now = max(0.05, 0.30 * 0.97 ** n_fb)
        c1, c2 = st.columns(2)
        c1.metric("Feedbacks", n_fb)
        c2.metric("Not-sure below", f"{agent.theta:.0%}")
        st.metric("Exploration ε", f"{eps_now:.0%}")
        st.markdown("**Rewards**  \n✅ right **+10**  \n❌ wrong **−20**  \n❔ 'not sure' on a known disease **−2**")
        st.caption("ε-greedy: when the top two answers are close, the agent sometimes asks about its 2nd guess. "
                   "ε falls from 30% to 5% as feedback grows. The treatment shown is always for the most likely disease.")
        if stretch(st.button, "Reset learning (demo)"):
            ok = all([safe_remove(f) for f in (STATE_FILE, MEMORY_FILE, LEGACY_MEMORY_FILE)])
            ok = clear_folder(UNDO_DIR) and ok
            ok = move_aside(FEEDBACK_DIR) and ok         # keep your collected photos, just move them aside
            if not ok:
                st.session_state.flash_info = ("Some files are locked by Windows/OneDrive and could not be removed. "
                                               "Wait a few seconds and press Reset again, or pause OneDrive sync.")
            else:
                st.session_state.flash = ("Learning reset: the agent is back to the trained model "
                                          "(old feedback photos moved to a backup folder).")
            st.session_state.observed = set()
            st.cache_resource.clear()
            st.rerun()

uploaded_file = st.file_uploader("Upload a mango leaf photo (JPG, PNG, WEBP or HEIC)", type=UPLOAD_TYPES)
if "flash_info" in st.session_state:
    st.info(st.session_state.pop("flash_info"))
if "flash" in st.session_state:
    st.success(st.session_state.pop("flash"))

if uploaded_file is None:
    st.markdown("""<div class="howto">
      <div><b>1. Take a clear photo</b>One leaf filling most of the frame, in daylight, held steady.</div>
      <div><b>2. Check the result</b>See the likely disease, how sure the model is, and the spray and dose.</div>
      <div><b>3. Tell the app if it was right</b>Your answer is a reward or a penalty for the agent,
           so it gets better on leaves like yours.</div></div>""", unsafe_allow_html=True)
else:
    raw = uploaded_file.getvalue()
    image, read_error = read_upload(raw, uploaded_file.name)
    if read_error:
        st.error("❌ " + read_error)
        st.stop()
    fatal, quality_warnings = photo_quality(image)
    if fatal:
        st.error("❌ " + fatal)
        st.stop()
    if quality_warnings:
        st.warning("📷 Photo quality: **" + ", ".join(quality_warnings) + "**. The result may be wrong - "
                   "retake the photo in good light, hold the phone steady, and fill the frame with one leaf.")
    image_hash = hashlib.md5(raw).hexdigest()
    look_hash = content_hash(image)
    memory = load_memory()
    known_as = already_labeled(memory, image_hash, look_hash)

    # ---------- 0. is it a leaf at all? ----------
    leaf_check = gate.check(image) if gate is not None else None
    forced = st.session_state.get("force_leaf_" + look_hash, False)
    if leaf_check and leaf_check["verdict"] == "not_leaf" and not forced:
        left, right = st.columns([1, 2], gap="large")
        with left:
            stretch(st.image, image, caption="Uploaded image")
        with right:
            step(1, "Result")
            st.markdown(f"""<div class="result-card lvl-no">
                <span class="level">Not a leaf</span>
                <h2>🚫 This does not look like a leaf</h2>
                <p>It looks more like <b>{leaf_check['looks_like']}</b>
                   (leaf score {leaf_check['score']:.1%}, needs at least 15%).</p>
                <p>No diagnosis or treatment is given. Upload a clear photo of one leaf.</p>
                </div>""", unsafe_allow_html=True)
            if st.button("It IS a leaf - analyse anyway"):
                st.session_state["force_leaf_" + look_hash] = True
                st.rerun()
        st.stop()

    # ---------- 0b. is it a MANGO leaf? ----------
    forced_mango = st.session_state.get("force_mango_" + look_hash, False)
    if leaf_check and leaf_check["mango"] is False and not forced and not forced_mango:
        left, right = st.columns([1, 2], gap="large")
        with left:
            stretch(st.image, image, caption="Uploaded image")
        with right:
            step(1, "Result")
            st.markdown(f"""<div class="result-card lvl-low">
                <span class="level">Probably not a mango leaf</span>
                <h2>🍃 This looks like a leaf of another plant</h2>
                <p>This app only knows mango leaf diseases, so a diagnosis for another plant would be wrong.
                   (mango score {max(leaf_check['mango_zs'], leaf_check['mango_probe']):.0%}, needs at least 50%)</p>
                <p>Please upload a photo of a mango leaf.</p></div>""", unsafe_allow_html=True)
            if st.button("It IS a mango leaf - analyse anyway"):
                st.session_state["force_mango_" + look_hash] = True
                st.rerun()
        st.stop()

    with torch.no_grad():
        feat = backbone(transform(image).unsqueeze(0)).numpy()[0]
    if look_hash not in agent.obs_ids:                   # count every new image once, and save it
        agent.observe(feat)
        agent.obs_ids.add(look_hash)
        agent.save(STATE_FILE, MODEL_ID)

    ruled_out = agent.rejected_for(feat)
    if len(ruled_out) >= len(class_names):
        # every class was marked wrong for this leaf -> the honest answer is "none of the 8"
        left, right = st.columns([1, 2], gap="large")
        with left:
            stretch(st.image, image, caption="Uploaded leaf")
        with right:
            step(1, "Result")
            st.markdown("""<div class="result-card lvl-no"><span class="level">All 8 classes ruled out</span>
                <h2>None of the known classes</h2>
                <p>You marked every one of the 8 classes as wrong for this leaf, so the app gives no diagnosis
                and no treatment. It may be a disease outside the dataset - ask an agricultural expert.</p></div>""",
                        unsafe_allow_html=True)
            if known_as == "None of these":
                st.info("Already saved as 'none of these' for this leaf.")
            elif st.button("Save as 'none of these' (teaches the 'not sure' level)", type="primary"):
                push_undo(memory, (image_hash, look_hash), "None of these", image_hash, "'None of these' for this leaf")
                out = agent.learn(feat, UNKNOWN)
                agent.save(STATE_FILE, MODEL_ID)
                remember_label(memory, image_hash, look_hash, label_name="None of these")
                save_feedback_image(image, "None of these", image_hash)
                st.session_state.flash = f"Saved as 'none of these'. 'Not sure' level is now {out['theta']:.0%}."
                st.rerun()
        st.stop()

    order, probs, confidence, not_sure, best, explored, eps = agent.choose(feat)
    top = int(order[0])
    level, level_text, heading = confidence_level(confidence, not_sure)

    # has the user already confirmed what this leaf is? -> show the final diagnosis and the cure plan
    seen_label = agent.find_seen(feat, SAME_IMAGE_SIMILARITY)
    confirmed = None
    if seen_label is not None and 0 <= seen_label < len(class_names):
        confirmed = seen_label
    elif known_as in class_names:
        confirmed = class_names.index(known_as)
    confirmed_none = seen_label == UNKNOWN or known_as == "None of these"

    if confirmed is not None or confirmed_none:
        left, right = st.columns([1, 2], gap="large")
        with left:
            stretch(st.image, image, caption="Uploaded leaf")
        with right:
            step(1, "Diagnosis")
            if confirmed is not None:
                name = class_names[confirmed]
                st.markdown(f"""<div class="result-card lvl-ok">
                    <span class="level">Confirmed by you</span>
                    <h2>{name}</h2>
                    <p>{kind_tag(name)}</p>
                    <p>Your answer is saved and the agent has learned from it. If it was a mistake, press
                       <b>Undo last feedback</b> in the sidebar.</p></div>""", unsafe_allow_html=True)
            else:
                st.markdown("""<div class="result-card lvl-no">
                    <span class="level">Confirmed by you</span>
                    <h2>Not one of the 8 known classes</h2>
                    <p>No treatment is given for an unknown problem. Show the leaf to your agriculture officer or
                       Krishi Vigyan Kendra.</p></div>""", unsafe_allow_html=True)
            with st.expander("What the model predicted before your answer"):
                st.bar_chart(pd.DataFrame({"probability": agent.probs(feat)}, index=class_names))
        if confirmed is not None:
            step(2, "Treatment plan" if class_names[confirmed] != "Healthy" else "What to do")
            st.markdown(plan_html(class_names[confirmed]), unsafe_allow_html=True)
            if class_names[confirmed] != "Healthy":
                st.markdown(SAFETY, unsafe_allow_html=True)
        st.stop()


    # ---------- 1. result ----------
    left, right = st.columns([1, 2], gap="large")
    with left:
        stretch(st.image, image, caption="Uploaded leaf")
    with right:
        step(1, "Result")
        if ruled_out:
            st.warning("Ruled out by your feedback: **" + ", ".join(class_names[i] for i in sorted(ruled_out))
                       + "**. Showing the next most likely disease (its own probability, not re-scaled).")
        if leaf_check and leaf_check["verdict"] == "unclear":
            st.warning(f"⚠️ This may not be a leaf (leaf score {leaf_check['score']:.0%}; it also looks like "
                       f"{leaf_check['looks_like']}). Check the photo before trusting the result.")
        elif forced:
            st.warning("⚠️ The leaf check said this is not a leaf; you chose to analyse it anyway.")
        elif forced_mango:
            st.warning("⚠️ The mango check said this is probably not a mango leaf; you chose to analyse it anyway.")
        result_card(class_names[top], confidence, level, level_text, heading)
        with st.expander("All 8 probabilities"):
            st.bar_chart(pd.DataFrame({"probability": probs}, index=class_names))

    # ---------- 2. treatment ----------
    if level == "high":
        step(2, "Treatment plan" if class_names[top] != "Healthy" else "What to do")
        st.caption("Based on a high-confidence result. Confirm it below to save it.")
        st.markdown(plan_html(class_names[top]), unsafe_allow_html=True)
    else:
        step(2, "Possible treatments")
        st.caption("The model is not certain, so the 3 closest matches are shown side by side. Tell the app which "
                   "one it is below, and you get the full step-by-step treatment plan.")
        top3 = [int(i) for i in order[:3] if probs[int(i)] > 0]
        for col, (rank, i) in zip(st.columns(len(top3)), enumerate(top3, 1)):
            with col:
                match_card(rank, class_names[i], probs[i], rank == 1)
    if class_names[top] != "Healthy" or level != "high":
        st.markdown(SAFETY, unsafe_allow_html=True)

    # ---------- 3. feedback ----------
    step(3, "Was this right? Your answer teaches the app")
    # a "marked wrong" answer only rules out one disease, so keep asking about the next guess;
    # stop asking only once a real label (or "none of these") has been given
    only_rejections = ((seen_label is None or seen_label < 0)
                       and (known_as is None or str(known_as).startswith("Not ")))
    if not only_rejections:
        given = known_as or "an earlier label"
        if seen_label is not None and seen_label < 0:
            given = f"{class_names[-1 - seen_label]} was marked wrong"
        elif seen_label is not None and seen_label < len(class_names):
            given = class_names[seen_label]
        elif seen_label == UNKNOWN:
            given = "none of these"
        st.info(f"Feedback was already given for this image (your label: **{given}**), so it is not asked again. "
                "It is loaded from the saved agent, so it still works after a refresh.")
    else:
        def confirm(label):
            """Store a confirmed disease for this leaf (reward +10 for that answer)."""
            push_undo(memory, (image_hash, look_hash), class_names[label], image_hash, f"confirmed {class_names[label]}")
            out = agent.learn(feat, label)
            st.session_state.flash = (f"✅ Learned: probability of {class_names[label]} on this image "
                                      f"{out['before']:.1%} → {out['after']:.1%}")
            agent.save(STATE_FILE, MODEL_ID)
            remember_label(memory, image_hash, look_hash, label_name=class_names[label])
            save_feedback_image(image, class_names[label], image_hash)
            st.rerun()

        with st.container(border=True):
            if explored:
                # the result above shows the TOP guess; the question tests the 2nd guess (epsilon-greedy)
                st.markdown(f"#### The result above is **{class_names[top]}** ({probs[top]:.1%}). "
                            f"Could it be **{class_names[best]}** ({probs[best]:.1%}) instead?")
                st.caption(f"🔍 Exploring (ε = {eps:.0%}): the top two answers are close, so the agent checks its "
                           "2nd guess too. This is how it learns which one is right for leaves like this. "
                           "The treatment above is still for the top result.")
                options = ["Not yet confirmed",
                           f"Yes, it is {class_names[best]}",
                           f"No, it is {class_names[top]} (the result above)",
                           "No, it's actually...",
                           "No, but I don't know the right disease"]
            else:
                st.markdown(f"#### Is this leaf **{class_names[best]}**?")
                options = ["Not yet confirmed",
                           f"Yes, it is {class_names[best]}",
                           "No, it's actually...",
                           "No, but I don't know the right disease"]
            choice = st.radio("Choose one:", options, label_visibility="collapsed")
            if choice.startswith("Yes"):
                st.caption("Reward +10: the agent becomes more sure of this disease for leaves like this one.")
                if st.button("Confirm", type="primary"):
                    confirm(best)
            elif choice.startswith("No, it is "):
                st.caption(f"The 2nd guess was wrong and the top result {class_names[top]} is right: the agent "
                           "learns that label for leaves like this one.")
                if st.button("Confirm", type="primary"):
                    confirm(top)
            elif choice.startswith("No, but"):
                st.caption("Reward-only feedback (pure RL): the agent only learns that this answer earned −20. "
                           "It rules this disease out for this leaf and asks about the next one.")
                if st.button("Submit (reward only)", type="primary"):
                    push_undo(memory, (image_hash, look_hash), "Not " + class_names[best], image_hash,
                              f"marked {class_names[best]} as wrong")
                    out = agent.learn_from_reward(feat, best, -20.0)
                    moved = "dropped" if out["after"] < out["before"] - 0.005 else "barely changed"
                    st.session_state.flash = (f"Reward −20 for {class_names[best]}. Its probability on this image "
                                              f"{moved}: {out['before']:.1%} → {out['after']:.1%}.")
                    agent.save(STATE_FILE, MODEL_ID)
                    remember_label(memory, image_hash, look_hash, label_name="Not " + class_names[best])
                    save_feedback_image(image, "Not " + class_names[best], image_hash)
                    st.rerun()
            elif choice.startswith("No, it's"):
                options = class_names + ["None of these / not a known mango disease"]
                picked = st.selectbox("What is it actually?", options)
                st.caption("Label feedback: the agent learns the correct disease directly (strongest update).")
                if st.button("Submit feedback", type="primary"):
                    label = options.index(picked) if picked in class_names else UNKNOWN
                    name = class_names[label] if label < len(class_names) else "None of these"
                    push_undo(memory, (image_hash, look_hash), name, image_hash, f"labelled as {name}")
                    out = agent.learn(feat, label)
                    agent.save(STATE_FILE, MODEL_ID)
                    name = class_names[label] if label < len(class_names) else "None of these"
                    remember_label(memory, image_hash, look_hash, label_name=name)
                    save_feedback_image(image, name, image_hash)
                    st.session_state.flash = (f"✅ Learned: probability of {name} on this image "
                                              f"{out['before']:.1%} → {out['after']:.1%} (steps: {out['steps']})"
                                              if label < len(class_names) else
                                              f"Saved as 'none of these'. 'Not sure' level is now {out['theta']:.0%}.")
                    st.rerun()
