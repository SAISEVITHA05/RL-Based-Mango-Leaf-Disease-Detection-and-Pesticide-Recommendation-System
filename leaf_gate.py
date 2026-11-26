"""
Leaf gate: checks "is this a leaf at all?" BEFORE the disease model runs.

Uses CLIP (OpenAI ViT-B/32, zero-shot). CLIP compares the photo with short text descriptions,
13 leaf descriptions vs 28 not-leaf descriptions (person, clothing, vehicle, screenshot, fruit, ...).
leaf score = total probability of the leaf descriptions (0..1).

    leaf score >= 0.50  -> leaf           -> run the disease model
    0.15 .. 0.50        -> unclear        -> run it, but warn "may not be a leaf"
    leaf score <  0.15  -> NOT a leaf     -> stop; user can still force "analyse anyway"

Measured (ViT-B/32, thresholds above):
  * MangoLeafBD photos (newest 40 per class, 320): 100% pass, 96.9% clearly "leaf"
  * bean-leaf field photos (133, other species):    100% pass
  * your t-shirt screenshot:                          leaf score 0.00003 -> NOT a leaf
  * random everyday photos (CIFAR-100 / Tiny-ImageNet, low resolution, ~2900):
        about 87-89% rejected; most of the rest really contain leaves or foliage

Needs:  python -m pip install open_clip_torch
The first run downloads the CLIP weights once (about 600 MB) and caches them.
If open_clip is missing the app still works, only without this check.

STAGE 2 - "is it a MANGO leaf?" (only for photos that passed as a leaf). Two signals, either one is enough:
  a) zero-shot: 7 mango-leaf descriptions vs 42 other species ("a photo of a tomato leaf", guava, neem, ...)
     -> mango score >= 0.50
  b) mango_probe.npz: a small logistic-regression trained on CLIP features of 704 MangoLeafBD training
     photos vs 1140 PlantVillage photos (14 species) + 133 bean-leaf photos -> probability >= 0.50
Measured with species the probe had NOT seen (5 random splits of the 14 PlantVillage species):
  * MangoLeafBD newest photos: 100% accepted as mango;  your 4 Google mango photos: 4/4 accepted
  * leaves of unseen species: ~92% flagged "not a mango leaf";  bean leaves: ~98% flagged
If mango_probe.npz is missing, the mango check is skipped (zero-shot alone is too weak on MangoLeafBD).
"""
import os
import numpy as np

LEAF_PROMPTS = [
    "a photo of a leaf", "a close-up photo of a plant leaf", "a photo of a diseased leaf with spots",
    "a photo of a green leaf", "a photo of a mango leaf", "a photo of a leaf held in a hand",
    "a photo of leaves on a branch", "a photo of a dry or yellow leaf",
    "a photo of a damaged leaf with holes or cut edges", "a photo of a leaf covered in white powder",
    "a photo of a leaf covered in black sooty mould", "a photo of a leaf with small bumps or galls",
    "a photo of a leaf with brown dead patches"]
NOT_LEAF_PROMPTS = {          # prompt -> short name shown to the user
    "a photo of a person": "a person", "a photo of a face": "a face", "a photo of an animal": "an animal",
    "a photo of a bird": "a bird", "a photo of an insect": "an insect", "a photo of clothing": "clothing",
    "a photo of a t-shirt": "a T-shirt", "a photo of a vehicle": "a vehicle", "a photo of food": "food",
    "a photo of a fruit": "a fruit", "a photo of a vegetable": "a vegetable", "a photo of a flower": "a flower",
    "a photo of a tree": "a whole tree", "a photo of grass": "grass", "a photo of a building": "a building",
    "a photo of a room": "a room", "a photo of furniture": "furniture", "a screenshot": "a screenshot",
    "a photo of text or a document": "text / a document", "a logo": "a logo",
    "a drawing or cartoon": "a drawing or cartoon", "a photo of an object": "an object",
    "a photo of an electronic device": "an electronic device", "a photo of the sky": "the sky",
    "a photo of a landscape": "a landscape", "a photo of the ground or soil": "the ground / soil",
    "a blurry photo of nothing": "a blurry / empty photo", "a photo of a wall": "a wall"}

LEAF_OK, LEAF_MIN = 0.50, 0.15

MANGO_PROMPTS = ["a photo of a mango leaf", "a photo of a long narrow mango tree leaf", "a photo of a diseased mango leaf",
                 "a photo of a mango leaf with black spots", "a photo of a mango leaf with white powder",
                 "a photo of a mango leaf covered in black sooty mould", "a photo of a damaged mango leaf"]
OTHER_SPECIES = ["tomato", "potato", "apple", "grape", "corn", "maize", "bean", "soybean", "cassava", "banana",
                 "coconut palm", "rose", "maple", "oak", "orange", "lemon", "guava", "neem", "tulsi", "peepal", "teak",
                 "jackfruit", "cherry", "peach", "bell pepper", "strawberry", "squash", "pumpkin", "raspberry",
                 "blueberry", "rice", "sugarcane", "cotton", "tobacco", "eucalyptus", "aloe vera", "cucumber",
                 "papaya", "betel", "money plant", "hibiscus", "curry"]
MANGO_ZS_OK, MANGO_PROBE_OK = 0.50, 0.50
PROBE_FILES = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "mango_probe.npz"),
               os.path.join(os.getcwd(), "mango_probe.npz")]          # next to this file, or the folder you run from


class LeafGate:
    def __init__(self, model_name="ViT-B-32-quickgelu", pretrained="openai"):
        import torch
        import open_clip
        self.torch = torch
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model.eval()
        tok = open_clip.get_tokenizer(model_name)
        self.prompts = LEAF_PROMPTS + list(NOT_LEAF_PROMPTS)
        with torch.no_grad():
            t = self.model.encode_text(tok(self.prompts))
        self.text = (t / t.norm(dim=-1, keepdim=True)).numpy()
        self.n_leaf = len(LEAF_PROMPTS)
        # stage 2: mango vs other species
        with torch.no_grad():
            t2 = self.model.encode_text(tok(MANGO_PROMPTS + [f"a photo of a {s} leaf" for s in OTHER_SPECIES]))
        self.species_text = (t2 / t2.norm(dim=-1, keepdim=True)).numpy()
        self.probe = None
        for path in PROBE_FILES:
            if os.path.exists(path):
                with np.load(path) as p:
                    self.probe = (np.array(p["w"], dtype=np.float32), float(p["b"]))
                break

    def check(self, image):
        """image: PIL RGB. Returns dict(verdict='leaf'|'unclear'|'not_leaf', score, looks_like)."""
        with self.torch.no_grad():
            e = self.model.encode_image(self.preprocess(image).unsqueeze(0))
        e = (e / e.norm(dim=-1, keepdim=True)).numpy()[0]
        z = 100.0 * self.text @ e
        p = np.exp(z - z.max()); p /= p.sum()
        score = float(p[:self.n_leaf].sum())
        other = self.prompts[self.n_leaf + int(p[self.n_leaf:].argmax())]
        verdict = "leaf" if score >= LEAF_OK else ("unclear" if score >= LEAF_MIN else "not_leaf")
        out = {"verdict": verdict, "score": score, "looks_like": NOT_LEAF_PROMPTS[other],
               "mango": None, "mango_zs": None, "mango_probe": None}
        if verdict != "not_leaf" and self.probe is not None:
            z2 = 100.0 * self.species_text @ e
            q = np.exp(z2 - z2.max()); q /= q.sum()
            n = len(MANGO_PROMPTS)
            zs = float(q[:n].sum())
            pr = float(1.0 / (1.0 + np.exp(-(e @ self.probe[0] + self.probe[1]))))
            out.update(mango=bool(zs >= MANGO_ZS_OK or pr >= MANGO_PROBE_OK), mango_zs=zs, mango_probe=pr)
        return out
