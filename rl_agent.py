"""
RL decision layer for the mango leaf project (numpy only, no torch needed).

Your trained ResNet18 stays FROZEN and only supplies a 512-d feature vector.
The agent starts as an exact copy of your trained classifier and then learns from
user feedback using this reward:
    right diagnosis +10 | wrong diagnosis -20 | "not sure" on a known disease -2
    "not sure" on a leaf that is none of the known diseases +10

What it learns
  1. Which disease to diagnose: every correction moves the disease scores towards the right
     answer (step size = reward that was missed). Older feedback is replayed so it does not forget.
  2. When to say "not sure": it says so only when its best guess has confidence below a threshold
     theta (starts at 0.40, like your original app). theta is re-chosen from your feedback so that
     it earns the most reward, and it is kept between 0.20 and 0.70 so it can never swallow everything.
The agent ALWAYS ranks all diseases, so the closest matches can be shown even when it is not sure.
"""
import os
import numpy as np


def _softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


class RLAgent:
    def __init__(self, W0, b0, class_names, anchors=None,
                 r_correct=10.0, r_wrong=-20.0, r_abstain=-2.0,
                 lr=1.0, l2=0.1, replay_k=24, target=0.8, max_steps=80,
                 theta0=0.40, theta_min=0.20, theta_max=0.70, seed=0):
        self.class_names = list(class_names)
        self.n = len(self.class_names)
        self.UNKNOWN = self.n                      # label meaning "none of the known diseases"
        self.r_correct, self.r_wrong, self.r_abstain = r_correct, r_wrong, r_abstain
        self.r_scale = max(abs(r_correct), abs(r_wrong), abs(r_abstain))
        self.lr, self.l2 = lr, l2
        self.replay_k, self.target, self.max_steps = replay_k, target, max_steps
        self.theta0, self.theta_min, self.theta_max = theta0, theta_min, theta_max
        self.theta = theta0
        self.rng = np.random.default_rng(seed)

        d = W0.shape[1]
        self.W, self.b = W0.astype(float).copy(), b0.astype(float).copy()    # = your classifier
        self.W_prior = self.W.copy()
        # running mean of the features seen so far (ResNet features share a big common part;
        # updates use the deviation from it so one correction does not shift every image)
        self.mu, self.n_mu = np.zeros(d), 0
        self.anchor_x = np.zeros((0, d)); self.anchor_y = np.zeros(0, dtype=int)
        if anchors is not None:
            self.anchor_x = np.asarray(anchors[0], dtype=float)
            self.anchor_y = np.asarray(anchors[1], dtype=int)
            self.mu, self.n_mu = self.anchor_x.mean(axis=0), len(self.anchor_x)
        self.fb_x = np.zeros((0, d)); self.fb_y = np.zeros(0, dtype=int)
        self.baseline = 0.0      # running average reward (used by the reward-only update)
        self.obs_ids = set()     # ids of images already counted in the feature average

    # ---------------- acting ----------------
    def observe(self, f):
        """Call once for every new image the app sees (keeps the feature mean up to date)."""
        self.n_mu += 1
        self.mu += (np.asarray(f, dtype=float) - self.mu) / self.n_mu

    def probs(self, f):
        return _softmax(self.W @ f + self.b)               # over the 8 diseases

    def rejected_for(self, f, thresh=0.98):
        """Diseases the user already marked WRONG for this image (or a near copy)."""
        if len(self.fb_y) == 0:
            return set()
        X = self.fb_x
        cos = (X @ f) / (np.linalg.norm(X, axis=1) * np.linalg.norm(f) + 1e-9)
        return {int(-1 - y) for c, y in zip(cos, self.fb_y) if c >= thresh and y < 0}

    def decide(self, f):
        """Returns (ranked disease indices, probabilities, confidence of best guess, not_sure flag).
        Diseases already rejected for this image get probability 0 (the others keep their own value),
        so the agent moves on to its next most likely disease without faking confidence."""
        p = self.probs(f)
        rej = self.rejected_for(f)
        if rej and len(rej) < self.n:
            # ruled-out diseases get 0, the others KEEP their own probability (no re-scaling):
            # re-scaling would make the last remaining disease look 100% sure "by elimination"
            p = p.copy(); p[list(rej)] = 0.0
        order = np.argsort(-p)
        conf = float(p[order[0]])
        return order, p, conf, conf < self.theta

    def choose(self, f, eps0=0.30, eps_min=0.05, decay=0.97, explore_below=0.70, min_p2=0.15):
        """Safe epsilon-greedy exploration (contextual bandit).
        Only explores when the agent is UNSURE (best confidence < explore_below) and its 2nd guess
        is plausible (>= min_p2). Then, with probability epsilon, it tests the 2nd guess instead.
        epsilon decays as feedback grows: 0.30 -> 0.05. Seeded by the image, so the same image
        always gets the same choice (stable across Streamlit reruns).
        Returns (order, probs, confidence, not_sure, action, explored, epsilon)."""
        order, p, conf, not_sure = self.decide(f)
        eps = max(eps_min, eps0 * decay ** len(self.fb_y))
        seed = int.from_bytes(np.round(np.asarray(f, dtype=np.float32), 3).tobytes()[:4096:7], "little") % (2**32)
        u = np.random.default_rng(seed + len(self.fb_y)).random()
        explored = (conf < explore_below and p[order[1]] >= min_p2 and u < eps)
        action = int(order[1]) if explored else int(order[0])
        return order, p, conf, not_sure, action, explored, eps

    def act(self, f):
        order, p, conf, not_sure = self.decide(f)
        return (self.UNKNOWN if not_sure else int(order[0])), p

    def find_seen(self, f, thresh=0.98):
        """Has the user already given feedback on (a copy of) this image? Returns its label or None."""
        if len(self.fb_y) == 0:
            return None
        X = self.fb_x
        cos = (X @ f) / (np.linalg.norm(X, axis=1) * np.linalg.norm(f) + 1e-9)
        same = np.where(cos >= thresh)[0]
        if len(same) == 0:
            return None
        real = [i for i in same if self.fb_y[i] >= 0]      # a real label beats a "this guess was wrong"
        return int(self.fb_y[real[-1]] if real else self.fb_y[same[-1]])

    # ---------------- learning ----------------
    def _mu_eff(self):
        return self.mu * (self.n_mu / (self.n_mu + 5.0))    # shrink towards 0 while few images were seen

    def _step(self, f, label):
        p = self.probs(f)
        regret = (1.0 - p[label]) * (self.r_correct - self.r_wrong) / self.r_scale   # reward missed
        best = np.zeros(self.n); best[label] = 1.0
        self._apply((best - p) * regret, f)

    def _apply(self, g, f):
        fc = f - self._mu_eff()
        denom = max(float(fc @ f), 1.0)                      # makes the logit change for this image = lr * g
        eta = self.lr / denom
        self.W += eta * np.outer(g, fc) - eta * self.l2 * (self.W - self.W_prior)

    def _replay_batch(self):
        k = self.replay_k
        fb_ok = np.where((self.fb_y >= 0) & (self.fb_y < self.n))[0]
        n_fb, n_an = len(fb_ok), len(self.anchor_y)
        k_fb = min(n_fb, k // 2) if n_an else min(n_fb, k)
        k_an = min(n_an, k - k_fb)
        xs, ys = [], []
        if k_fb:
            i = self.rng.choice(fb_ok, k_fb, replace=False); xs.append(self.fb_x[i]); ys.append(self.fb_y[i])
        if k_an:
            i = self.rng.choice(n_an, k_an, replace=False); xs.append(self.anchor_x[i]); ys.append(self.anchor_y[i])
        if not xs:
            return np.zeros((0, self.W.shape[1])), np.zeros(0, dtype=int)
        return np.vstack(xs), np.concatenate(ys)

    def _tune_theta(self):
        """Pick the 'not sure' threshold that earns the most reward on all feedback so far."""
        valid = self.fb_y >= 0                               # rejected-action feedback has no known label
        if valid.sum() < 4:
            self.theta = self.theta0
            return
        fy = self.fb_y[valid]
        P = np.array([self.probs(x) for x in self.fb_x[valid]])
        conf, top = P.max(axis=1), P.argmax(axis=1)
        known = fy < self.n
        predict_r = np.where(known & (top == fy), self.r_correct, self.r_wrong)
        abstain_r = np.where(known, self.r_abstain, self.r_correct)
        best_t, best_v = self.theta0, -1e9
        for t in np.arange(self.theta_min, self.theta_max + 1e-9, 0.05):
            v = np.where(conf >= t, predict_r, abstain_r).mean() - 0.5 * abs(t - self.theta0)
            if v > best_v:
                best_t, best_v = float(t), v
        self.theta = best_t

    def learn(self, f, label):
        """One user feedback. label = disease index, or self.UNKNOWN for 'none of these'."""
        f = np.asarray(f, dtype=float)
        self.fb_x = np.vstack([self.fb_x, f[None]])
        self.fb_y = np.append(self.fb_y, label)
        steps = 0
        before = float(self.probs(f)[label]) if label < self.n else float(self.probs(f).max())
        if label < self.n:                                   # a disease label teaches the disease scores
            while steps < self.max_steps:
                steps += 1
                self._step(f, label)
                bx, by = self._replay_batch()                # rehearse old examples -> no forgetting
                for xi, yi in zip(bx, by):
                    self._step(xi, int(yi))
                p = self.probs(f)
                if p.argmax() == label and p[label] >= self.target:
                    break
        self._tune_theta()                                   # every feedback also teaches "when to be unsure"
        after = float(self.probs(f)[label]) if label < self.n else float(self.probs(f).max())
        return {"before": before, "after": after, "steps": steps, "theta": self.theta}

    def learn_from_reward(self, f, action, reward, max_steps=60, boost=2.0):
        """Pure RL feedback: the user only says how good the action we TOOK was
        (+10 it was right, -20 it was wrong). No label is given. REINFORCE with a running baseline."""
        f = np.asarray(f, dtype=float)
        action = int(action)
        self.fb_x = np.vstack([self.fb_x, f[None]])
        self.fb_y = np.append(self.fb_y, action if reward > 0 else -1 - action)   # remember: confirmed / rejected action
        r = float(reward) / self.r_scale
        adv = r - self.baseline
        onehot = np.zeros(self.n); onehot[action] = 1.0
        before, steps = float(self.probs(f)[action]), 0
        while steps < max_steps:
            steps += 1
            p = self.probs(f)
            self._apply(boost * adv * (onehot - p), f)       # gradient of  adv * log pi(action)
            bx, by = self._replay_batch()                    # rehearse earlier feedback -> no forgetting
            for xi, yi in zip(bx, by):
                self._step(xi, int(yi))
            p = self.probs(f)
            if reward < 0 and p.argmax() != action and p[action] < 0.3:
                break                                        # it no longer picks the rejected action
            if reward > 0 and p[action] >= self.target:
                break
        self.baseline = 0.9 * self.baseline + 0.1 * r
        return {"before": before, "after": float(self.probs(f)[action]), "steps": steps, "theta": self.theta}

    def evaluate(self, X, y):
        pred = np.array([self.act(x)[0] for x in X])
        y = np.asarray(y)
        return {"accuracy": float((pred == y).mean()), "abstain_rate": float((pred == self.UNKNOWN).mean())}

    # ---------------- persistence ----------------
    def save(self, path, model_id=""):
        folder = os.path.dirname(os.path.abspath(path)) or "."
        os.makedirs(folder, exist_ok=True)
        tmp = path + ".tmp.npz"
        np.savez(tmp, W=self.W, b=self.b, mu=self.mu, n_mu=self.n_mu, theta=self.theta,
                 fb_x=self.fb_x, fb_y=self.fb_y, baseline=self.baseline,
                 obs_ids=np.array(sorted(self.obs_ids), dtype="U40"), version=2,
                 model_id=np.array(str(model_id)))
        os.replace(tmp, path)

    def load(self, path, model_id=""):
        """Returns False (= start fresh) if the file is missing, broken, or was learned on top of a
        DIFFERENT model file (e.g. after retraining), so old weights never mix with a new model."""
        if not os.path.exists(path):
            return False
        try:
            with np.load(path, allow_pickle=False) as s:
                if "version" not in s.files:
                    return False
                if s["W"].shape != self.W.shape:
                    return False
                if model_id and ("model_id" not in s.files or str(s["model_id"]) != str(model_id)):
                    return False
                self.W = np.array(s["W"], dtype=float, copy=True)
                if "b" in s.files and s["b"].shape == self.b.shape:
                    self.b = np.array(s["b"], dtype=float, copy=True)
                self.mu = np.array(s["mu"], dtype=float, copy=True)
                self.n_mu = int(s["n_mu"])
                self.theta = float(s["theta"])
                self.fb_x = np.array(s["fb_x"], dtype=float, copy=True)
                self.fb_y = np.array(s["fb_y"], dtype=int, copy=True)
                if "baseline" in s.files:
                    self.baseline = float(s["baseline"])
                if "obs_ids" in s.files:
                    self.obs_ids = set(s["obs_ids"].tolist())
            return True
        except (OSError, ValueError, KeyError, EOFError):
            return False