import os, copy
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# 1. Load Features (Auto-find)
def find_file(filename):
    for root, _, files in os.walk("/kaggle/input"):
        if filename in files:
            return os.path.join(root, filename)
    return None

txt_dir = os.path.dirname(find_file("train_text_features.pt"))
img_dir = os.path.dirname(find_file("train_image_raw_features.pt"))

# Use the fine-tuned text features
txt_tr = torch.load(os.path.join(txt_dir, "train_text_features.pt"))
txt_va = torch.load(os.path.join(txt_dir, "val_text_features.pt"))
txt_te = torch.load(os.path.join(txt_dir, "test_text_features.pt"))

img_tr = torch.load(os.path.join(img_dir, "train_image_raw_features.pt"))
img_va = torch.load(os.path.join(img_dir, "val_image_raw_features.pt"))
img_te = torch.load(os.path.join(img_dir, "test_image_raw_features.pt"))

class FusionDataset(Dataset):
    def __init__(self, t, i, y):
        self.t, self.i, self.y = t, i, y
    def __len__(self): return len(self.y)
    def __getitem__(self, k): return self.t[k], self.i[k], self.y[k]

train_loader = DataLoader(FusionDataset(txt_tr["features"], img_tr["features"], img_tr["labels"]), batch_size=32, shuffle=True)
val_loader   = DataLoader(FusionDataset(txt_va["features"], img_va["features"], img_va["labels"]), batch_size=32, shuffle=False)
test_loader  = DataLoader(FusionDataset(txt_te["features"], img_te["features"], img_te["labels"]), batch_size=32, shuffle=False)

# 2. Model with LayerNorm (Prevents scale bias)
class TwoFuseW(nn.Module):
    def __init__(self, text_dim=768, image_dim=1024, shared=768, n_classes=6):
        super().__init__()
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, shared), nn.ReLU(), nn.Linear(shared, shared), nn.LayerNorm(shared)
        )
        self.img_proj = nn.Sequential(
            nn.Linear(image_dim, shared), nn.ReLU(), nn.Linear(shared, shared), nn.LayerNorm(shared)
        )
        self.gate = nn.Sequential(
            nn.Linear(2 * shared, 128), nn.ReLU(), nn.Linear(128, 2)
        )
        nn.init.zeros_(self.gate[2].weight)
        nn.init.zeros_(self.gate[2].bias)
        self.classifier = nn.Linear(shared, n_classes)

    def forward(self, text_feats, img_feats):
        T = self.text_proj(text_feats)
        V = self.img_proj(img_feats)
        a = torch.softmax(self.gate(torch.cat([T, V], dim=1)), dim=1)
        aT, aV = a[:, 0:1], a[:, 1:2]
        F = aT * T + aV * V
        return self.classifier(F), aT, aV

model = TwoFuseW().to(device)

# 3. Training Loop with Balance Regularization
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()
EPOCHS = 20
total_steps = len(train_loader) * EPOCHS
warmup_steps = int(total_steps * 0.1)
scheduler = LambdaLR(optimizer, lambda s: s/warmup_steps if s < warmup_steps else max(0.0, (total_steps-s)/(total_steps-warmup_steps)))

@torch.no_grad()
def evaluate(loader, ret_weights=False):
    model.eval()
    preds, labels, aTs = [], [], []
    for t, i, y in loader:
        logits, aT, _ = model(t.to(device), i.to(device))
        preds.extend(logits.argmax(1).cpu().numpy())
        labels.extend(y.numpy())
        if ret_weights: aTs.extend(aT.squeeze(1).cpu().numpy())
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    out = {"acc": accuracy_score(labels, preds), "f1": f1, "labels": np.array(labels), "preds": np.array(preds)}
    if ret_weights: out["alpha_T"] = np.array(aTs)
    return out

best_f1, best_state, patience = -1, None, 0
for ep in range(EPOCHS):
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {ep+1}/{EPOCHS}")
    for t, i, y in pbar:
        t, i, y = t.to(device), i.to(device), y.to(device)
        optimizer.zero_grad()
        logits, aT, aV = model(t, i)
        
        # ADDING BALANCE REGULARIZATION (Forces aT towards 0.5)
        balance_penalty = 0.5 * ((aT - 0.5)**2).mean() 
        loss = criterion(logits, y) + balance_penalty
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        
    val = evaluate(val_loader)
    print(f"Epoch {ep+1}: val_acc={val['acc']:.4f} val_macroF1={val['f1']:.4f}")
    
    if val["f1"] > best_f1:
        best_f1 = val["f1"]
        best_state = copy.deepcopy(model.state_dict())
        patience = 0
    else:
        patience += 1
        if patience >= 4:
            print("Early stopping triggered.")
            break

model.load_state_dict(best_state)
test = evaluate(test_loader, ret_weights=True)

print("\n" + "="*60)
print(f"FINAL TEST MACRO F1: {test['f1']:.4f}")
print("="*60)

LABEL_NAMES = {0: "True", 1: "Satire", 2: "Misleading", 3: "Imposter", 4: "False Connection", 5: "Manipulated"}
print("\n=== Table IV: Learned α_T / α_V per class ===")
for c in range(6):
    m = test["labels"] == c
    if m.sum() == 0: continue
    print(f"{LABEL_NAMES[c]:18s} α_T={test['alpha_T'][m].mean():.3f}  α_V={1-test['alpha_T'][m].mean():.3f}")

# Plot
classes = list(range(6))
mT = [test["alpha_T"][test["labels"] == c].mean() for c in classes]
mV = [1 - x for x in mT]
plt.figure(figsize=(10, 4))
x = np.arange(6); w = 0.35
plt.bar(x-w/2, mT, w, label="Text α_T", color="#2b9fd8")
plt.bar(x+w/2, mV, w, label="Image α_V", color="#f2c200")
plt.xticks(x, [LABEL_NAMES[i] for i in classes], rotation=20)
plt.ylim(0, 1); plt.title("Fig. 4 — Modality Weights"); plt.legend(); plt.show()

# ============================================================
# TweFuse-W + switchable increments (ablation runner)
# Built on your twefuse.py. Works on the cached features:
#   train/val/test_text_features.pt        (fine-tuned BERTweet, 768-d)
#   train/val/test_image_raw_features.pt   (Swin pooled, 1024-d)
#
# Every increment is a config flag, so each one becomes an
# ablation row. All runs use the same split, same seeds.
# ============================================================

# import os, copy, random
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.nn.functional as Fn
# import matplotlib.pyplot as plt
# from torch.utils.data import Dataset, DataLoader
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# from torch.optim.lr_scheduler import LambdaLR
# from tqdm.auto import tqdm


# Add this to the imports at the top if you don't have it:
# import seaborn as sns
# from sklearn.metrics import confusion_matrix

# Compute confusion matrix
cm = confusion_matrix(test["labels"], test["preds"])

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=[LABEL_NAMES[i] for i in range(6)],
            yticklabels=[LABEL_NAMES[i] for i in range(6)])
plt.title("TwoFuse-W — Confusion Matrix (Fig. 5)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

SEEDS = [42, 43, 44]      # mean +- std over seeds (paper notes single-seed as a limitation)
EPOCHS = 20
BATCH_SIZE = 32
PATIENCE = 4
N_CLASSES = 6
LABEL_NAMES = {0: "True", 1: "Satire", 2: "Misleading", 3: "Imposter",
               4: "False Connection", 5: "Manipulated"}


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------
# 1. Load cached features
# ------------------------------------------------------------
def find_file(filename):
    for root, _, files in os.walk("/kaggle/input"):
        if filename in files:
            return os.path.join(root, filename)
    raise FileNotFoundError(f"{filename} not found under /kaggle/input")


txt_dir = os.path.dirname(find_file("train_text_features.pt"))
img_dir = os.path.dirname(find_file("train_image_raw_features.pt"))

txt = {s: torch.load(os.path.join(txt_dir, f"{s}_text_features.pt")) for s in ("train", "val", "test")}
img = {s: torch.load(os.path.join(img_dir, f"{s}_image_raw_features.pt")) for s in ("train", "val", "test")}

# Alignment check: text and image features must describe the same rows
for s in ("train", "val", "test"):
    assert torch.equal(txt[s]["labels"], img[s]["labels"]), \
        f"Text/image labels differ on '{s}' split -> features are not row-aligned!"
print("Label alignment check passed.")


class FusionDataset(Dataset):
    def __init__(self, t, i, y):
        self.t, self.i, self.y = t, i, y
    def __len__(self): return len(self.y)
    def __getitem__(self, k): return self.t[k], self.i[k], self.y[k]


def make_loader(split, shuffle, seed=0):
    g = torch.Generator(); g.manual_seed(seed)
    return DataLoader(
        FusionDataset(txt[split]["features"], img[split]["features"], img[split]["labels"]),
        batch_size=BATCH_SIZE, shuffle=shuffle, generator=g if shuffle else None
    )


# Inverse-frequency class weights from the TRAIN split only
_counts = torch.bincount(img["train"]["labels"].long(), minlength=N_CLASSES).float()
CLASS_WEIGHTS = (_counts.sum() / (N_CLASSES * _counts.clamp(min=1)))
print("Train class counts:", _counts.int().tolist())


# ------------------------------------------------------------
# 2. Model (your TweFuse-W + optional increments)
# ------------------------------------------------------------
class TweFuseW(nn.Module):
    def __init__(self, cfg, text_dim=768, image_dim=1024, shared=768, n_classes=N_CLASSES):
        super().__init__()
        self.cfg = cfg
        act = nn.GELU if cfg["act"] == "gelu" else nn.ReLU

        def proj(d):
            return nn.Sequential(
                nn.Linear(d, shared), act(), nn.Dropout(cfg["proj_dropout"]),
                nn.Linear(shared, shared), nn.LayerNorm(shared)
            )

        self.text_proj = proj(text_dim)
        self.img_proj = proj(image_dim)

        gate_in = 2 * shared + (1 if cfg["use_cos"] else 0)
        self.gate = nn.Sequential(
            nn.Linear(gate_in, 128), act(), nn.Dropout(cfg["gate_dropout"]), nn.Linear(128, 2)
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.classifier = nn.Linear(shared, n_classes)

    def forward(self, text_feats, img_feats):
        T = self.text_proj(text_feats)
        V = self.img_proj(img_feats)

        # Increment: modality dropout (training only; never drops both)
        p = self.cfg["modality_dropout"]
        if self.training and p > 0:
            r = torch.rand(T.size(0), 1, device=T.device)
            T = T * (r >= p / 2).float()                       # drop text w.p. p/2
            V = V * ((r < p / 2) | (r >= p)).float()           # drop image w.p. p/2

        gate_in = torch.cat([T, V], dim=1)
        # Increment: explicit text-image agreement signal
        if self.cfg["use_cos"]:
            cos = Fn.cosine_similarity(T, V, dim=1).unsqueeze(1)
            gate_in = torch.cat([gate_in, cos], dim=1)

        a = torch.softmax(self.gate(gate_in), dim=1)
        aT, aV = a[:, 0:1], a[:, 1:2]
        fused = aT * T + aV * V
        return self.classifier(fused), aT, aV


# ------------------------------------------------------------
# 3. Optimizer / loss factories
# ------------------------------------------------------------
def make_optimizer(model, cfg):
    params = model.parameters()
    name, lr = cfg["optimizer"], cfg["lr"]
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "sgd_momentum":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr)
    raise ValueError(name)


def make_criterion(cfg):
    w = CLASS_WEIGHTS.to(device) if cfg["class_weighted"] else None
    return nn.CrossEntropyLoss(weight=w, label_smoothing=cfg["label_smoothing"])


# ------------------------------------------------------------
# 4. Evaluation
# ------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, ret_weights=False):
    model.eval()
    preds, labels, aTs = [], [], []
    for t, i, y in loader:
        logits, aT, _ = model(t.to(device), i.to(device))
        preds.extend(logits.argmax(1).cpu().numpy())
        labels.extend(y.numpy())
        if ret_weights:
            aTs.extend(aT.squeeze(1).cpu().numpy())
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    out = {"acc": accuracy_score(labels, preds), "prec": p, "rec": r, "f1": f1,
           "labels": np.array(labels), "preds": np.array(preds)}
    if ret_weights:
        out["alpha_T"] = np.array(aTs)
    return out


# ------------------------------------------------------------
# 5. One full training run for a config + seed
# ------------------------------------------------------------
def run_one(cfg, seed, verbose=False):
    seed_everything(seed)
    train_loader = make_loader("train", True, seed)
    val_loader = make_loader("val", False)
    test_loader = make_loader("test", False)

    model = TweFuseW(cfg).to(device)
    optimizer = make_optimizer(model, cfg)
    criterion = make_criterion(cfg)

    total_steps = len(train_loader) * EPOCHS
    warmup = int(total_steps * 0.1)
    scheduler = LambdaLR(
        optimizer,
        lambda s: s / max(1, warmup) if s < warmup
        else max(0.0, (total_steps - s) / max(1, total_steps - warmup))
    )

    best_f1, best_state, bad = -1.0, None, 0
    for ep in range(EPOCHS):
        model.train()
        it = tqdm(train_loader, desc=f"{cfg['name']} s{seed} ep{ep+1}", leave=False) if verbose else train_loader
        for t, i, y in it:
            t, i, y = t.to(device), i.to(device), y.to(device)
            optimizer.zero_grad()
            logits, aT, _ = model(t, i)
            loss = criterion(logits, y)
            if cfg["balance_lambda"] > 0:
                loss = loss + cfg["balance_lambda"] * ((aT - 0.5) ** 2).mean()
            loss.backward()
            optimizer.step()
            scheduler.step()

        val = evaluate(model, val_loader)
        if val["f1"] > best_f1:
            best_f1, best_state, bad = val["f1"], copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    model.load_state_dict(best_state)
    test = evaluate(model, test_loader, ret_weights=True)
    test["val_f1"] = best_f1
    return test


# ------------------------------------------------------------
# 6. Experiment grid (each row = one increment vs. baseline)
# ------------------------------------------------------------
BASE = dict(
    name="baseline (your twefuse.py)",
    act="relu", use_cos=False,
    proj_dropout=0.0, gate_dropout=0.0, modality_dropout=0.0,
    class_weighted=False, label_smoothing=0.0,
    balance_lambda=0.5,                 # your current balance penalty
    optimizer="adamw", lr=3e-5,
)


def variant(name, **kw):
    cfg = dict(BASE); cfg.update(kw); cfg["name"] = name
    return cfg


EXPERIMENTS = [
    BASE,
    variant("no balance penalty (lambda=0)", balance_lambda=0.0),
    variant("+ cos(T,V) in gate", use_cos=True),
    variant("+ GELU", act="gelu"),
    variant("+ dropout (proj 0.1, gate 0.1)", proj_dropout=0.1, gate_dropout=0.1),
    variant("+ modality dropout (p=0.2)", modality_dropout=0.2),
    variant("+ class-weighted CE", class_weighted=True),
    variant("+ label smoothing 0.1", label_smoothing=0.1),
    variant("optimizer: Adam", optimizer="adam"),
    variant("optimizer: SGD+momentum (lr 1e-2)", optimizer="sgd_momentum", lr=1e-2),
    variant("optimizer: RMSProp (lr 1e-4)", optimizer="rmsprop", lr=1e-4),
    variant("combo: cos + dropout + modality drop + weighted",
            use_cos=True, proj_dropout=0.1, gate_dropout=0.1,
            modality_dropout=0.2, class_weighted=True),
]


# ------------------------------------------------------------
# 7. Run everything, build the ablation table
# ------------------------------------------------------------
rows, all_results = [], {}
for cfg in EXPERIMENTS:
    res = [run_one(cfg, s) for s in SEEDS]
    all_results[cfg["name"]] = res
    f1s = np.array([r["f1"] for r in res])
    accs = np.array([r["acc"] for r in res])
    precs = np.array([r["prec"] for r in res])
    recs = np.array([r["rec"] for r in res])
    rows.append({
        "Configuration": cfg["name"],
        "Macro-F1": f"{f1s.mean():.4f} +- {f1s.std():.4f}",
        "Acc": f"{accs.mean():.4f}",
        "Macro-P": f"{precs.mean():.4f}",
        "Macro-R": f"{recs.mean():.4f}",
        "_f1_mean": f1s.mean(),
    })
    print(f"{cfg['name']:55s} macro-F1 = {f1s.mean():.4f} +- {f1s.std():.4f}")

table = pd.DataFrame(rows)
base_f1 = table.loc[0, "_f1_mean"]
table["dF1 vs baseline"] = (table["_f1_mean"] - base_f1).map(lambda d: f"{d:+.4f}")
table = table.drop(columns="_f1_mean")

print("\n" + "=" * 100)
print("ABLATION TABLE (test split, mean +- std over seeds %s)" % SEEDS)
print("=" * 100)
print(table.to_string(index=False))
table.to_csv("/kaggle/working/twefuse_ablation.csv", index=False)


# ------------------------------------------------------------
# 8. Table IV / Fig. 4 style modality weights (first seed)
# ------------------------------------------------------------
def report_weights(result, title):
    print(f"\n=== Learned alpha_T / alpha_V per class: {title} ===")
    mT = []
    for c in range(N_CLASSES):
        m = result["labels"] == c
        if m.sum() == 0:
            mT.append(np.nan); continue
        a = result["alpha_T"][m]
        mT.append(a.mean())
        print(f"{LABEL_NAMES[c]:18s} alpha_T={a.mean():.3f} +- {a.std():.3f}   "
              f"alpha_V={1 - a.mean():.3f}")
    mV = [1 - x for x in mT]
    x = np.arange(N_CLASSES); w = 0.35
    plt.figure(figsize=(10, 4))
    plt.bar(x - w / 2, mT, w, label="Text alpha_T", color="#2b9fd8")
    plt.bar(x + w / 2, mV, w, label="Image alpha_V", color="#f2c200")
    plt.xticks(x, [LABEL_NAMES[i] for i in range(N_CLASSES)], rotation=20)
    plt.ylim(0, 1); plt.title(f"Modality weights - {title}"); plt.legend(); plt.tight_layout()
    plt.show()


report_weights(all_results[BASE["name"]][0], "baseline (balance penalty on)")
report_weights(all_results["no balance penalty (lambda=0)"][0],
               "no balance penalty (the version to use for interpretability claims)")
