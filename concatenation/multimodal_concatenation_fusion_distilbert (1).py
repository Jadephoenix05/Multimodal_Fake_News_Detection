# ============================================================
# MULTIMODAL CONCATENATION BASELINE (DistilBERT + Swin)
# Paper: "Interpreting Multimodal Fake News Detection Models"
#
# This is the Section III-D / IV.J "Multimodal Concatenation
# Baseline": simple feature concatenation, defined in Eq. (4):
#
#       F_concat = [T ; V]
#
# followed by a single linear softmax classifier, Eq. (7):
#
#       y_hat = Softmax(Wc * F_concat + bc)
#
# It is the reference point TweFuse-W's adaptive weighted-sum
# fusion (Eq. 5-6) is later compared against (paper Table VI:
# Concat 0.820 vs TweFuse-W 0.838 macro-F1).
# ============================================================

import os

BASE_PATH = "/kaggle/input/datasets/charumaurya15/multimodal-fake-news-processed"
CSV_PATH = os.path.join(BASE_PATH, "master_dataset.csv")

# Trained weights from your two branch scripts
TEXT_WEIGHTS_PATH = "/kaggle/working/distilbert_fake_news_model_multiclass.pt"
IMAGE_WEIGHTS_PATH = "/kaggle/working/best_image_only_ablation.pt"

# Where to save this baseline's outputs
OUT_DIR = "/kaggle/working"

TEXT_COLUMN_CANDIDATES = [
    "text", "clean_title", "title", "headline", "content"
]

MAX_LENGTH = 80        # DistilBERT tokenization budget (matches text branch)
IMG_SIZE = 224          # Swin input size (matches image branch)
FEATURE_BATCH_SIZE = 32
CLASSIFIER_EPOCHS = 40
CLASSIFIER_LR = 1e-3


# ============================================================
# 2. IMPORTS
# ============================================================

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)
from transformers import AutoTokenizer, AutoModel, SwinModel
from tqdm.auto import tqdm


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# ============================================================
# 3. LOAD DATA + UNIFIED FILTER
# ============================================================
# Unify the two branches' filtering: a row must have BOTH a
# resolvable image path AND non-null text, otherwise text
# features and image features won't line up 1:1.

df = pd.read_csv(CSV_PATH)
print("Dataset shape:", df.shape)

TEXT_COLUMN = None
for candidate in TEXT_COLUMN_CANDIDATES:
    if candidate in df.columns:
        TEXT_COLUMN = candidate
        break
if TEXT_COLUMN is None:
    raise ValueError(f"No text column found among {TEXT_COLUMN_CANDIDATES}")
print("Using text column:", TEXT_COLUMN)


def get_actual_image_path(row):
    if pd.isna(row["image_path"]):
        return None
    filename = os.path.basename(str(row["image_path"]))
    source = str(row["source"]).lower()
    if source == "fakenewsnet":
        folder = os.path.join(BASE_PATH, "fakenewsnet", "fakenewsnet", "images")
    elif source == "reddit":
        folder = os.path.join(BASE_PATH, "reddit", "reddit", "images")
    elif source == "twitter":
        folder = os.path.join(BASE_PATH, "twitter", "twitter", "images")
    else:
        return None
    path = os.path.join(folder, filename)
    return path if os.path.exists(path) else None


df["actual_image_path"] = df.apply(get_actual_image_path, axis=1)

fusion_df = df[df["actual_image_path"].notna()].copy()
fusion_df = fusion_df[fusion_df[TEXT_COLUMN].notna()].copy()
fusion_df[TEXT_COLUMN] = fusion_df[TEXT_COLUMN].astype(str)
fusion_df = fusion_df.reset_index(drop=True)

print("Rows with usable image + text:", len(fusion_df))

NUM_CLASSES = int(fusion_df["label"].max()) + 1
print("NUM_CLASSES:", NUM_CLASSES)
print(fusion_df["label"].value_counts().sort_index())


# ============================================================
# 4. SPLIT
# ============================================================

train_df, temp_df = train_test_split(
    fusion_df, test_size=0.20, random_state=42, stratify=fusion_df["label"]
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, random_state=42, stratify=temp_df["label"]
)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print("Train/Val/Test:", len(train_df), len(val_df), len(test_df))


# ============================================================
# 5. TEXT MODEL 
# ============================================================

class DistilBertFakeNewsClassifier(nn.Module):
    def __init__(self, text_model, num_classes):
        super().__init__()
        self.distilbert = text_model
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        features = outputs.last_hidden_state[:, 0, :]  # [CLS]
        logits = self.classifier(features)
        return logits, features


tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
distilbert_backbone = AutoModel.from_pretrained("distilbert-base-uncased")
text_model = DistilBertFakeNewsClassifier(distilbert_backbone, NUM_CLASSES).to(device)

text_state = torch.load(TEXT_WEIGHTS_PATH, map_location=device)
text_model.load_state_dict(text_state)
text_model.eval()
for p in text_model.parameters():
    p.requires_grad = False

print("Loaded text model from:", TEXT_WEIGHTS_PATH)


# ============================================================
# 6. IMAGE MODEL 
# ============================================================

class ImageOnlyModel(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.backbone = SwinModel.from_pretrained(
            "microsoft/swin-base-patch4-window7-224", use_safetensors=True
        )
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.projection_head = nn.Sequential(
            nn.Linear(1024, 768),
            nn.ReLU(),
            nn.Linear(768, 768)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, pixel_values):
        with torch.no_grad():
            outputs = self.backbone(pixel_values=pixel_values)
            pooled_output = outputs.pooler_output
        projected_features = self.projection_head(pooled_output)
        logits = self.classifier(projected_features)
        return logits, projected_features


image_model = ImageOnlyModel(num_classes=NUM_CLASSES).to(device)

image_state = torch.load(IMAGE_WEIGHTS_PATH, map_location=device)
image_model.load_state_dict(image_state)
image_model.eval()
for p in image_model.parameters():
    p.requires_grad = False

print("Loaded image model from:", IMAGE_WEIGHTS_PATH)


# ============================================================
# 7. DATASETS FOR FEATURE EXTRACTION 
# ============================================================

class TextOnlyDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return row[TEXT_COLUMN], int(row["label"])


def text_collate_fn(batch):
    texts, labels = zip(*batch)
    return list(texts), torch.tensor(labels)


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

eval_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


class ImageOnlyDataset(Dataset):
    def __init__(self, dataframe, transform):
        self.data = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img = Image.open(row["actual_image_path"]).convert("RGB")
        return self.transform(img), int(row["label"])


# ============================================================
# 8. EXTRACT + CONCATENATE FEATURES FOR ONE SPLIT
# ============================================================

@torch.no_grad()
def extract_concat_features(split_df, split_name):

    text_loader = DataLoader(
        TextOnlyDataset(split_df), batch_size=FEATURE_BATCH_SIZE,
        shuffle=False, collate_fn=text_collate_fn
    )
    image_loader = DataLoader(
        ImageOnlyDataset(split_df, eval_tfms), batch_size=FEATURE_BATCH_SIZE,
        shuffle=False
    )

    text_feats, text_labels = [], []
    for texts, labels in tqdm(text_loader, desc=f"Text features [{split_name}]"):
        inputs = tokenizer(
            texts, padding="max_length", truncation=True,
            max_length=MAX_LENGTH, return_tensors="pt"
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        _, feats = text_model(input_ids, attention_mask)
        text_feats.append(feats.cpu())
        text_labels.append(labels)

    image_feats, image_labels = [], []
    for imgs, labels in tqdm(image_loader, desc=f"Image features [{split_name}]"):
        imgs = imgs.to(device)
        _, feats = image_model(imgs)
        image_feats.append(feats.cpu())
        image_labels.append(labels)

    text_feats = torch.cat(text_feats)
    text_labels = torch.cat(text_labels)
    image_feats = torch.cat(image_feats)
    image_labels = torch.cat(image_labels)

    assert torch.equal(text_labels, image_labels), (
        f"Label mismatch between text and image features on {split_name} split! "
        "The two loaders processed rows in different order."
    )

    # F_concat = [T ; V]
    concat_feats = torch.cat([text_feats, image_feats], dim=1)  
    print(f"{split_name}: text {text_feats.shape}, image {image_feats.shape}, "
          f"concat {concat_feats.shape}")

    return concat_feats, text_labels


train_concat, train_labels = extract_concat_features(train_df, "train")
val_concat, val_labels = extract_concat_features(val_df, "val")
test_concat, test_labels = extract_concat_features(test_df, "test")


# ============================================================
# 9. SAVE CONCATENATED FEATURES
# ============================================================
# Reusable by the later TweFuse-W weighted-sum fusion script so
# text/image features don't need to be re-extracted.

torch.save({"features": train_concat, "labels": train_labels},
           os.path.join(OUT_DIR, "train_concat_features_distilbert.pt"))
torch.save({"features": val_concat, "labels": val_labels},
           os.path.join(OUT_DIR, "val_concat_features_distilbert.pt"))
torch.save({"features": test_concat, "labels": test_labels},
           os.path.join(OUT_DIR, "test_concat_features_distilbert.pt"))

print("Saved concatenated features to", OUT_DIR)


# ============================================================
# 10. CONCATENATION CLASSIFIER (Eq. 7: linear softmax head)
# ============================================================

class ConcatClassifier(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.classifier = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.classifier(x)


concat_model = ConcatClassifier(train_concat.shape[1], NUM_CLASSES).to(device)
optimizer = torch.optim.AdamW(concat_model.parameters(), lr=CLASSIFIER_LR, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

train_concat_dev = train_concat.to(device)
train_labels_dev = train_labels.to(device)
val_concat_dev = val_concat.to(device)
val_labels_dev = val_labels.to(device)
test_concat_dev = test_concat.to(device)
test_labels_dev = test_labels.to(device)


def macro_metrics(labels, preds):
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    return acc, p, r, f1


best_val_f1 = -1.0
best_state = None

for epoch in range(CLASSIFIER_EPOCHS):

    concat_model.train()
    optimizer.zero_grad()
    logits = concat_model(train_concat_dev)
    loss = criterion(logits, train_labels_dev)
    loss.backward()
    optimizer.step()

    concat_model.eval()
    with torch.no_grad():
        val_logits = concat_model(val_concat_dev)
        val_preds = val_logits.argmax(1).cpu().numpy()
        val_acc, val_p, val_r, val_f1 = macro_metrics(val_labels.numpy(), val_preds)

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_state = {k: v.clone() for k, v in concat_model.state_dict().items()}

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{CLASSIFIER_EPOCHS}: "
              f"train_loss={loss.item():.4f} val_acc={val_acc:.4f} val_macroF1={val_f1:.4f}")

concat_model.load_state_dict(best_state)
print(f"\nBest validation macro-F1: {best_val_f1:.4f}")


# ============================================================
# 11. FINAL TEST EVALUATION (Table VI style)
# ============================================================

concat_model.eval()
with torch.no_grad():
    test_logits = concat_model(test_concat_dev)
    test_preds = test_logits.argmax(1).cpu().numpy()

test_acc, test_p, test_r, test_f1 = macro_metrics(test_labels.numpy(), test_preds)

print("\n" + "=" * 60)
print("CONCATENATION BASELINE -- FINAL TEST RESULTS")
print("=" * 60)
print(f"Accuracy        : {test_acc:.4f}")
print(f"Macro Precision : {test_p:.4f}")
print(f"Macro Recall    : {test_r:.4f}")
print(f"Macro F1        : {test_f1:.4f}")

print("\nClassification report:")
print(classification_report(test_labels.numpy(), test_preds, zero_division=0))

print("Confusion matrix:")
print(confusion_matrix(test_labels.numpy(), test_preds))


# ============================================================
# 12. SAVE CLASSIFIER
# ============================================================

torch.save(concat_model.state_dict(),
           os.path.join(OUT_DIR, "concat_classifier_distilbert.pt"))
print("\nSaved concatenation classifier to",
      os.path.join(OUT_DIR, "concat_classifier_distilbert.pt"))

print("\n========== SAVED FILES ==========")
print(os.path.join(OUT_DIR, "train_concat_features_distilbert.pt"))
print(os.path.join(OUT_DIR, "val_concat_features_distilbert.pt"))
print(os.path.join(OUT_DIR, "test_concat_features_distilbert.pt"))
print(os.path.join(OUT_DIR, "concat_classifier_distilbert.pt"))

