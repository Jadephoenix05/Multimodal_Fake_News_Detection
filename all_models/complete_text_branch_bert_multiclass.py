# ============================================================
# COMPLETE TEXT BRANCH - BERT (MULTICLASS)
# Multimodal Fake News Detection Project
# Number of classes is auto-detected from your label column
# (see step 5b) -- don't assume binary Fake/Real.
#
# This mirrors complete_image_branch_swin.py so that the
# train/val/test splits line up row-for-row with the image
# branch, which is required for multimodal fusion later
# (TweFuse-W style weighted-sum fusion, Eq. 4-6 in the paper).
#
# This single script:
# 1. Loads master_dataset.csv
# 2. Applies the SAME image-availability filter + SAME
#    stratified split (random_state=42) as the image branch
# 3. Loads BERT (bert-base-uncased)
# 4. Fine-tunes BERT for fine-grained classification
# 5. Evaluates validation and test performance
# 6. Extracts 768-dimensional [CLS] text features
# 7. Saves the features and trained model in the same format
#    as the image branch's .pt files
#
# Run this in Kaggle / Colab.
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES AND CHECK GPU
# ============================================================

import os
import torch
import torch.nn as nn
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from tqdm.auto import tqdm

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# 2. INSTALL TRANSFORMERS
# ============================================================

# !pip install -q transformers


# ============================================================
# 3. CONFIG - EDIT THESE PATHS FOR YOUR ENVIRONMENT
# ============================================================
# BASE_PATH should point to the SAME extracted dataset root
# used by the image branch, so get_actual_image_path finds the
# same files and produces the same `available` dataframe.

BASE_PATH = "/kaggle/input/datasets/charumaurya15/multimodal-fake-news-processed"
csv_path = os.path.join(BASE_PATH, "master_dataset.csv")

# Candidate column names for the text field. The first one
# found in your CSV will be used automatically.
TEXT_COLUMN_CANDIDATES = [
    "text", "clean_title", "title", "headline", "content"
]


# ============================================================
# 4. LOAD MASTER DATASET
# ============================================================

df = pd.read_csv(csv_path)

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

TEXT_COLUMN = None
for candidate in TEXT_COLUMN_CANDIDATES:
    if candidate in df.columns:
        TEXT_COLUMN = candidate
        break

if TEXT_COLUMN is None:
    raise ValueError(
        "Could not auto-detect a text column. "
        f"Columns available: {df.columns.tolist()}. "
        "Set TEXT_COLUMN manually."
    )

print(f"\nUsing text column: '{TEXT_COLUMN}'")

print("\nLabel counts:")
print(df["label"].value_counts())

# Label convention:
# 0 = Real
# 1 = Fake


# ============================================================
# 5. FIND ACTUAL IMAGE PATHS (SAME LOGIC AS IMAGE BRANCH)
# ============================================================
# This filtering must exactly match the image branch so that
# `available` (and therefore the splits) contain the same rows
# in the same order. If your image branch's folder structure
# differs, update the `folder` values below to match it.

def get_actual_image_path(row):

    if pd.isna(row["image_path"]):
        return None

    filename = os.path.basename(str(row["image_path"]))

    if row["source"] == "fakenewsnet":
        folder = os.path.join(BASE_PATH, "fakenewsnet/fakenewsnet/images")
    elif row["source"] == "reddit":
        folder = os.path.join(BASE_PATH, "reddit/reddit/images")
    elif row["source"] == "twitter":
        folder = os.path.join(BASE_PATH, "twitter/twitter/images")
    else:
        return None

    path = os.path.join(folder, filename)

    if os.path.exists(path):
        return path

    return None


df["actual_image_path"] = df.apply(get_actual_image_path, axis=1)

available = df[df["actual_image_path"].notna()].copy()

print("\nTotal samples:", len(df))
print("Images available:", len(available))
print("Images missing:", len(df) - len(available))

# Drop rows with empty/NaN text, mirroring how the image branch
# drops rows without a usable image.
available = available[available[TEXT_COLUMN].notna()].copy()
available[TEXT_COLUMN] = available[TEXT_COLUMN].astype(str)

print("Samples with usable text + image:", len(available))

# ============================================================
# 5b. DETECT NUMBER OF CLASSES
# ============================================================
# Don't assume binary Fake/Real -- your label column may follow
# the paper's fine-grained taxonomy (e.g. 6 classes). Detect it
# from the data so the classifier head and loss line up.

unique_labels = sorted(available["label"].unique().tolist())
NUM_CLASSES = int(available["label"].max()) + 1

print("\nUnique label values:", unique_labels)
print("NUM_CLASSES set to:", NUM_CLASSES)

if len(unique_labels) != NUM_CLASSES:
    print(
        "WARNING: label values are not a contiguous 0..N-1 range. "
        "CrossEntropyLoss expects contiguous integer class ids -- "
        "you may need to remap labels."
    )


# ============================================================
# 6. TRAIN / VALIDATION / TEST SPLIT (IDENTICAL TO IMAGE BRANCH)
# ============================================================
# Same random_state, same ratios, same stratify column, on the
# same `available` dataframe -> identical split membership.

train_df, temp_df = train_test_split(
    available,
    test_size=0.20,
    random_state=42,
    stratify=available["label"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["label"]
)

print("\nTrain:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))


# ============================================================
# 7. TEXT DATASET CLASS
# ============================================================

class FakeNewsTextDataset(Dataset):

    def __init__(self, dataframe):
        self.data = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        text = row[TEXT_COLUMN]
        label = int(row["label"])
        return text, label


train_dataset = FakeNewsTextDataset(train_df)
val_dataset = FakeNewsTextDataset(val_df)
test_dataset = FakeNewsTextDataset(test_df)

print("\nDatasets created!")


# ============================================================
# 8. CUSTOM COLLATE FUNCTION
# ============================================================

def collate_fn(batch):
    texts, labels = zip(*batch)
    return list(texts), torch.tensor(labels)


# ============================================================
# 9. DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset, batch_size=16, shuffle=True,
    num_workers=0, collate_fn=collate_fn
)

val_loader = DataLoader(
    val_dataset, batch_size=16, shuffle=False,
    num_workers=0, collate_fn=collate_fn
)

test_loader = DataLoader(
    test_dataset, batch_size=16, shuffle=False,
    num_workers=0, collate_fn=collate_fn
)

print("DataLoaders created!")


# ============================================================
# 10. LOAD BERT
# ============================================================

from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

bert_encoder = AutoModel.from_pretrained("bert-base-uncased")
bert_encoder = bert_encoder.to(device)

print("BERT loaded!")

MAX_LENGTH = 80  # matches the paper's headline token budget


# ============================================================
# 11. TEST PREPROCESSING
# ============================================================

texts, labels = next(iter(train_loader))

inputs = tokenizer(
    texts,
    padding="max_length",
    truncation=True,
    max_length=MAX_LENGTH,
    return_tensors="pt"
)

print("\nNumber of texts:", len(texts))
print("Labels shape:", labels.shape)
print("input_ids shape:", inputs["input_ids"].shape)


# ============================================================
# 12. DEFINE BERT CLASSIFIER
# ============================================================
# We use the final [CLS] token representation (position 0 of
# last_hidden_state) as the 768-d sentence embedding, per the
# paper's Section III-C.

class BertFakeNewsClassifier(nn.Module):

    def __init__(self, text_model, num_classes):
        super().__init__()
        self.bert = text_model
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # [CLS] token representation
        features = outputs.last_hidden_state[:, 0, :]

        logits = self.classifier(features)

        return logits, features


model = BertFakeNewsClassifier(bert_encoder, NUM_CLASSES).to(device)

print("Model created!")


# ============================================================
# 13. FINE-TUNING SETUP
# ============================================================
# We fine-tune the complete BERT encoder, mirroring the
# image branch's approach (full fine-tune, not frozen).

for param in model.bert.parameters():
    param.requires_grad = True

model.classifier = nn.Linear(768, NUM_CLASSES).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    [
        {"params": model.bert.parameters(), "lr": 2e-5},
        {"params": model.classifier.parameters(), "lr": 1e-4}
    ]
)

print("Model ready for fine-tuning!")


# ============================================================
# 14. TRAIN THE TEXT MODEL
# ============================================================

EPOCHS = 3

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(
        train_loader,
        desc=f"Training Epoch {epoch + 1}/{EPOCHS}"
    )

    for texts, labels in progress_bar:

        inputs = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits, features = model(input_ids, attention_mask)

        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            accuracy=f"{correct / total:.4f}"
        )

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct / total

    print(
        f"Epoch {epoch + 1}: "
        f"Loss={epoch_loss:.4f}, "
        f"Accuracy={epoch_accuracy:.4f}"
    )


# ============================================================
# 15. EVALUATION FUNCTION
# ============================================================

def evaluate(loader, name):

    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for texts, labels in loader:

            inputs = tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt"
            )

            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            logits, features = model(input_ids, attention_mask)

            predictions = torch.argmax(logits, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())

    # average="macro" matches the paper's macro-F1 / macro-precision /
    # macro-recall reporting and works for both binary and multiclass.
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_predictions, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average="macro", zero_division=0)
    cm = confusion_matrix(all_labels, all_predictions)

    print(f"\n========== {name} RESULTS ==========")
    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)
    print("\nConfusion Matrix:")
    print(cm)

    return accuracy, precision, recall, f1, cm


# ============================================================
# 16. VALIDATION
# ============================================================

val_results = evaluate(val_loader, "VALIDATION")


# ============================================================
# 17. FINAL TEST EVALUATION
# ============================================================

test_results = evaluate(test_loader, "FINAL TEST")


# ============================================================
# 18. EXTRACT 768-D TEXT FEATURES
# ============================================================
# For multimodal fusion:
#
# Text -> BERT -> 768-dimensional [CLS] feature vector
#
# These are saved in the SAME order as the image branch's
# splits, so row i of train_text_features.pt corresponds to
# row i of train_image_features.pt.

def extract_features(loader):

    model.eval()

    all_features = []
    all_labels = []

    with torch.no_grad():

        for texts, labels in tqdm(loader, desc="Extracting features"):

            inputs = tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt"
            )

            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            logits, features = model(input_ids, attention_mask)

            all_features.append(features.cpu())
            all_labels.append(labels)

    return torch.cat(all_features), torch.cat(all_labels)


train_features, train_labels = extract_features(train_loader)
val_features, val_labels = extract_features(val_loader)
test_features, test_labels = extract_features(test_loader)


# ============================================================
# 19. PRINT FEATURE SHAPES
# ============================================================

print("\n========== FEATURE SHAPES ==========")
print("Train features:", train_features.shape)
print("Validation features:", val_features.shape)
print("Test features:", test_features.shape)


# ============================================================
# 20. SAVE TEXT FEATURES
# ============================================================
# Same dict format as the image branch's .pt files, so the
# fusion script can load both with an identical pattern.

torch.save(
    {"features": train_features, "labels": train_labels},
    "/kaggle/working/train_text_features_bert_multiclass.pt"
)

torch.save(
    {"features": val_features, "labels": val_labels},
    "/kaggle/working/val_text_features_bert_multiclass.pt"
)

torch.save(
    {"features": test_features, "labels": test_labels},
    "/kaggle/working/test_text_features_bert_multiclass.pt"
)

print("\nText features saved successfully!")


# ============================================================
# 21. SAVE TRAINED TEXT MODEL
# ============================================================

torch.save(
    model.state_dict(),
    "/kaggle/working/bert_fake_news_model_multiclass.pt"
)

print("Trained BERT model saved successfully!")


# ============================================================
# 22. FINAL OUTPUT FILES
# ============================================================

print("\n========== SAVED FILES ==========")
print("/kaggle/working/train_text_features_bert_multiclass.pt")
print("/kaggle/working/val_text_features_bert_multiclass.pt")
print("/kaggle/working/test_text_features_bert_multiclass.pt")
print("/kaggle/working/bert_fake_news_model_multiclass.pt")


# ============================================================
# END OF TEXT BRANCH
# ============================================================
#
# Final pipeline:
#
#                 TEXT
#                   |
#                   v
#          BERT Transformer
#                   |
#                   v
#             768-D FEATURES
#                   |
#          +--------+--------+
#          |                 |
#          v                 v
#      Fake/Real        MULTIMODAL FUSION
#      classifier              ^
#                               |
#                        IMAGE FEATURES
#
# Since train/val/test splits here are identical (row-for-row)
# to the image branch, train_text_features.pt[i] and
# train_image_features.pt[i] describe the same sample. Next
# step: a gating network per Eq. (5)-(6) of the paper --
# Linear(1536->128) -> ReLU -> Linear(128->2) -> softmax --
# that learns alpha_T, alpha_V and fuses
# F = alpha_T * T + alpha_V * V.
# ============================================================
