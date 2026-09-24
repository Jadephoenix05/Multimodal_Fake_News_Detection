# ============================================================
# COMPLETE IMAGE BRANCH - SWIN-TINY
# Multimodal Fake News Detection Project
# Binary Classification: 0 = Real, 1 = Fake
#
# This single script:
# 1. Extracts archive.zip
# 2. Loads master_dataset.csv
# 3. Finds available images
# 4. Creates train/validation/test splits
# 5. Loads Swin-Tiny
# 6. Fine-tunes Swin for Fake/Real classification
# 7. Evaluates validation and test performance
# 8. Extracts 768-dimensional image features
# 9. Saves the features and trained model
#
# Run this in Google Colab.
# Before running, upload archive.zip to /content/
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES AND CHECK GPU
# ============================================================

import os
import zipfile
import torch
import torch.nn as nn
import pandas as pd

from PIL import Image
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

!pip install -q transformers


# ============================================================
# 3. EXTRACT DATASET
# ============================================================

zip_path = "/content/archive.zip"
extract_path = "/content/fakenewsnet"

with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(extract_path)

print("Extraction completed!")


# ============================================================
# 4. LOAD MASTER DATASET
# ============================================================

csv_path = "/content/fakenewsnet/master_dataset.csv"

df = pd.read_csv(csv_path)

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

print("\nLabel counts:")
print(df["label"].value_counts())

print("\nSource counts:")
print(df["source"].value_counts())

# Label convention:
# 0 = Real
# 1 = Fake


# ============================================================
# 5. FIND ACTUAL IMAGE PATHS
# ============================================================
# The paths stored in master_dataset.csv do not directly match
# the extracted ZIP folder structure.
# Therefore, we map each image filename to its real location.

def get_actual_image_path(row):

    # Some records do not have images.
    if pd.isna(row["image_path"]):
        return None

    filename = os.path.basename(
        str(row["image_path"])
    )

    if row["source"] == "fakenewsnet":

        folder = (
            "/content/fakenewsnet/"
            "fakenewsnet/fakenewsnet/images"
        )

    elif row["source"] == "reddit":

        folder = (
            "/content/fakenewsnet/"
            "reddit/reddit/images"
        )

    elif row["source"] == "twitter":

        folder = (
            "/content/fakenewsnet/"
            "twitter/twitter/images"
        )

    else:
        return None

    path = os.path.join(
        folder,
        filename
    )

    if os.path.exists(path):
        return path

    return None


df["actual_image_path"] = df.apply(
    get_actual_image_path,
    axis=1
)

# Keep only samples for which an image exists.
available = df[
    df["actual_image_path"].notna()
].copy()

print("\nTotal samples:", len(df))
print("Images available:", len(available))
print(
    "Images missing:",
    len(df) - len(available)
)


# ============================================================
# 6. CHECK IMAGE LABEL DISTRIBUTION
# ============================================================

print("\nAvailable image samples by label:")

print(
    available["label"]
    .value_counts()
    .sort_index()
)

print("\n0 = Real")
print("1 = Fake")


# ============================================================
# 7. TRAIN / VALIDATION / TEST SPLIT
# ============================================================
# 80% training
# 10% validation
# 10% testing
#
# Stratification keeps the Fake/Real distribution similar
# in all three datasets.

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
# 8. IMAGE DATASET CLASS
# ============================================================
# Images remain PIL images here.
# The Hugging Face Swin processor will handle:
# - resizing
# - normalization
# - conversion to tensors

class FakeNewsImageDataset(Dataset):

    def __init__(self, dataframe):
        self.data = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        image = Image.open(
            row["actual_image_path"]
        ).convert("RGB")

        label = int(row["label"])

        return image, label


train_dataset = FakeNewsImageDataset(
    train_df
)

val_dataset = FakeNewsImageDataset(
    val_df
)

test_dataset = FakeNewsImageDataset(
    test_df
)

print("\nDatasets created!")


# ============================================================
# 9. CUSTOM COLLATE FUNCTION
# ============================================================
# PyTorch's default collate function cannot batch PIL images.
# This function keeps images as a list and converts labels
# into a tensor.

def collate_fn(batch):

    images, labels = zip(*batch)

    return (
        list(images),
        torch.tensor(labels)
    )


# ============================================================
# 10. DATALOADERS
# ============================================================
# num_workers=0 avoids multiprocessing problems in Colab.

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn
)

print("DataLoaders created!")


# ============================================================
# 11. LOAD SWIN-TINY
# ============================================================

from transformers import (
    AutoImageProcessor,
    SwinModel
)

processor = AutoImageProcessor.from_pretrained(
    "microsoft/swin-tiny-patch4-window7-224"
)

swin = SwinModel.from_pretrained(
    "microsoft/swin-tiny-patch4-window7-224"
)

swin = swin.to(device)

print("Swin Transformer loaded!")


# ============================================================
# 12. TEST PREPROCESSING
# ============================================================

images, labels = next(
    iter(train_loader)
)

inputs = processor(
    images=images,
    return_tensors="pt"
)

print("\nImage type:", type(images[0]))
print("Number of images:", len(images))
print("Labels shape:", labels.shape)
print(
    "Pixel values shape:",
    inputs["pixel_values"].shape
)
print(
    "Pixel values dtype:",
    inputs["pixel_values"].dtype
)


# ============================================================
# 13. DEFINE SWIN FAKE/REAL CLASSIFIER
# ============================================================
# Swin-Tiny produces a 768-dimensional representation.
#
# We mean-pool the Swin token representations and pass the
# resulting 768 features through a 2-class linear classifier.
#
# Class 0 = Real
# Class 1 = Fake

class SwinFakeNewsClassifier(nn.Module):

    def __init__(self, swin_model):

        super().__init__()

        self.swin = swin_model

        self.classifier = nn.Linear(
            768,
            2
        )

    def forward(self, pixel_values):

        outputs = self.swin(
            pixel_values=pixel_values
        )

        # Mean pooling over Swin tokens
        features = (
            outputs.last_hidden_state
            .mean(dim=1)
        )

        logits = self.classifier(
            features
        )

        return logits, features


model = SwinFakeNewsClassifier(
    swin
).to(device)

print("Model created!")


# ============================================================
# 14. FINE-TUNING SETUP
# ============================================================
# We fine-tune the complete Swin model.
#
# We do NOT use class weighting here.
# The earlier weighted experiment pushed the model too strongly
# toward predicting Fake.

for param in model.swin.parameters():
    param.requires_grad = True

# Reset the classification head.
model.classifier = nn.Linear(
    768,
    2
).to(device)

# Standard loss for binary classes represented as 2 logits.
criterion = nn.CrossEntropyLoss()

# Small learning rate for pretrained Swin.
# Slightly larger learning rate for the new classifier.

optimizer = torch.optim.AdamW(
    [
        {
            "params": model.swin.parameters(),
            "lr": 1e-5
        },
        {
            "params": model.classifier.parameters(),
            "lr": 1e-4
        }
    ]
)

print("Model ready for fine-tuning!")


# ============================================================
# 15. TRAIN THE IMAGE MODEL
# ============================================================

EPOCHS = 3

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(
        train_loader,
        desc=(
            f"Training Epoch "
            f"{epoch + 1}/{EPOCHS}"
        )
    )

    for images, labels in progress_bar:

        # Convert PIL images into Swin input tensors.
        inputs = processor(
            images=images,
            return_tensors="pt"
        )

        pixel_values = inputs[
            "pixel_values"
        ].to(device)

        labels = labels.to(device)

        # Clear old gradients.
        optimizer.zero_grad()

        # Forward pass.
        logits, features = model(
            pixel_values
        )

        # Calculate classification loss.
        loss = criterion(
            logits,
            labels
        )

        # Backpropagation.
        loss.backward()

        # Update model weights.
        optimizer.step()

        running_loss += loss.item()

        # Calculate training accuracy.
        predictions = torch.argmax(
            logits,
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            accuracy=f"{correct / total:.4f}"
        )

    epoch_loss = (
        running_loss /
        len(train_loader)
    )

    epoch_accuracy = (
        correct /
        total
    )

    print(
        f"Epoch {epoch + 1}: "
        f"Loss={epoch_loss:.4f}, "
        f"Accuracy={epoch_accuracy:.4f}"
    )


# ============================================================
# 16. EVALUATION FUNCTION
# ============================================================
# This function calculates:
# - Accuracy
# - Precision
# - Recall
# - F1 score
# - Confusion matrix

def evaluate(
    loader,
    name
):

    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for images, labels in loader:

            inputs = processor(
                images=images,
                return_tensors="pt"
            )

            pixel_values = inputs[
                "pixel_values"
            ].to(device)

            logits, features = model(
                pixel_values
            )

            predictions = torch.argmax(
                logits,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.numpy()
            )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    cm = confusion_matrix(
        all_labels,
        all_predictions
    )

    print(
        f"\n========== {name} RESULTS =========="
    )

    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)

    print("\nConfusion Matrix:")
    print(cm)

    return (
        accuracy,
        precision,
        recall,
        f1,
        cm
    )


# ============================================================
# 17. VALIDATION
# ============================================================

val_results = evaluate(
    val_loader,
    "VALIDATION"
)


# ============================================================
# 18. FINAL TEST EVALUATION
# ============================================================
# The test set is kept separate from training and validation.
# This is the final standalone performance of the image branch.

test_results = evaluate(
    test_loader,
    "FINAL TEST"
)


# ============================================================
# 19. EXTRACT 768-D IMAGE FEATURES
# ============================================================
# For multimodal fusion, we need the internal image features,
# not just Fake/Real predictions.
#
# Each image becomes:
#
# Image -> Swin -> 768-dimensional feature vector
#
# These vectors will later be combined with the text features.

def extract_features(loader):

    model.eval()

    all_features = []
    all_labels = []

    with torch.no_grad():

        for images, labels in tqdm(
            loader,
            desc="Extracting features"
        ):

            inputs = processor(
                images=images,
                return_tensors="pt"
            )

            pixel_values = inputs[
                "pixel_values"
            ].to(device)

            logits, features = model(
                pixel_values
            )

            # Save features on CPU to reduce GPU memory usage.
            all_features.append(
                features.cpu()
            )

            all_labels.append(
                labels
            )

    return (
        torch.cat(all_features),
        torch.cat(all_labels)
    )


# Extract train features
train_features, train_labels = (
    extract_features(train_loader)
)

# Extract validation features
val_features, val_labels = (
    extract_features(val_loader)
)

# Extract test features
test_features, test_labels = (
    extract_features(test_loader)
)


# ============================================================
# 20. PRINT FEATURE SHAPES
# ============================================================

print("\n========== FEATURE SHAPES ==========")

print(
    "Train features:",
    train_features.shape
)

print(
    "Validation features:",
    val_features.shape
)

print(
    "Test features:",
    test_features.shape
)

# Expected:
# Train      -> [8108, 768]
# Validation -> [1014, 768]
# Test       -> [1014, 768]


# ============================================================
# 21. SAVE IMAGE FEATURES
# ============================================================
# These files can be used later by the multimodal fusion model.
#
# This means we do not need to run Swin again just to obtain
# the same image features.

torch.save(
    {
        "features": train_features,
        "labels": train_labels
    },
    "/content/train_image_features.pt"
)

torch.save(
    {
        "features": val_features,
        "labels": val_labels
    },
    "/content/val_image_features.pt"
)

torch.save(
    {
        "features": test_features,
        "labels": test_labels
    },
    "/content/test_image_features.pt"
)

print(
    "\nImage features saved successfully!"
)


# ============================================================
# 22. SAVE TRAINED IMAGE MODEL
# ============================================================
# This saves the trained Swin model weights.

torch.save(
    model.state_dict(),
    "/content/swin_fake_news_model.pt"
)

print(
    "Trained Swin model saved successfully!"
)


# ============================================================
# 23. FINAL OUTPUT FILES
# ============================================================

print("\n========== SAVED FILES ==========")

print(
    "/content/train_image_features.pt"
)

print(
    "/content/val_image_features.pt"
)

print(
    "/content/test_image_features.pt"
)

print(
    "/content/swin_fake_news_model.pt"
)


# ============================================================
# END OF IMAGE BRANCH
# ============================================================
#
# Final pipeline:
#
#                 IMAGE
#                   |
#                   v
#          Swin-Tiny Transformer
#                   |
#                   v
#             768-D FEATURES
#                   |
#          +--------+--------+
#          |                 |
#          v                 v
#      Fake/Real        MULTIMODAL FUSION
#      classifier              ^
#                              |
#                       TEXT FEATURES
#
# The image branch is now ready to be combined with the
# text branch.
#
# ============================================================
