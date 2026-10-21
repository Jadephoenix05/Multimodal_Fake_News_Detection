# Multimodal_Fake_News_Detection

### Research_Paper_Links
`https://www.researchgate.net/publication/400288968_Interpreting_Multimodal_Fake_News_Detection_Models_An_Experimental_Study_of_Performance_Factors_and_Modality_Contributions/link/697e0fd564ca8a382087e658/download?_tp=eyJjb250ZXh0Ijp7ImZpcnN0UGFnZSI6InB1YmxpY2F0aW9uIiwicGFnZSI6InB1YmxpY2F0aW9uIn19`
# Multimodal Fake News Detection

## Overview
A deep learning project that detects fake news by analyzing both textual and visual information. It combines Natural Language Processing (NLP) and Computer Vision to improve news classification.

## Features
- Text and image preprocessing
- Multimodal feature extraction
- Adaptive fusion of text and image features
- Fake news classification into six categories

## Model Architecture
- **Text Encoder:** BERTweet
- **Image Encoder:** Swin Transformer
- **Fusion:** TweFuse-W (adaptive multimodal fusion)
- **Output:** Six-class classification

## Dataset
The project uses data from:
- [Fakeddit](https://github.com/entitize/Fakeddit)
- [FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet)
- [PHEME](https://github.com/kochkinaelena/Multimedia-Untrue-News-Detection)

The datasets are preprocessed and combined into a master dataset.

## Workflow
1. Collect and combine datasets.
2. Clean and preprocess text and images.
3. Extract textual and visual features.
4. Apply adaptive multimodal fusion.
5. Classify news into six categories.
6. Evaluate performance using standard classification metrics.

## Classification Categories
- True
- Satire
- Misleading Content
- False Connection
- Manipulated Content
- Imposter Content

## Evaluation
Model performance is evaluated using Accuracy, Precision, Recall, and Macro-F1 Score.

**Reported Macro-F1 Score:** 0.838 for TweFuse-W.

## Technologies
- Python
- PyTorch
- Hugging Face Transformers
- BERTweet
- Swin Transformer
- Pandas and NumPy
- Scikit-learn

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/Jadephoenix05/Multimodal_Fake_News_Detection.git
   cd Multimodal_Fake_News_Detection
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the preprocessing, training, and evaluation scripts as applicable.

> Note: Ensure that the dependency file and script names match the current repository structure before running these commands. Download and preprocess the datasets separately if they are not included in the repository.

## Purpose
This project explores multimodal deep learning techniques for identifying misinformation using textual and visual evidence.
