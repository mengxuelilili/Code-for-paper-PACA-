# PACA‑Affinity: Sequence‑Based Antigen‑Antibody Binding Affinity Prediction

This repository contains the code for **PACA‑Affinity**, a sequence‑based deep learning model for predicting antigen–antibody binding affinity. The model leverages RoFormer and CNN architectures with attention mechanisms to achieve high accuracy using only amino acid sequences, without requiring 3D structural information. It is designed for applications in computational immunology and antibody engineering.

## 📁 Project Structure

> **Note**: All raw input data should be placed in the `data/` directory. Preprocessed data will be generated automatically during execution.

## Usage
### 1. Data Preprocessing
Run `dataset/datadeal.py` to perform data preprocessing.

Then run `dataset/pre_split_test_sets.py` first to perform fixed 6:2:2 train‑val‑test split for each dataset.
The partitioned test set will be saved as independent `.pt` files and **permanently locked**.
Only `train + val` subsets are used for main model experiments.

### 2. Model Training
Run `model/train.py` to train the model.

### 3. Model Prediction
Run `predictionPA.py` to make predictions using the trained model.

### 4. Model Architecture
Refer to `models/roformerccnn.py` to view the model architecture.

## Dependencies
- Python 3.8 or higher
- PyTorch 1.13.1
- NumPy
- Pandas
- Matplotlib



