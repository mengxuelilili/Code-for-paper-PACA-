# PACA-Affinity: Sequence-Based Antigen-Antibody Binding Affinity Prediction

This repository contains the code for **PACA-Affinity**, a sequence-based deep learning model for predicting antigen–antibody binding affinity. The model leverages RoFormer and CNN architectures with attention mechanisms to achieve high accuracy using only amino acid sequences, without requiring 3D structural information. It is designed for applications in computational immunology and antibody engineering.

## 📁 Project Structure

PACA-Affinity/
├── data/                   # Raw dataset files (e.g., CSV/FASTA with affinity values)
├── dataset/
│   └── datadeal.py         # Data preprocessing, encoding, and embedding generation
├── model/
│   └── train.py            # Training script
├── models/
│   └── roformerccnn.py     # Model architecture definition (RoFormer + CCNN)
├── pltantigen.py           # Script for antigen sequence processing
├── pltattentioncdr.py      # Script to visualize attention weights in CDR regions
├── pltCDR.py               # Script for CDR region extraction and analysis
├── predict.py              # Inference script for predicting binding affinity
└── README.md

**Note**: All raw input data should be placed in the `data/` directory. Preprocessed data will be generated automatically during execution.

##  Usage

### 1. Data Preprocessing
Run dataset/datadeal.py to perform data preprocessing.
Model Training
Run model/train.py to train the model.
Model Prediction
Run predict.py to make predictions using the trained model.
Model Architecture
Refer to models/roformerccnn.py to view the model architecture.
 
## Dependencies
Python 3.8 or higher
PyTorch 1.13.1
NumPy
Pandas
Matplotlib


