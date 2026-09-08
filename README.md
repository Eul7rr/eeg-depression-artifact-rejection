# A Comparative Study of Artifact Rejection Methods for EEG-Based Depression Detection

This repository contains the complete, reproducible analysis pipeline for a 
study comparing four artifact handling strategies—no removal, and three ICA 
variants (Infomax, FastICA, Picard)—for resting-state EEG-based depression 
detection, evaluated with leave-one-subject-out cross-validation (LOSOCV).

## Data

The dataset is openly available on OpenNeuro: 
[ds003478](https://openneuro.org/datasets/ds003478) (version 1.1.0). 
Download the dataset and set the data path in `config.py` before running.

## Installation

    conda create -n eeg python=3.10
    conda activate eeg
    pip install -r requirements.txt

## Usage

    python main.py           # preprocessing, ICA, feature extraction, LOSOCV
    python roc_analysis.py   # ROC curves and AUC summary

Outputs (predictions, F1 summary, ROC/AUC tables, figures) are written to 
`results/`.

## Software

Python 3.10.20 with MNE-Python 1.12.1, NumPy 2.2.6, SciPy 1.15.3, and 
scikit-learn 1.7.2. See `requirements.txt` for the full dependency list.

## Citation

If you use this code, please cite the associated manuscript (citation to be 
added upon publication).

## License

MIT