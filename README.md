# Disaster Tweet Classification using LSTM and GloVe Embeddings

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.x-brightgreen.svg)
![Status](https://img.shields.io/badge/status-Complete-success.svg)

## Author
### Panagiotis S. Georgiadis

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Model Architecture](#model-architecture)
- [Performance](#performance)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Results](#results)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project focuses on classifying tweets as disaster-related or non-disaster-related using a deep learning approach. We implemented an LSTM model initialized with GloVe embeddings to capture the semantic meaning of the text, and performed hyperparameter tuning to optimize the model's performance.

The final model can be used in real-world disaster monitoring systems to automatically identify tweets that are relevant to ongoing disasters.

## Dataset

The dataset used in this project is provided by [Kaggle’s Natural Language Processing with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started). It contains two columns:
- `text`: The tweet content.
- `target`: Binary labels (1 for disaster-related, 0 for non-disaster-related).

## Project Structure

```bash
├── data
│   ├── train.csv             # Training data
│   ├── test.csv              # Test data
├── glove.6B.300d.txt         # GloVe Embeddings (300-dimensional)
├── hyperband_tuning
│   └── disaster_tweets_lstm  # Hyperparameter trials using keras hyperband
├── notebooks
│   └── disaster_tweets.ipynb # Jupyter notebook with the full workflow
├── threshold_optimization_results.json  # JSON file with the best threshold and metrics
├── submission.csv            # CSV file for submission
├── README.md                 # Project README file
├── images                    # Images from notebook
└── requirements.txt          # List of Python dependencies

```

## Installation

To run this project, you will need Python 3.x and the required libraries listed in `requirements.txt`.

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/disaster-tweet-classification.git
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the GloVe embeddings from [this link](https://nlp.stanford.edu/projects/glove/) and place the `glove.6B.300d.txt` file in the root directory.

## Model Architecture

We used an LSTM network for this project, initialized with pre-trained GloVe embeddings. The architecture includes:
- **Embedding Layer:** Initialized with GloVe embeddings (300d).
- **LSTM Layers:** Two LSTM layers with 64 and 128 units respectively.
- **Dense Layer:** One dense layer with 128 units and ReLU activation.
- **Dropout:** Applied dropout (rate = 0.3) to reduce overfitting.
- **Output Layer:** Sigmoid activation for binary classification.

The model was compiled with the Adam optimizer and trained with binary cross-entropy loss.

## Hyperparameter Tuning

We used Keras Tuner's `Hyperband` algorithm to optimize the model's hyperparameters. The best configuration found after tuning includes:
- **LSTM Units 1:** 64
- **LSTM Units 2:** 128
- **Dense Units:** 128
- **Dropout Rate:** 0.3
- **Learning Rate:** 0.00123
- **Trainable Embeddings:** True

## Performance

The final model achieved the following performance on the validation set:
- **Accuracy:** 82%
- **Precision (Disaster):** 0.89
- **Recall (Disaster):** 0.71
- **F1-Score (Disaster):** 0.79

A detailed confusion matrix and precision-recall curves are available in the notebook.

## Results

The optimal threshold for the model was determined to be 0.374. This threshold balances precision and recall to improve the detection of disaster-related tweets.

- **Confusion Matrix with Optimal Threshold:**
    ```
    Non-Disaster: Precision = 0.79, Recall = 0.92
    Disaster: Precision = 0.89, Recall = 0.71
    ```

## Future Work

Potential improvements for future iterations of this project include:
- Experimenting with transformer-based models such as BERT or RoBERTa to capture more context.
- Implementing an ensemble of different NLP models for more robust predictions.
- Applying advanced feature engineering techniques to extract more meaningful signals from the tweet text.

## Contributing

If you would like to contribute to this project, feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
