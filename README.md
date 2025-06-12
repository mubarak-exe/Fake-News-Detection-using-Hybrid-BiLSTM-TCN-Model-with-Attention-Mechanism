# Fake-News-Detection-using-Hybrid-BiLSTM-TCN-Model-with-Attention-Mechanism

![Project Logo](project_logo.png)

Welcome to the Fake News Detection Using Deep Learning project repository. This project focuses on developing deep learning models to detect fake news articles. By leveraging state-of-the-art techniques and utilizing the ISOT Fake News Dataset, we aim to build models that can effectively distinguish between genuine and fake news content.

## Table of Contents

- [Introduction](#introduction)
- [Project Details](#project-details)
  - [Importing the Required Libraries](#importing-the-required-libraries)
  - [Dataset](#dataset)
  - [Preprocessing](#preprocessing)
  - [Data Splitting](#data-splitting)
  - [Embedding Layer](#embedding-layer)
  - [Model Architectures](#model-architectures)
    - [CNN_LSTM](#cnn_lstm)
    - [CNN_Bi-LSTM_ATT](#cnn_bi-lstm_att)
    - [Bi-LSTM_ATT_TCN](#bi-lstm_att_tcn)
  - [Attention Mechanism](#attention-mechanism)
  - [Callback Function](#callback-function)
- [Experiment Training](#experiment-training)
- [Data Visualization](#data-visualization)

## Introduction

Fake news has become a significant concern in today's information-driven world. This project addresses this issue by utilizing deep learning models to classify news articles as either genuine or fake. The project employs the Python programming language and a variety of libraries, including NumPy, Pandas, NLTK, Sklearn, Matplotlib, TensorFlow, and Keras, to achieve its objectives.

## Project Details

### Importing the Required Libraries

The project's execution takes place within Google Colab, a cloud-based Jupyter notebook environment. Python serves as the primary coding language, with essential packages such as NumPy, Pandas, and NLTK utilized for data ingestion and preliminary processing. Sklearn is employed for data manipulation, partitioning, and outcome derivation. Visualization of results is facilitated by Matplotlib. Deep learning model construction is performed using TensorFlow, streamlined by Keras.

### Dataset

The ISOT Fake News Dataset, consisting of approximately 45,000 articles, forms the foundation of this project. The dataset includes two CSV files: "True.csv" containing about 21,418 genuine articles and "Fake.csv" with around 23,503 fake articles. Genuine articles originate from Reuters.com, while fake articles are sourced from flagged websites and Wikipedia, curated by fact-checking groups like Politifact. The dataset covers a wide range of subjects and includes titles, news text, subjects, and dates.

### Preprocessing

1. **Removal of Websites and URLs**: Lambda functions and regular expressions are utilized to eliminate stopwords, URLs, and IP addresses from the text corpus. NLTK aids in identifying stopwords.

2. **Stopword Removal**: NLTK's Porter stemmer is applied for stemming to reduce the presence of insignificant words while preserving context.

3. **Tokenization of Data**: Keras' tokenizer function is employed to tokenize sentences into smaller fragments. The tokenized text is encoded into integer sequences using the texts_to_sequences method.

### Data Splitting

The dataset is divided using the Train-Valid-Test methodology, with an 80-10-10 split. This approach ensures consistent data distribution across all experiments, allowing for effective training, validation, and testing of the models.

### Embedding Layer

The project employs the Embedding layer, which converts words into real-valued vectors in a condensed dimensionality. This approach captures word meanings through global context, utilizing embeddings of size 100 to encapsulate semantic information.

### Model Architectures

Three model architectures are employed in this project:

#### CNN_LSTM

#### CNN_Bi-LSTM_ATT

#### Bi-LSTM_ATT_TCN

### Attention Mechanism

The attention mechanism is a key component of the models, enabling the identification of relevant text segments through attention weights. This mechanism enhances efficiency by aligning and translating words effectively, yielding customized attention outputs for individual words.

### Callback Function

Callback functions play a crucial role in monitoring and influencing the training process. They handle tasks such as adjusting learning rates, saving model snapshots, enabling early stopping, and logging metrics, contributing to enhanced training and optimized model outcomes.

## Experiment Training

Experiments involve training the models on the training subset, validating their performance on the validation subset, and evaluating their efficacy on the designated test subset.

## Data Visualization

The project utilizes Matplotlib to generate visual representations, including graphs, that aid in understanding data distribution and model performance.

For a more in-depth understanding, detailed code implementation, and comprehensive results, please refer to the project's code files.

## Credits

This project was carried out by Mubarak Shikalgar as a part of Research Project Course at Sanjay Ghodawat University. We extend our gratitude to Dr. Chetan Arage and Dr. Deepika Patil for their valuable support and guidance.
