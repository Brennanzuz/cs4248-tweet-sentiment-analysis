# cs4248-tweet-sentiment-analysis

This repository contains the code for CS4248 Team 27's group project on **Sentiment Analysis of Tweets for Better Interpretations of Social Media Posts** based off the Tweet Sentiment Analysis dataset (TSAD) from [Kaggle](https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset/data?select=train.csv).

Important files in these branches include:
- `main`:
  - `brennanzuz_transformers.ipynb` holds the code for the simple implementation of the BERT model with a few added layers using the simple preprocessed data `train_preprocessed.csv` and `test_preprocessed.csv`.
  - `model2.ipynb` explores the use of Word2Vec word embeddings and many other different contextual embeddings for the same task. It also contains the code for feature extraction and other preprocessing tasks `train_merged.csv` and `test_merged.csv`.
- `static-embeddings`:
  - `brennanzuz_transformers.ipynb` contains code for running static GloVe word embeddings using the simple preprocessed data `train_preprocessed.csv` and `test_preprocessed.csv`.
- `84ebd71ee0658a63af6dc378def21af24152d351`:
  - `brennanzuz_transformers.ipynb` contains code for running a custom GloVe transformer model using the simple preprocessed data `train_preprocessed.csv` and `test_preprocessed.csv`.