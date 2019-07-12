## Comment sentiment analysis with LightGBM

Dataset is from:
- VNPT AI team
- VietNLP team
- VLSP 2016
- Foody comment

Total more than 50k sample with 2 labels (0 for positive comment and 1 for negative comment). The baseline here using
gradient boosting for classification, implement by [LightGBM](https://github.com/microsoft/LightGBM).

For each sample, features are extracted contain some statistic feature as count words, emoji and simple classify emoji 
the sentiment, combine with tf-idf vector for each sentence.

Repo contain:
- Dataset to train and evaluate model (data.zip)
- Code for preprocess data and train model (make_model_sentiment.ipynb)
- Pre-trained sentiment classification model (model_gbm_sentiment.txt) and tf-idf model (vectorizer.pk)
- Code for infer new sample (infer_gbm.ipynb)
