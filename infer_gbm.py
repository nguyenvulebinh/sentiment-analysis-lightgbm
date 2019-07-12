import pickle as pickle
import lightgbm as lgb
import emoji
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy.sparse import hstack, csr_matrix, vstack
import numpy as np
analyzer_emoji = SentimentIntensityAnalyzer()


# ### Load trained model

print('Loading tfidf model...')
tfidf = pickle.load(open("./vectorizer.pk", "rb" ))


print('Loading model sentiment to predict...')
sentiment_model = lgb.Booster(model_file='./model_gbm_sentiment.txt')


# ### Utils extract emoji sentiments


def extract_emojis(str):
    return [c for c in str if c in emoji.UNICODE_EMOJI]
def sentiment_emojis(sentence):
    emojis = extract_emojis(sentence)
    result = [0,0,0,0]
    if len(emojis) == 0:
        return result
    for icon in emojis:
        sen_dict = analyzer_emoji.polarity_scores(icon)
        sen = [sen_dict['neg'],sen_dict['neu'],sen_dict['pos'],sen_dict['compound']]
        result = [result[i] + sen[i] for i in range(4)]
    return [result[i] / len(emojis) for i in range(4)]
def sentiment_emojis_row(row):
    comment = row['comment']
    sen_comment = sentiment_emojis(comment)
    
    row['emoji_neg'] = sen_comment[0]
    row['emoji_neu'] = sen_comment[1]
    row['emoji_pos'] = sen_comment[2]
    row['emoji_compound'] = sen_comment[3]
    
    return row


# ### Clean input

def clean_input(input_str):
    if len(input_str) == 0:
        input_str = ' '
    return input_str.lower()


# ### Get statistic feature

def get_statistic_feature(input_str): 
    # Add num words of comment as feature
    num_words = len(input_str.split())
    # Add num words unique of comment as feature
    num_unique_words = len(set(w for w in input_str.split()))
    # Add num words unique per num words of comment as feature
    words_vs_unique = num_unique_words / num_words * 100
    # Add emojis features
    emoji_neg, emoji_neu, emoji_pos, emoji_compound = sentiment_emojis(input_str)
    return np.array([num_words, num_unique_words, words_vs_unique, emoji_neg, emoji_neu, emoji_pos, emoji_compound])


# ### TfIdf vector

def sent2vec(input_str):
    return tfidf.transform([input_str])[0]


# ### Get sentence features

def sent2features(input_str):
    return hstack([sent2vec(input_str), csr_matrix(get_statistic_feature(input_str))]).tocsr()

if __name__ == '__main__': 

    # ### Infer sentences
    list_input = [
        '😀 tốt quá',
        'thấy không được',
        'quá tệ',
        'tạm ổn'
    ]
    result = [sentiment_model.predict(sent2features(sen)) for sen in list_input]
    for str_input, sentiment in zip(list_input, result):
        print('=))' if sentiment < 0.5 else '=((', "%.2f" % sentiment, str_input)

