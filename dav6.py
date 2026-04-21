#Method 1: Using Positive and Negative Word Count – With Normalization
import pandas as pd
import re

df = pd.read_csv('20191226-items.csv')

positive_words = set(['great', 'bright', 'excellent', 'good', 'quality', 'best', 'smooth'])
negative_words = set(['horrible', 'dirty', 'unpleasant', 'awful', 'bad', 'rugged', 'discontinued'])

def calculate_normalized_score(text):
    if pd.isna(text):
        return 0
    words = re.findall(r'\w+', text.lower())
    pos_count = sum(1 for word in words if word in positive_words)
    neg_count = sum(1 for word in words if word in negative_words)
    total_words = len(words)
    if total_words == 0:
        return 0
    return (pos_count - neg_count) / total_words

def classify_sentiment(score):
    if score > 0:
        return 'POSITIVE'
    elif score < 0:
        return 'NEGATIVE'
    else:
        return 'NEUTRAL'

df['Sentiment_Score'] = df['title'].apply(calculate_normalized_score)
df['Sentiment_Class'] = df['Sentiment_Score'].apply(classify_sentiment)

print(df[['title', 'Sentiment_Score', 'Sentiment_Class']].head(20))


#Method 2: Using Positive and Negative Word Count – With Semi-Normalization

import pandas as pd
import re

df = pd.read_csv('20191226-items.csv')

positive_words = set(['great', 'bright', 'excellent', 'good', 'quality', 'best', 'smooth'])
negative_words = set(['horrible', 'dirty', 'unpleasant', 'awful', 'bad', 'rugged', 'discontinued'])

def calculate_semi_normalized_score(text):
    if pd.isna(text):
        return 0
    words = re.findall(r'\w+', text.lower())
    pos_count = sum(1 for word in words if word in positive_words)
    neg_count = sum(1 for word in words if word in negative_words)
    
    emotive_sum = pos_count + neg_count
    if emotive_sum == 0:
        return 0
    return (pos_count - neg_count) / emotive_sum

df['Method2_Score'] = df['title'].apply(calculate_semi_normalized_score)
df['Method2_Class'] = df['Method2_Score'].apply(
    lambda x: 'POSITIVE' if x > 0 else ('NEGATIVE' if x < 0 else 'NEUTRAL')
)

print(df[['title', 'Method2_Score', 'Method2_Class']].head(10))
