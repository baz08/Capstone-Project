import pandas as pd
from cleaner import *


a_df = pd.read_csv("./BTC_comments.csv")
b_df = pd.read_csv("./ETH_comments.csv")
c_df = pd.read_csv("./sentiment.csv")
d_df = pd.read_csv("./sentiment_labels.csv")
c_df['body'] = c_df['Comment Text']


d_df['body'] = d_df['text']
d_df['Sentiment'] = d_df['sentiment']

d_df = d_df.drop(columns=['id', 'text', 'sentiment'])

c_df = c_df.drop(columns=['Comment Text', 'URL'])

gme_df = pd.concat([a_df, b_df, c_df, d_df])

gme_df['body'] = normalize_corpus(gme_df['body'])

gme_df['body'] = gme_df['body'].replace(
    r'https\S+', '', regex=True).replace(r'www\S+', '', regex=True)

gme_df.to_csv("./Crypto_c.csv", header=True, index=False)
