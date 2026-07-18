import pandas as pd
import json
import re
from nlp_english import stop_words

df = pd.read_csv('netflix_reviews.csv')
df['at'] = pd.to_datetime(df['at'], errors='coerce')
df = df[(df['at'].dt.year >= 2021) & (df['at'].dt.year <= 2026)].dropna(subset=['content'])

for _, r in df.iterrows():
    content = str(r['content'])
    words = content.lower().split()
    # check for emoji/emoticon by checking if there's any non-ascii char
    has_emo = any(ord(c) > 127 for c in content)
    if has_emo and len(words) > 5 and sum([1 for w in words if w in stop_words]) > 2:
        print(content)
        break
