import pandas as pd
import json
from preprocess_netflix import clean_text
from nlp_english import stop_words

df = pd.read_csv('netflix_reviews.csv')
df['at'] = pd.to_datetime(df['at'], errors='coerce')
df = df[(df['at'].dt.year >= 2021) & (df['at'].dt.year <= 2026)].dropna(subset=['content'])

valid = []
for _, r in df.iterrows():
    cl = clean_text(r['content'])
    words = cl.lower().split()
    if len(words) > 5 and sum([1 for w in words if w in stop_words]) > 2:
        valid.append({
            'content': r['content'], 
            'score': r['score'], 
            'at': r['at'].strftime('%Y-%m-%d %H:%M:%S')
        })
    if len(valid) == 10:
        break

with open('sample_10_english.json', 'w', encoding='utf-8') as f:
    json.dump(valid, f)
