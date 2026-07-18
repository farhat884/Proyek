import pandas as pd
import json
from preprocess_netflix import clean_text
from nlp_english import normalize_text, preprocess_steps
from sentiment_labeling import get_sentiment
from nltk.tokenize import word_tokenize

df = pd.DataFrame(json.load(open('sample_10_english.json', encoding='utf-8')))
res = []
for _, row in df.iterrows():
    t_clean = clean_text(row['content'])
    t_lower = t_clean.lower()
    t_norm = normalize_text(t_lower)
    tokens = word_tokenize(t_norm)
    t_pre = preprocess_steps(row['content'])
    s = get_sentiment({'content': t_pre, 'score': row['score']})
    res.append({
        'ori': row['content'], 
        'score': row['score'], 
        'clean': t_clean, 
        'lower': t_lower, 
        'norm': t_norm,
        'tokens': tokens,
        'final': t_pre, 
        'sentiment': s
    })

with open('sim_res_english.json', 'w', encoding='utf-8') as f:
    json.dump(res, f, indent=2)
