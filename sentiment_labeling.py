import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os
from collections import Counter

generic_words = {'netflix', 'app', 'movie', 'show', 'time', 'watch', 'even', 'one', 'get', 'just', 'really', 'please', 'plz', 'phone', 'would', 'could', 'can', 'make', 'use', 'want', 'see', 'also', 'much', 'many', 'way', 'thing', 'say', 'go', 'know'}

def filter_generic_words(text):
    words = text.split()
    filtered = [w for w in words if w.lower() not in generic_words]
    return " ".join(filtered)

def get_sentiment(row):
    text = row['content']
    score = row['score']
    
    # Prioritaskan rating dari user jika tersedia
    if pd.notna(score):
        try:
            score = float(score)
            if score >= 4:
                return 'positive'
            elif score <= 2:
                return 'negative'
            else:
                return 'neutral'
        except ValueError:
            pass
            
    # Fallback ke TextBlob jika score tidak valid atau tidak ada
    if not isinstance(text, str):
        return 'neutral'
    
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0.05:
        return 'positive'
    elif analysis.sentiment.polarity < -0.05:
        return 'negative'
    else:
        return 'neutral'

def generate_wordcloud(df, sentiment, title, filename):
    text = " ".join(review for review in df[df['sentiment'] == sentiment].content.astype(str))
    if not text.strip():
        print(f"Peringatan: Tidak ada data untuk WordCloud {sentiment}")
        return
        
    text = filter_generic_words(text)
    if not text.strip():
        return
        
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title)
    plt.axis('off')
    plt.savefig(f"{filename}")
    plt.close()
    print(f"WordCloud disimpan: {filename}")

def generate_barchart(df, sentiment, title, filename):
    text = " ".join(review for review in df[df['sentiment'] == sentiment].content.astype(str))
    if not text.strip():
        return
        
    text = filter_generic_words(text)
    words = text.split()
    if not words:
        return
        
    word_counts = Counter(words)
    top_5 = word_counts.most_common(5)
    
    words_list = [w[0] for w in top_5]
    counts_list = [w[1] for w in top_5]
    
    plt.figure(figsize=(10, 5))
    plt.barh(words_list[::-1], counts_list[::-1], color='skyblue')
    plt.title(title)
    plt.xlabel('Frequency')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"BarChart disimpan: {filename}")

def main():
    input_file = "netflix_reviews_preprocessed.csv"
    output_file = "netflix_reviews_labeled.csv"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} tidak ditemukan!")
        return

    print(f"Membaca {input_file}...")
    df = pd.read_csv(input_file)
    
    print("Melakukan labeling sentiment dengan menggabungkan TextBlob dan Rating User...")
    df['sentiment'] = df.apply(get_sentiment, axis=1)
    
    # Menampilkan distribusi label
    print("\nDistribusi Sentiment:")
    print(df['sentiment'].value_counts())
    
    # Optimasi: Hapus data yang mungkin kosong (lagi) setelah pemrosesan
    df = df.dropna(subset=['content', 'sentiment'])
    
    print(f"Menyimpan hasil ke {output_file}...")
    df.to_csv(output_file, index=False)
    
    # Visualisasi dengan WordCloud
    print("\nMembuat WordCloud untuk evaluasi visual...")
    generate_wordcloud(df, 'positive', 'Positive Sentiment Word Cloud', 'wordcloud_positive.png')
    generate_wordcloud(df, 'negative', 'Negative Sentiment Word Cloud', 'wordcloud_negative.png')
    generate_wordcloud(df, 'neutral', 'Neutral Sentiment Word Cloud', 'wordcloud_neutral.png')
    
    print("\nMembuat BarChart Top 5 Kata...")
    generate_barchart(df, 'positive', 'Top 5 Words - Positive', 'barchart_positive.png')
    generate_barchart(df, 'negative', 'Top 5 Words - Negative', 'barchart_negative.png')
    generate_barchart(df, 'neutral', 'Top 5 Words - Neutral', 'barchart_neutral.png')
    
    print("-" * 30)
    print("Labeling SELESAI!")
    print(f"File: {output_file}")
    print("-" * 30)

if __name__ == "__main__":
    main()
