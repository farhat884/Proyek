import pandas as pd
import re
import os
import emoji
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\S+', '', text)
    text = re.sub(r'#\S+', '', text)
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = " ".join(text.split()).lower()
    
    # Tokenization, Stopwords, Stemming
    tokens = word_tokenize(text)
    cleaned_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(cleaned_tokens)

def get_sentiment(row):
    score = row['score']
    text = row['content']
    
    if pd.notna(score):
        try:
            score = float(score)
            if score >= 4: return 'positive'
            elif score <= 2: return 'negative'
            else: return 'neutral'
        except ValueError:
            pass
            
    if not isinstance(text, str):
        return 'neutral'
    
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0.05: return 'positive'
    elif analysis.sentiment.polarity < -0.05: return 'negative'
    else: return 'neutral'

def run_pipeline():
    input_file = "../netflix_reviews.csv"
    output_cleaned = "netflix_cleaned_python.csv"
    output_labeled = "netflix_labeled_python.csv"
    
    print("--- POIN 1: PREPROCESSING DATA ---")
    df = pd.read_csv(input_file)
    print(f"Total baris asli: {len(df)}")
    
    # Pemotongan (Cutting) Data: Hanya tahun 2021 ke atas
    if 'at' in df.columns:
        df['at_dt_filter'] = pd.to_datetime(df['at'], errors='coerce')
        df = df[df['at_dt_filter'].dt.year >= 2021].copy()
        df = df.drop(columns=['at_dt_filter'])
        print(f"Total baris setelah filter tahun (>= 2021): {len(df)}")
    
    df = df.dropna(subset=['content'])
    df['content'] = df['content'].apply(clean_text)
    df = df[df['content'].str.strip() != ""]
    print(f"Total baris setelah cleaning: {len(df)}")
    df.to_csv(output_cleaned, index=False)
    
    print("\n--- POIN 2: LABELING SENTIMEN ---")
    df['sentiment'] = df.apply(get_sentiment, axis=1)
    df.to_csv(output_labeled, index=False)
    print("Distribusi Sentimen:")
    print(df['sentiment'].value_counts())
    
    print("\n--- POIN 3: MACHINE LEARNING (TF-IDF & SVM) ---")
    X = df['content']
    y = df['sentiment']
    
    # Supaya tidak terlalu berat, kita gunakan subset untuk TF-IDF jika terlalu besar
    # Tapi demi instruksi, kita split semuanya
    print("Mengekstraksi fitur (TF-IDF)...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_tfidf = vectorizer.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.3, random_state=42)
    
    print("Melatih model SVM (Support Vector Machine)...")
    svm_model = SVC(kernel='linear', random_state=42)
    svm_model.fit(X_train, y_train)
    
    print("Melakukan prediksi SVM...")
    y_pred_svm = svm_model.predict(X_test)
    
    acc_svm = accuracy_score(y_test, y_pred_svm)
    print(f"\nAkurasi Model SVM: {acc_svm:.4f}")
    print("\nLaporan Klasifikasi SVM:")
    print(classification_report(y_test, y_pred_svm))
    
    print("Melatih model Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, class_weight='balanced')
    rf_model.fit(X_train, y_train)
    
    print("Melakukan prediksi Random Forest...")
    y_pred_rf = rf_model.predict(X_test)
    
    acc_rf = accuracy_score(y_test, y_pred_rf)
    print(f"\nAkurasi Model Random Forest: {acc_rf:.4f}")
    print("\nLaporan Klasifikasi Random Forest:")
    print(classification_report(y_test, y_pred_rf))
    
    print("\n--- POIN 3: ANALISIS BISNIS (CLV & CHURN) ---")
    # Menggunakan model SVM karena terbukti lebih baik pada laporan metodologi
    df['pred_sentiment'] = svm_model.predict(X_tfidf)
    
    if 'at' in df.columns and 'userName' in df.columns:
        df['at_dt'] = pd.to_datetime(df['at'], errors='coerce')
        df = df.sort_values(by=['userName', 'at_dt'])
        
        # Lifespan
        def get_lifespan(sentiment):
            if sentiment == 'positive': return 12
            elif sentiment == 'neutral': return 6
            else: return 1
            
        df['lifespan'] = df['pred_sentiment'].apply(get_lifespan)
        df['clv'] = df['lifespan'] * 120000
        
        # Detect churn
        df['prev_sentiment'] = df.groupby('userName')['pred_sentiment'].shift(1)
        df['is_churn_event'] = ((df['prev_sentiment'].isin(['positive', 'neutral'])) & (df['pred_sentiment'] == 'negative'))
        df['potential_loss'] = df['is_churn_event'].apply(lambda x: 1200000 if x else 0)
        
        output_clv = "netflix_clv_python.csv"
        df.to_csv(output_clv, index=False)
        print(f"Total Potensi Kerugian Finansial: Rp {df['potential_loss'].sum():,}")
    
    print("\nPROSES PYTHON SELESAI!")

if __name__ == "__main__":
    run_pipeline()
