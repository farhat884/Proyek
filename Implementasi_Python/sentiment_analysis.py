import re
import pandas as pd
import time
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def case_folding(text):
    return str(text).lower()

def normalization(text):
    norm_dict = {
        "can't": "can not",
        "i'm": "i am",
        "don't": "do not",
        "won't": "will not",
        "it's": "it is",
        "that's": "that is"
    }
    result = str(text)
    for k, v in norm_dict.items():
        result = result.replace(k, v)
    result = result.replace("'", "")
    return result

def remove_noise(text):
    result = re.sub(r'[^a-z0-9\s]', '', str(text))
    result = re.sub(r'\s+', ' ', result).strip()
    return result

def tokenization(text):
    return text.split()

def remove_stopwords(tokens):
    stopwords = {"i", "me", "is", "the", "and", "a", "an", "am", "not", "to", "it", "that", "this", "just"}
    return [word for word in tokens if word not in stopwords and not word.isdigit()]

def stemming(tokens):
    result = []
    for word in tokens:
        if word.endswith("ing") and len(word) > 4:
            word = word[:-3]
        elif word.endswith("ed") and len(word) > 3:
            word = word[:-2]
        elif word.endswith("s") and len(word) > 3 and not word.endswith("ss"):
            word = word[:-1]
        result.append(word)
    return result

def map_score_to_sentiment(score):
    if score >= 4:
        return "Positive"
    elif score <= 2:
        return "Negative"
    else:
        return "Neutral"

def clv_lifespan(sentiment):
    if sentiment.lower() == "positive":
        return 12
    elif sentiment.lower() == "neutral":
        return 6
    else:
        return 1

def clv_loss(lifespan):
    monthly_fee = 186000
    loss_months = 12 - lifespan
    return loss_months * monthly_fee

if __name__ == "__main__":
    start_time = time.time()
    
    print("Membaca file netflix_reviews.csv...")
    try:
        df = pd.read_csv("netflix_reviews.csv")
    except Exception as e:
        print(f"Error membaca file: {e}")
        sys.exit()
        
    df['content'] = df['content'].fillna('')
    print(f"Total data: {len(df)} baris.\n")
    
    # 0. Menyiapkan True Label dari kolom score
    if 'score' in df.columns:
        df['True Sentiment'] = df['score'].apply(map_score_to_sentiment)
    else:
        print("Kolom 'score' tidak ditemukan! Pastikan dataset memiliki kolom score.")
        sys.exit()
        
    pd.set_option('display.max_colwidth', 80)
        
    # 1. Case Folding
    print("--- 1. Proses Case Folding Selesai ---")
    df['1. Case Folding'] = df['content'].apply(case_folding)
    print(df[['content', '1. Case Folding']].head(10))
    print("\n")
    
    # 2. Normalization
    print("--- 2. Proses Normalization Selesai ---")
    df['2. Normalization'] = df['1. Case Folding'].apply(normalization)
    print(df[['1. Case Folding', '2. Normalization']].head(10))
    print("\n")
    
    # 3. Remove Noise
    print("--- 3. Proses Remove Noise Selesai ---")
    df['3. Remove Noise'] = df['2. Normalization'].apply(remove_noise)
    print(df[['2. Normalization', '3. Remove Noise']].head(10))
    print("\n")
    
    # 4. Tokenization
    print("--- 4. Proses Tokenization Selesai ---")
    df['4. Tokenization (List)'] = df['3. Remove Noise'].apply(tokenization)
    df['4. Tokenization'] = df['4. Tokenization (List)'].apply(str) 
    print(df[['3. Remove Noise', '4. Tokenization']].head(10))
    print("\n")
    
    # 5. Stopword Removal
    print("--- 5. Proses Stopword Removal Selesai ---")
    df['5. Stopword Removal (List)'] = df['4. Tokenization (List)'].apply(remove_stopwords)
    df['5. Stopword Removal'] = df['5. Stopword Removal (List)'].apply(str)
    print(df[['4. Tokenization', '5. Stopword Removal']].head(10))
    print("\n")
    
    # 6. Stemming
    print("--- 6. Proses Stemming Selesai ---")
    df['6. Stemming (List)'] = df['5. Stopword Removal (List)'].apply(stemming)
    # Untuk fitur TF-IDF, scikit-learn membutuhkan text utuh berspasi (bukan array)
    df['6. Stemming'] = df['6. Stemming (List)'].apply(lambda x: ' '.join(x)) 
    print(df[['5. Stopword Removal', '6. Stemming']].head(10))
    print("\n")
    
    # 7. TF-IDF dengan Scikit-Learn
    print("--- 7. Proses TF-IDF (Scikit-Learn) Selesai ---")
    # Menggunakan max_features=1000 agar memori tidak over-load saat training
    tfidf_vectorizer = TfidfVectorizer(max_features=1000)
    X_tfidf = tfidf_vectorizer.fit_transform(df['6. Stemming'])
    y = df['True Sentiment']
    print(f"Bentuk matrix TF-IDF: {X_tfidf.shape}")
    print("Cuplikan 10 kata fitur pertama yg dikenali:", tfidf_vectorizer.get_feature_names_out()[:10])
    print("\n")
    
    # 8. Training & Prediksi Random Forest
    print("--- 8. Proses Machine Learning: Random Forest Selesai ---")
    rf_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf_model.fit(X_tfidf, y)
    df['8. RF Prediction'] = rf_model.predict(X_tfidf)
    print(df[['True Sentiment', '8. RF Prediction']].head(10))
    print("\n")
    
    # 9. Training & Prediksi SVM
    print("--- 9. Proses Machine Learning: SVM (LinearSVC) Selesai ---")
    # LinearSVC berjalan jauh lebih cepat dari SVC biasa untuk dataset teks
    svm_model = LinearSVC(random_state=42)
    svm_model.fit(X_tfidf, y)
    df['9. SVM Prediction'] = svm_model.predict(X_tfidf)
    print(df[['True Sentiment', '9. SVM Prediction']].head(10))
    print("\n")
    
    # 10 & 11. Analisis Bisnis (CLV & Potential Loss)
    print("--- 10 & 11. Proses Kalkulasi CLV dan Potential Loss Selesai ---")
    df['10. CLV Lifespan (Bulan)'] = df['8. RF Prediction'].apply(clv_lifespan)
    df['11. Potential Loss (Rp)'] = df['10. CLV Lifespan (Bulan)'].apply(clv_loss)
    print(df[['8. RF Prediction', '10. CLV Lifespan (Bulan)', '11. Potential Loss (Rp)']].head(10))
    print("\n")
    
    # Membersihkan kolom array temporary
    df = df.drop(columns=['4. Tokenization (List)', '5. Stopword Removal (List)', '6. Stemming (List)'])
    
    excel_filename = "hasil_analisis_netflix_ml.xlsx"
    print(f"Pemrosesan teks selesai. Menyimpan {len(df)} baris ke {excel_filename} (Bisa butuh beberapa menit)...")
    
    df.to_excel(excel_filename, index=False)
    
    end_time = time.time()
    print(f"\nProses ML dan Ekstraksi selesai secara keseluruhan dalam {end_time - start_time:.2f} detik!")
    print(f"Hasil lengkap dari seluruh dataset telah disimpan ke {excel_filename}")
