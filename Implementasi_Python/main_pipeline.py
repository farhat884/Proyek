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

# Download resource NLTK jika belum ada
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# =====================================================================
# 1. FUNGSI CLEANING TEKS INDIVIDUAL & TAHAP PREPROCESSING
# =====================================================================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\S+', '', text)
    text = re.sub(r'#\S+', '', text)
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = " ".join(text.split()).lower()
    
    tokens = word_tokenize(text)
    cleaned_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(cleaned_tokens)

def process_preprocessing(df):
    print("\n==================================================")
    print(" 📌 TAHAP 1: PREPROCESSING DATA")
    print("==================================================")
    print(f"[+] Total baris data mentah : {len(df):,} baris")
    
    # Filter tahun >= 2021
    if 'at' in df.columns:
        df['at_dt_filter'] = pd.to_datetime(df['at'], errors='coerce')
        df = df[df['at_dt_filter'].dt.year >= 2021].copy()
        df = df.drop(columns=['at_dt_filter'])
        print(f"[+] Filter Tahun (>= 2021)  : {len(df):,} baris tersisa")
    
    # Hapus missing values & lakukan cleaning
    df = df.dropna(subset=['content']).copy()
    sample_before = df['content'].head(2).tolist()
    
    print("[+] Melakukan Text Cleaning (Lowercasing, Regex, Stopwords, Stemming)...")
    df['cleaned_content'] = df['content'].apply(clean_text)
    df = df[df['cleaned_content'].str.strip() != ""].copy()
    print(f"[+] Total setelah Cleaning  : {len(df):,} baris valid")
    
    print("\n--- 🔍 [OUTPUT] Sampel Perubahan Cleaning Teks ---")
    for i, (before, after) in enumerate(zip(sample_before, df['cleaned_content'].head(2)), 1):
        print(f"  {i}. TEKS ASLI   : {before}")
        print(f"     TEKS BERSIH : {after}\n")
        
    return df

# =====================================================================
# 2. FUNGSI LABELING SENTIMEN
# =====================================================================
def get_sentiment(row):
    score = row.get('score', None)
    text = row.get('content', '')
    
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

def process_labeling(df):
    print("==================================================")
    print(" 📌 TAHAP 2: SENTIMENT LABELING")
    print("==================================================")
    print("[+] Menghitung label sentimen (berdasarkan Rating & TextBlob)...")
    df['sentiment'] = df.apply(get_sentiment, axis=1)
    
    print("\n--- 📊 [OUTPUT] Distribusi Label Sentimen ---")
    counts = df['sentiment'].value_counts()
    for label, count in counts.items():
        percentage = (count / len(df)) * 100
        print(f"  • {label.capitalize():<10} : {count:>7,} data ({percentage:.2f}%)")
        
    print("\n--- 🔍 [OUTPUT] Pratinjau Hasil Labeling ---")
    cols_to_show = [c for c in ['score', 'cleaned_content', 'sentiment'] if c in df.columns]
    print(df[cols_to_show].head(3).to_string(index=False))
    print("\n")
    return df

# =====================================================================
# 3. FUNGSI MACHINE LEARNING (TF-IDF, SVM & RANDOM FOREST)
# =====================================================================
def process_machine_learning(df):
    print("==================================================")
    print(" 📌 TAHAP 3: MACHINE LEARNING MODELING")
    print("==================================================")
    
    X = df['cleaned_content']
    y = df['sentiment']
    
    print("[+] Mengekstraksi Fitur Teks dengan TF-IDF Vectorizer (max_features=5000)...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_tfidf = vectorizer.fit_transform(X)
    print(f"[✓] Bentuk Matrix TF-IDF (Baris x Fitur/Kata): {X_tfidf.shape}")
    
    # ------------------ 🔢 TAMPILKAN 3 SAMPEL TF-IDF ------------------
    print("\n--- 🔍 [OUTPUT] Sampel Fitur Teks Dipetakan ke Nilai Angka TF-IDF (3 Data) ---")
    feature_names = vectorizer.get_feature_names_out()
    
    for i in range(min(3, len(df))):
        sample_text = df['cleaned_content'].iloc[i]
        
        # Mengambil baris vektor sparse untuk data ke-i
        row_vector = X_tfidf[i]
        
        # Mencari indeks kata/kolom yang nilainya > 0
        non_zero_indices = row_vector.nonzero()[1]
        
        # Mengambil pasangan (kata, nilai_tfidf)
        word_scores = [(feature_names[col], row_vector[0, col]) for col in non_zero_indices]
        # Mengurutkan berdasarkan skor TF-IDF tertinggi
        word_scores = sorted(word_scores, key=lambda x: x[1], reverse=True)
        
        print(f"  [Sampel {i+1}] Teks Bersih : \"{sample_text}\"")
        print("   ├─ Konversi Angka (TF-IDF Weights):")
        if not word_scores:
            print("   │  (Tidak ada kata yang masuk dalam kamus TF-IDF)")
        else:
            for word, score in word_scores:
                print(f"   │   • Kata: {word:<15} ──► Nilai TF-IDF: {score:.4f}")
        print("   " + "─"*55)
    print("\n")

    # Train-Test Split (70:30)
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.3, random_state=42)
    print(f"[+] Pembagian Data Split -> Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,} samples")
    
    # ------------------ SVM ------------------
    print("\n[1/2] Melatih Model Support Vector Machine (SVM)...")
    svm_model = SVC(kernel='linear', random_state=42)
    svm_model.fit(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)
    acc_svm = accuracy_score(y_test, y_pred_svm)
    
    print("\n--- 🤖 [OUTPUT] Laporan Evaluasi SVM ---")
    print(f"  • Akurasi SVM: {acc_svm * 100:.2f}%")
    print(classification_report(y_test, y_pred_svm))
    
    # ------------------ Random Forest ------------------
    print("[2/2] Melatih Model Random Forest Classifier...")
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, class_weight='balanced')
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    
    print("\n--- 🌲 [OUTPUT] Laporan Evaluasi Random Forest ---")
    print(f"  • Akurasi Random Forest: {acc_rf * 100:.2f}%")
    print(classification_report(y_test, y_pred_rf))
    
    # Ringkasan Komparasi
    print("--- 🏆 [OUTPUT] RINGKASAN KOMPARASI AKURASI MODEL ---")
    print(f"  • Support Vector Machine (SVM) : {acc_svm * 100:.2f}%")
    print(f"  • Random Forest Classifier      : {acc_rf * 100:.2f}%")
    best = "SVM" if acc_svm >= acc_rf else "Random Forest"
    print(f"  ⭐ Model Terbaik Dipilih untuk Bisnis: {best}\n")
    
    return svm_model, X_tfidf



# =====================================================================
# 4. FUNGSI ANALISIS BISNIS (CLV & CHURN DETECTION)
# =====================================================================
def export_10_churn_samples(df, output_csv="sampel_10_churn.csv"):
    # Filter data yang mengalami kejadian churn
    churn_df = df[df['is_churn_event'] == True].copy()
    
    # Pilih kolom penting yang ingin ditampilkan
    cols = ['userName', 'at', 'prev_sentiment', 'pred_sentiment', 'is_churn_event', 'potential_loss']
    cols_exist = [c for c in cols if c in churn_df.columns]
    
    # Ambil 10 sampel data pertama
    churn_10_samples = churn_df[cols_exist].head(10)
    
    print("\n--- 🔍 [OUTPUT] 10 Sampel User Terdeteksi Churn Event ---")
    if not churn_10_samples.empty:
        # Format tampilan rupiah agar lebih rapi di konsol/terminal
        display_df = churn_10_samples.copy()
        if 'potential_loss' in display_df.columns:
            display_df['potential_loss'] = display_df['potential_loss'].apply(lambda x: f"Rp {x:,.0f}")
            
        print(display_df.to_string(index=False))
        
        # Simpan 10 sampel tersebut ke file CSV tersendiri jika diperlukan
        churn_10_samples.to_csv(output_csv, index=False)
        print(f"\n[✓] 10 sampel data churn berhasil disimpan ke file: '{output_csv}'")
    else:
        print("  Tidak ada kejadian pergeseran sentimen ke negatif pada dataset ini.")
        
    return churn_10_samples
def process_business_analysis(df, model, X_tfidf):
    print("==================================================")
    print(" 📌 TAHAP 4: ANALISIS BISNIS (CLV & POTENTIAL LOSS)")
    print("==================================================")
    
    print("[+] Melakukan prediksi sentimen pada seluruh dataset menggunakan model SVM...")
    df['pred_sentiment'] = model.predict(X_tfidf)
    
    if 'at' in df.columns and 'userName' in df.columns:
        df['at_dt'] = pd.to_datetime(df['at'], errors='coerce')
        df = df.sort_values(by=['userName', 'at_dt'])
        
        # Hitung Lifespan & CLV
        def get_lifespan(sentiment):
            if sentiment == 'positive': return 12
            elif sentiment == 'neutral': return 6
            else: return 1
            
        df['lifespan'] = df['pred_sentiment'].apply(get_lifespan)
        df['clv'] = df['lifespan'] * 120000
        
        # Deteksi Churn Event
        df['prev_sentiment'] = df.groupby('userName')['pred_sentiment'].shift(1)
        df['is_churn_event'] = ((df['prev_sentiment'].isin(['positive', 'neutral'])) & (df['pred_sentiment'] == 'negative'))
        df['potential_loss'] = df['is_churn_event'].apply(lambda x: 1200000 if x else 0)
        
        total_loss = df['potential_loss'].sum()
        churn_count = df['is_churn_event'].sum()
        avg_clv = df['clv'].mean()
        
        print("\n--- 💼 [OUTPUT] RINGKASAN METRIK BISNIS & CLV ---")
        print(f"  • Rata-rata Nilai CLV per Pengguna : Rp {avg_clv:,.2f}")
        print(f"  • Jumlah Kejadian Churn Terdeteksi : {churn_count:,} pengguna")
        print(f"  • TOTAL POTENSI KERUGIAN FINANSIAL  : Rp {total_loss:,.0f}")
        
        # -------------------------------------------------------------
        # EKSTRAKSI & TAMPILKAN 10 SAMPEL USER CHURN
        # -------------------------------------------------------------
        churn_df = df[df['is_churn_event'] == True].copy()
        cols = ['userName', 'at', 'prev_sentiment', 'pred_sentiment', 'is_churn_event', 'potential_loss']
        cols_exist = [c for c in cols if c in churn_df.columns]
        
        churn_10_samples = churn_df[cols_exist].head(10)
        
        print("\n--- 🔍 [OUTPUT] Sampel 10 User Terdeteksi Churn Event ---")
        if not churn_10_samples.empty:
            print(churn_10_samples.to_string(index=False))
            # Opsi opsional: simpan 10 sampel ke CSV terpisah
            churn_10_samples.to_csv("sampel_10_churn_users.csv", index=False)
            print("\n[✓] File 'sampel_10_churn_users.csv' berhasil dibuat.")
        else:
            print("  Tidak ada kejadian pergeseran sentimen ke negatif pada dataset ini.")
            
    print("\n==================================================")
    print(" 🎉 SELURUH PIPELINE PYTHON BERHASIL DIJALANKAN!")
    print("==================================================")
    return df

# =====================================================================
# MAIN PIPELINE EXECUTION
# =====================================================================
def run_pipeline():
    input_file = "../netflix_reviews.csv"
    output_cleaned = "netflix_cleaned_python.csv"
    output_labeled = "netflix_labeled_python.csv"
    output_clv = "netflix_clv_python.csv"
    
    if not os.path.exists(input_file):
        print(f"❌ Error: File '{input_file}' tidak ditemukan! Pastikan path file csv kamu sudah benar.")
        return
        
    # Read raw data
    df_raw = pd.read_csv(input_file)
    
    # 1. Preprocessing
    df_cleaned = process_preprocessing(df_raw)
    df_cleaned.to_csv(output_cleaned, index=False)
    
    # 2. Labeling
    df_labeled = process_labeling(df_cleaned)
    df_labeled.to_csv(output_labeled, index=False)
    
    # 3. Machine Learning
    best_model, X_tfidf = process_machine_learning(df_labeled)
    
    # 4. Business Analysis
    df_final = process_business_analysis(df_labeled, best_model, X_tfidf)
    df_final.to_csv(output_clv, index=False)

if __name__ == "__main__":
    run_pipeline()