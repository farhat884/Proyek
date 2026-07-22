import re
import pandas as pd
import time
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

# Memastikan UTF-8 agar ikon/garis terminal tidak error di Windows CMD
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# =====================================================================
# 1. FUNGSI TEKNIS PREPROCESSING & NLP
# =====================================================================
def case_folding(text):
    return str(text).lower()

def normalization(text):
    norm_dict = {
        "can't": "can not", "i'm": "i am", "don't": "do not",
        "won't": "will not", "it's": "it is", "that's": "that is"
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
    if score >= 4: return "Positive"
    elif score <= 2: return "Negative"
    else: return "Neutral"

def clv_lifespan(sentiment):
    if sentiment.lower() == "positive": return 12
    elif sentiment.lower() == "neutral": return 6
    else: return 1

def clv_loss(lifespan):
    monthly_fee = 120000
    loss_months = 12 - lifespan
    return loss_months * monthly_fee

# =====================================================================
# 2. HELPER UNTUK TAMPILAN OUTPUT TERMINAL YANG RAPI
# =====================================================================
def print_section_header(title):
    print("\n" + "═"*75)
    print(f"  📌 {title.upper()}")
    print("═"*75)

def print_nlp_samples(df, col_before, col_after, sample_size=3):
    """Menampilkan perbandingan Sebelum vs Sesudah secara vertikal agar tidak berantakan"""
    for i, (_, row) in enumerate(df.head(sample_size).iterrows(), 1):
        before = str(row[col_before])
        after = str(row[col_after])
        # Potong teks jika terlalu panjang untuk tampilan konsol
        if len(before) > 80: before = before[:77] + "..."
        if len(after) > 80: after = after[:77] + "..."
        
        print(f"  [Sampel {i}]")
        print(f"   ├─ Sebelum : {before}")
        print(f"   └─ Sesudah : {after}")
        print("  " + "─"*71)

# =====================================================================
# 3. PIPELINE UTAMA
# =====================================================================
if __name__ == "__main__":
    start_time = time.time()
    
    print("\n🚀 MEMULAI PIPELINE NLP & MACHINE LEARNING NETFLIX...")
    print("-----------------------------------------------------")
    
    try:
        df = pd.read_csv("netflix_reviews.csv")
        print(f"  [✓] Sukses membaca file 'netflix_reviews.csv'")
    except Exception as e:
        print(f"  [❌] Error membaca file: {e}")
        sys.exit()
        
    df['content'] = df['content'].fillna('')
    print(f"  [✓] Total data dimuat: {len(df):,} baris.")
    
    # Labeling Awal
    if 'score' in df.columns:
        df['True Sentiment'] = df['score'].apply(map_score_to_sentiment)
    else:
        print("  [❌] Kolom 'score' tidak ditemukan!")
        sys.exit()

    # --- 1. Case Folding ---
    df['1. Case Folding'] = df['content'].apply(case_folding)
    print_section_header("1. Case Folding (Konversi Ke Huruf Kecil)")
    print_nlp_samples(df, 'content', '1. Case Folding')

    # --- 2. Normalization ---
    df['2. Normalization'] = df['1. Case Folding'].apply(normalization)
    print_section_header("2. Normalization (Perbaikan Singkatan)")
    print_nlp_samples(df, '1. Case Folding', '2. Normalization')

    # --- 3. Remove Noise ---
    df['3. Remove Noise'] = df['2. Normalization'].apply(remove_noise)
    print_section_header("3. Remove Noise (Pembersihan Simbol & Tanda Baca)")
    print_nlp_samples(df, '2. Normalization', '3. Remove Noise')

    # --- 4. Tokenization ---
    df['4. Tokenization (List)'] = df['3. Remove Noise'].apply(tokenization)
    df['4. Tokenization'] = df['4. Tokenization (List)'].apply(str) 
    print_section_header("4. Tokenization (Pemotongan Kata)")
    print_nlp_samples(df, '3. Remove Noise', '4. Tokenization')

    # --- 5. Stopword Removal ---
    df['5. Stopword Removal (List)'] = df['4. Tokenization (List)'].apply(remove_stopwords)
    df['5. Stopword Removal'] = df['5. Stopword Removal (List)'].apply(str)
    print_section_header("5. Stopword Removal (Pebuangan Kata Umum)")
    print_nlp_samples(df, '4. Tokenization', '5. Stopword Removal')

    # --- 6. Stemming ---
    df['6. Stemming (List)'] = df['5. Stopword Removal (List)'].apply(stemming)
    df['6. Stemming'] = df['6. Stemming (List)'].apply(lambda x: ' '.join(x)) 
    print_section_header("6. Stemming (Pengembalian Kata Dasar)")
    print_nlp_samples(df, '5. Stopword Removal', '6. Stemming')

    # --- 7. TF-IDF ---
    print_section_header("7. Ekstraksi Fitur TF-IDF")
    tfidf_vectorizer = TfidfVectorizer(max_features=1000)
    X_tfidf = tfidf_vectorizer.fit_transform(df['6. Stemming'])
    y = df['True Sentiment']
    
    print(f"  • Matriks Dimensi Fitur (Baris x Fitur) : {X_tfidf.shape}")
    sample_features = tfidf_vectorizer.get_feature_names_out()[:10]
    print(f"  • Cuplikan 10 Kata Fitur Pertama      : {', '.join(sample_features)}")

    # --- 8. Machine Learning: Random Forest ---
    print_section_header("8. Permodelan Machine Learning: Random Forest")
    rf_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf_model.fit(X_tfidf, y)
    df['8. RF Prediction'] = rf_model.predict(X_tfidf)
    
    print("  Cuplikan Hasil Prediksi Model Random Forest (5 Data Teratas):")
    print("  ┌──────┬─────────────────┬──────────────────┐")
    print("  │ No.  │ True Sentiment  │ Prediksi RF      │")
    print("  ├──────┼─────────────────┼──────────────────┤")
    for idx, row in df.head(5).iterrows():
        print(f"  │ {idx+1:<4} │ {row['True Sentiment']:<15} │ {row['8. RF Prediction']:<16} │")
    print("  └──────┴─────────────────┴──────────────────┘")

    # --- 9. Machine Learning: SVM ---
    print_section_header("9. Permodelan Machine Learning: SVM (LinearSVC)")
    svm_model = LinearSVC(random_state=42)
    svm_model.fit(X_tfidf, y)
    df['9. SVM Prediction'] = svm_model.predict(X_tfidf)
    
    print("  Cuplikan Hasil Prediksi Model SVM (5 Data Teratas):")
    print("  ┌──────┬─────────────────┬──────────────────┐")
    print("  │ No.  │ True Sentiment  │ Prediksi SVM     │")
    print("  ├──────┼─────────────────┼──────────────────┤")
    for idx, row in df.head(5).iterrows():
        print(f"  │ {idx+1:<4} │ {row['True Sentiment']:<15} │ {row['9. SVM Prediction']:<16} │")
    print("  └──────┴─────────────────┴──────────────────┘")

    # --- 10 & 11. Analisis Bisnis (CLV & Loss) ---
    print_section_header("10 & 11. Kalkulasi Bisnis (CLV & Potential Loss)")
    df['10. CLV Lifespan (Bulan)'] = df['8. RF Prediction'].apply(clv_lifespan)
    df['11. Potential Loss (Rp)'] = df['10. CLV Lifespan (Bulan)'].apply(clv_loss)
    
    print("  Cuplikan Estimasi Umur & Potensi Kerugian (Berdasarkan Prediksi RF):")
    print("  ┌──────┬──────────────────┬──────────────────┬──────────────────────┐")
    print("  │ No.  │ Prediksi RF      │ Lifespan (Bulan) │ Potential Loss (Rp)  │")
    print("  ├──────┼──────────────────┼──────────────────┼──────────────────────┤")
    for idx, row in df.head(5).iterrows():
        loss_formatted = f"Rp {row['11. Potential Loss (Rp)']:,.0f}"
        print(f"  │ {idx+1:<4} │ {row['8. RF Prediction']:<16} │ {row['10. CLV Lifespan (Bulan)']:<16} │ {loss_formatted:<20} │")
    print("  └──────┴──────────────────┴──────────────────┴──────────────────────┘")

    # Membersihkan kolom array temporary sebelum ekspor
    df = df.drop(columns=['4. Tokenization (List)', '5. Stopword Removal (List)', '6. Stemming (List)'])
    
    # Ekspor Ke Excel
    excel_filename = "hasil_analisis_netflix_ml.xlsx"
    print_section_header("Penyimpanan File Ekspor")
    print(f"  [⏳] Memproses penyimpanan {len(df):,} baris data ke file Excel...")
    df.to_excel(excel_filename, index=False)
    print(f"  [✓] File berhasil disimpan sebagai '{excel_filename}'!")

    end_time = time.time()
    
    # --- RINGKASAN AKHIR ---
    print("\n" + "█"*75)
    print("  🎉 SELURUH PIPELINE SELESAI DIJALANKAN!")
    print(f"  ⏱️  Total Waktu Eksekusi : {end_time - start_time:.2f} Detik")
    print(f"  📊 Total Potensi Loss    : Rp {df['11. Potential Loss (Rp)'].sum():,.0f}")
    print("█"*75 + "\n")