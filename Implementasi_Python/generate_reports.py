import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
import numpy as np
import warnings
import sys
warnings.filterwarnings('ignore')

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Konfigurasi plot
plt.style.use('ggplot')
sns.set_palette("Set2")

print("1. Membaca data hasil analisis dari Excel (mohon tunggu)...")
df = pd.read_excel("hasil_analisis_netflix_ml.xlsx")
df = df.dropna(subset=['6. Stemming', 'True Sentiment'])

# Fitur dan Label
X_text = df['6. Stemming'].astype(str)
y = df['True Sentiment']

# TRAIN-TEST SPLIT (80% Train, 20% Test)
print("2. Splitting Data (80% Train, 20% Test)...")
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, random_state=42, stratify=y
)

print("3. Melatih Model Random Forest (Pipeline + GridSearchCV + N-Grams)...")
print("Ini akan memakan waktu karena mesin mencari parameter terbaik dari banyak kombinasi!")

# Setup Pipeline persis seperti di Folder 1
rf_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 3))),
    ('clf', RandomForestClassifier(class_weight='balanced', random_state=42))
])

# Setup GridSearch params
rf_param_grid = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [10, 20, 30],
    'clf__min_samples_split': [2, 5]
}

# Run GridSearch
rf_grid = GridSearchCV(rf_pipeline, rf_param_grid, cv=3, n_jobs=-1, verbose=1)
rf_grid.fit(X_train_text, y_train)

best_rf = rf_grid.best_estimator_
print(f"\nParameter terbaik yang ditemukan: {rf_grid.best_params_}")

print("4. Melakukan Prediksi pada Data Uji (Test Set 20%)...")
y_pred = best_rf.predict(X_test_text)
y_prob = best_rf.predict_proba(X_test_text)

# ==========================================
# 1. Classification Report & Confusion Matrix
# ==========================================
print("\n--- CLASSIFICATION REPORT (Random Forest GridSearchCV - Test Set) ---")
report = classification_report(y_test, y_pred)
print(report)

print("\n--- CONFUSION MATRIX (Test Set) ---")
cm = confusion_matrix(y_test, y_pred, labels=["Negative", "Neutral", "Positive"])
print(pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted']))

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Negative", "Neutral", "Positive"],
            yticklabels=["Negative", "Neutral", "Positive"])
plt.title('Confusion Matrix - RF GridSearch (Test Set)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix_test_gridsearch.png')
plt.close()

# ==========================================
# 2. ROC-AUC Score
# ==========================================
classes = best_rf.classes_
y_test_bin = label_binarize(y_test, classes=classes)
roc_auc = roc_auc_score(y_test_bin, y_prob, multi_class='ovr')
print(f"\n>>> ROC-AUC Score (OVR) pada Ujian Akhir: {roc_auc:.4f} <<<\n")


# ==========================================
# 3. Top 10 Words Chart (menggunakan N-Grams)
# ==========================================
print("Membuat Chart Top 10 Frasa per Sentimen...")
def plot_top_words(sentiment_label, color):
    text_data = df[df['True Sentiment'] == sentiment_label]['6. Stemming'].astype(str)
    if len(text_data) == 0: return
    # Menggunakan N-Gram 1-3 agar selaras dengan pendekatan baru
    vec = CountVectorizer(max_features=10, ngram_range=(1,3)).fit(text_data)
    bag_of_words = vec.transform(text_data)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=False)
    
    print(f"\nTop 10 Frasa [{sentiment_label}]:")
    for w, c in sorted(words_freq, key=lambda x: x[1], reverse=True):
        print(f" - {w}: {c}")
        
    words, counts = zip(*words_freq)
    plt.figure(figsize=(8, 5))
    plt.barh(words, counts, color=color)
    plt.title(f'Top 10 Frasa untuk Sentimen: {sentiment_label}')
    plt.xlabel('Frekuensi')
    plt.tight_layout()
    plt.savefig(f'top10_{sentiment_label.lower()}_ngrams.png')
    plt.close()

plot_top_words("Positive", "mediumseagreen")
plot_top_words("Negative", "tomato")
plot_top_words("Neutral", "steelblue")

print("\nLaporan evaluasi tes independen menggunakan GridSearchCV berhasil diselesaikan!")
