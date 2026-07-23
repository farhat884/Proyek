import pandas as pd
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os
from tqdm import tqdm

# Download resource NLTK yang diperlukan
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')


# Tqdm untuk progress bar
tqdm.pandas()

# 1. Kamus Normalisasi (Singkatan & Slang Bahasa Inggris)
# Karena dataset sudah di-clean (simbol dihapus), kita sesuaikan tanpa apostrof
norm_dict = {
    # Contractions (without apostrophes)
    "don t": "do not", "can t": "cannot", "isn t": "is not", "it s": "it is",
    "i m": "i am", "won t": "will not", "didn t": "did not", "doesn t": "does not",
    "aren t": "are not", "weren t": "were not", "hasn t": "has not", "haven t": "have not",
    "hadn t": "had not", "shouldn t": "should not", "wouldn t": "would not", "couldn t": "could not",
    "you re": "you are", "they re": "they are", "we re": "we are", "i ve": "i have",
    "you ve": "you have", "we ve": "we have", "they ve": "they have", "i ll": "i will",
    "you ll": "you will", "he ll": "he will", "she ll": "she will", "it ll": "it will",
    "we ll": "we will", "they ll": "they will", "there s": "there is", "that s": "that is",
    "what s": "what is", "where s": "where is", "how s": "how is", "let s": "let us",
    "i d": "i would", "you d": "you would", "he d": "he would", "she d": "she would",
    "we d": "we would", "they d": "they would",
    
    # Common Netflix/App Slang & Abbreviations
    "u": "you", "r": "are", "n": "and", "y": "why", "b": "be", "c": "see", "k": "ok",
    "pls": "please", "plz": "please", "thx": "thanks", "thks": "thanks", "tks": "thanks",
    "ty": "thank you", "omg": "oh my god", "tv": "television", "idk": "i do not know",
    "btw": "by the way", "fyi": "for your information", "asap": "as soon as possible",
    "bc": "because", "cuz": "because", "cos": "because", "coz": "because",
    "wanna": "want to", "gonna": "going to", "gotta": "got to", "imma": "i am going to",
    "lol": "laughing out loud", "lmao": "laughing my ass off", "rofl": "rolling on the floor laughing",
    "sub": "subtitle", "subs": "subtitles", "ep": "episode", "eps": "episodes",
    "msg": "message", "pic": "picture", "vids": "videos", "vid": "video",
    "fav": "favorite", "fave": "favorite", "ads": "advertisements", "ad": "advertisement",
    "info": "information", "app": "application", "apps": "applications",
    "movi": "movie", "movis": "movies", "vidio": "video", "phn": "phone",
    "netflx": "netflix", "netfix": "netflix", "ntflix": "netflix"
}

# Membuat regex pattern untuk normalisasi yang lebih "ketat"
# urutkan kunci berdasarkan panjang (descending) untuk menghindari partial matching
# (Misal 'don t' harus di-check sebelum 'don')
sorted_keys = sorted(norm_dict.keys(), key=len, reverse=True)
pattern = re.compile(r'\b(' + '|'.join(re.escape(k) for k in sorted_keys) + r')\b')

def normalize_text(text):
    if not isinstance(text, str):
        return ""
    # Menggunakan regex untuk mengganti kata yang sesuai di dalam kamus
    return pattern.sub(lambda x: norm_dict[x.group()], text)

# 2. Inisialisasi Stopwords (Bahasa Inggris)
stop_words = set(stopwords.words('english'))
# Kecualikan kata-kata negasi penting agar sentimen tidak berbalik arah
negation_words = {'not', 'no', 'nor', 'neither', 'never', 'none'}
stop_words = stop_words - negation_words

# 3. Inisialisasi Stemmer (Porter Stemmer untuk Bahasa Inggris)
stemmer = PorterStemmer()

def preprocess_steps(text):
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Normalisasi
    text = normalize_text(text)
    
    # Tokenizing
    tokens = word_tokenize(text)
    
    # Stopword Removal & Filtering (Hapus kata pendek < 3 karakter kecuali kata negasi)
    tokens = [word for word in tokens if word not in stop_words and (len(word) > 2 or word in negation_words)]
    
    # Stemming
    tokens = [stemmer.stem(word) for word in tokens]
    
    return " ".join(tokens)

def main():
    # Input dari file yang sudah di-clean sebelumnya
    input_file = "netflix_reviews_cleaned.csv"
    # Output file baru sesuai permintaan
    output_file = "netflix_reviews_preprocessed.csv"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} tidak ditemukan!")
        return

    print(f"Membaca {input_file}...")
    df = pd.read_csv(input_file)
    
    print("Memulai Preprocessing (English):")
    print("- Normalisasi")
    print("- Tokenisasi")
    print("- Stopword Removal")
    print("- Stemming (Porter Stemmer)")
    
    # Menjalankan proses pada kolom 'content'
    df['content'] = df['content'].progress_apply(preprocess_steps)
    
    # Hapus baris yang mungkin kosong setelah preprocessing
    df = df.dropna(subset=['content'])
    df = df[df['content'].str.strip() != ""]
    
    print(f"Menyimpan hasil ke {output_file}...")
    df.to_csv(output_file, index=False)
    
    print("-" * 30)
    print("PROSES SELESAI!")
    print(f"File: {output_file}")
    print("-" * 30)
    print("\nContoh Hasil (5 Data Teratas):")
    print(df[['content']].head())

if __name__ == "__main__":
    main()
