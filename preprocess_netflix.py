import pandas as pd
import re
import os
import emoji

def clean_text(text):
    """
    Fungsi untuk membersihkan teks review sesuai permintaan:
    - Menghapus URL
    - Menghapus Mention (@)
    - Menghapus Hashtag (#)
    - Menghapus Emoji
    - Menghapus karakter non-alfanumerik
    - Menghapus spasi berlebihan
    """
    if not isinstance(text, str):
        return ""
    
    # Menghapus URL
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    # Menghapus Mention (@username)
    text = re.sub(r'@\S+', '', text)
    # Menghapus Hashtag (#hashtag)
    text = re.sub(r'#\S+', '', text)
    # Menghapus Emoji
    text = emoji.replace_emoji(text, replace='')
    # Menghapus karakter non-alfanumerik (hanya menyisakan huruf, angka, dan spasi)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    # Menghapus spasi yang berlebihan
    text = " ".join(text.split())
    
    return text

def preprocess_dataset(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Error: File '{input_path}' tidak ditemukan.")
        return

    print(f"Memuat data dari {input_path}...")
    try:
        # Membaca seluruh data
        df = pd.read_csv(input_path)
        
        # Filter data dari tahun 2021 hingga 2026
        if 'at' in df.columns:
            print("Memfilter data dari tahun 2021 hingga 2026...")
            df['at'] = pd.to_datetime(df['at'], errors='coerce')
            df = df[(df['at'].dt.year >= 2021) & (df['at'].dt.year <= 2026)]
            
        # Tidak ada batasan jumlah data, menggunakan seluruh data 2021-2026
    except Exception as e:
        print(f"Gagal membaca CSV: {e}")
        return
    
    # Memastikan kolom 'content' ada
    if 'content' not in df.columns:
        print(f"Error: Kolom 'content' tidak ditemukan. Kolom yang tersedia: {df.columns.tolist()}")
        return

    # Drop isnull (pada kolom content)
    print("Menghapus data kosong (isnull)...")
    df = df.dropna(subset=['content'])
    
    # Melakukan cleaning pada kolom content
    print("Melakukan cleaning data (URL, Mention, Hashtag, Emoji, Non-Alfanumerik, Spasi)...")
    df['content'] = df['content'].apply(clean_text)
    
    # Menghapus baris yang menjadi kosong setelah cleaning
    df = df[df['content'].str.strip() != ""]
    
    # Menyimpan file csv baru 'netflix_reviews_cleaned.csv'
    print(f"Menyimpan data hasil cleaning ke {output_path}...")
    df.to_csv(output_path, index=False)
    
    print("-" * 30)
    print("Proses Selesai!")
    print(f"Total data setelah cleaning: {len(df)}")
    print("-" * 30)

if __name__ == "__main__":
    # Path disesuaikan dengan struktur folder
    INPUT_FILE = "netflix_reviews.csv"
    OUTPUT_FILE = "netflix_reviews_cleaned.csv"
    
    preprocess_dataset(INPUT_FILE, OUTPUT_FILE)
