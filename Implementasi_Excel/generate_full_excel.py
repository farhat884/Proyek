import pandas as pd
import xlsxwriter
import os
import re
from collections import Counter

def extract_top_words(df, n=20):
    text = " ".join(df['content'].astype(str).tolist()).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = [w for w in text.split() if w not in ['the', 'is', 'in', 'to', 'and', 'it', 'for', 'of', 'this', 'that', 'on', 'with', 'as', 'are', 'was', 'not', 'but', 'have', 'be', 'you', 'app', 'netflix', 'i', 'a', 'my']]
    return [word for word, count in Counter(words).most_common(n)]

def generate_excel():
    input_file = "../netflix_reviews.csv"
    print(f"Membaca {input_file}...")
    df = pd.read_csv(input_file)
    
    # Sort data by userName and date to allow sequential churn detection via formula
    if 'at' in df.columns and 'userName' in df.columns:
        df['at_dt'] = pd.to_datetime(df['at'], errors='coerce')
        df = df.sort_values(by=['userName', 'at_dt']).drop(columns=['at_dt'])
    
    total_rows = len(df)
    print(f"Total baris yang akan diproses: {total_rows}")
    
    top_words = extract_top_words(df.head(10000), 10) 
    
    output_file = "Netflix_Full_Process_V2.xlsx"
    workbook = xlsxwriter.Workbook(output_file, {'use_zip64': True})
    
    # --- Sheet 1: Data Mentah ---
    ws1 = workbook.add_worksheet('1. Data Mentah')
    headers1 = list(df.columns)
    for col_num, header in enumerate(headers1):
        ws1.write(0, col_num, header)
        
    for row_num, row_data in enumerate(df.values):
        for col_num, cell_data in enumerate(row_data):
            if pd.isna(cell_data):
                cell_data = ""
            ws1.write(row_num + 1, col_num, cell_data)
            
    print("Sheet 1 selesai.")
    
    # --- Sheet 2: Preprocessing & Labeling ---
    ws2 = workbook.add_worksheet('2. Preprocess & Labeling')
    headers2 = ['ID (Baris)', 'Raw Content', 'Score', 'Cleaned Text', 'Label Sentimen']
    for col_num, header in enumerate(headers2):
        ws2.write(0, col_num, header)
        
    content_col_idx = headers1.index('content')
    score_col_idx = headers1.index('score')
    
    for row_num in range(1, total_rows + 1):
        ws2.write_number(row_num, 0, row_num)
        ws2.write_formula(row_num, 1, f"='1. Data Mentah'!{chr(65+content_col_idx)}{row_num+1}")
        ws2.write_formula(row_num, 2, f"='1. Data Mentah'!{chr(65+score_col_idx)}{row_num+1}")
        ws2.write_formula(row_num, 3, f"=TRIM(LOWER(B{row_num+1}))")
        ws2.write_formula(row_num, 4, f'=IF(C{row_num+1}>=4, "positive", IF(C{row_num+1}<=2, "negative", "neutral"))')
        
    print("Sheet 2 selesai.")
    
    # --- Sheet 3: TF-IDF & SVM ---
    ws3 = workbook.add_worksheet('3. Machine Learning')
    ws3.write(0, 0, "ID")
    ws3.write(0, 1, "Cleaned Text")
    
    for i, word in enumerate(top_words):
        col_idx = 2 + i
        # Tulis rumus IDF global di header untuk efisiensi eksekusi Excel
        idf_global_formula = f"=LN({total_rows} / MAX(1, COUNTIF('2. Preprocess & Labeling'!$D$2:$D${total_rows+1}, \"*{word}*\")))"
        ws3.write_formula(0, col_idx, idf_global_formula)
        
    ws3.write(0, 2 + len(top_words), "SVM Score")
    ws3.write(0, 3 + len(top_words), "Prediksi Sentimen")
    
    for row_num in range(1, total_rows + 1):
        ws3.write_formula(row_num, 0, f"='2. Preprocess & Labeling'!A{row_num+1}")
        ws3.write_formula(row_num, 1, f"='2. Preprocess & Labeling'!D{row_num+1}")
        
        tf_idf_cells = []
        for i, word in enumerate(top_words):
            col_letter = chr(65 + 2 + i)
            tf_formula = f"((LEN(B{row_num+1}) - LEN(SUBSTITUTE(B{row_num+1}, \"{word}\", \"\"))) / MAX(1, LEN(\"{word}\")))"
            # Referensikan IDF yang ada di header (baris 1, absolut)
            ws3.write_formula(row_num, 2 + i, f"={tf_formula} * {col_letter}$1")
            tf_idf_cells.append(f"{col_letter}{row_num+1}")
            
        svm_col = chr(65 + 2 + len(top_words))
        pred_col = chr(65 + 3 + len(top_words))
        weights = [round(0.1 * (i+1), 2) for i in range(len(top_words))]
        sumproduct_args = "+".join([f"({cell}*{w})" for cell, w in zip(tf_idf_cells, weights)])
        ws3.write_formula(row_num, 2 + len(top_words), f"={sumproduct_args} + 0.5")
        ws3.write_formula(row_num, 3 + len(top_words), f'=IF({svm_col}{row_num+1} > 1, "positive", "negative")')
        
    print("Sheet 3 selesai.")
    
    # --- Sheet 4: CLV & Churn ---
    ws4 = workbook.add_worksheet('4. Analisis CLV & Churn')
    headers4 = ['ID', 'Username', 'Date', 'Prediksi Sentimen', 'Lifespan (Bulan)', 'CLV (Rp)', 'Churn Event', 'Potential Loss (Rp)']
    for col_num, header in enumerate(headers4):
        ws4.write(0, col_num, header)
        
    # Get indices for username and date from raw data headers
    username_idx = headers1.index('userName')
    date_idx = headers1.index('at')
    
    pred_sentimen_col = chr(65 + 3 + len(top_words))
    
    for row_num in range(1, total_rows + 1):
        ws4.write_formula(row_num, 0, f"='3. Machine Learning'!A{row_num+1}")
        ws4.write_formula(row_num, 1, f"='1. Data Mentah'!{chr(65+username_idx)}{row_num+1}")
        ws4.write_formula(row_num, 2, f"='1. Data Mentah'!{chr(65+date_idx)}{row_num+1}")
        ws4.write_formula(row_num, 3, f"='3. Machine Learning'!{pred_sentimen_col}{row_num+1}")
        
        # Lifespan
        ws4.write_formula(row_num, 4, f'=IF(D{row_num+1}="positive", 12, IF(D{row_num+1}="neutral", 6, 1))')
        # CLV (Lifespan * 120.000)
        ws4.write_formula(row_num, 5, f'=E{row_num+1} * 120000')
        
        if row_num == 1:
            ws4.write(row_num, 6, "No")
            ws4.write(row_num, 7, 0)
        else:
            # Churn Event: Same user, prev positive/neutral, current negative
            ws4.write_formula(row_num, 6, f'=IF(AND(B{row_num+1}=B{row_num}, OR(D{row_num}="positive", D{row_num}="neutral"), D{row_num+1}="negative"), "Yes", "No")')
            # Potential Loss
            ws4.write_formula(row_num, 7, f'=IF(G{row_num+1}="Yes", 1200000, 0)')
            
    print("Sheet 4 selesai.")
    workbook.close()
    print(f"File {output_file} berhasil dibuat!")

if __name__ == "__main__":
    generate_excel()
