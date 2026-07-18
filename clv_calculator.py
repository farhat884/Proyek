import pandas as pd

def detect_churn_events(df):
    """
    Mendeteksi user yang berubah sentimen dari positif/netral ke negatif 
    beserta teks review sebelumnya dan sekarang.
    """
    df['at'] = pd.to_datetime(df['at'])
    df = df.sort_values(by=['userName', 'at'])
    
    # Ambil sentimen dan teks sebelumnya untuk user yang sama
    df['prev_sentiment'] = df.groupby('userName')['sentiment'].shift(1)
    df['prev_content'] = df.groupby('userName')['content'].shift(1)
    
    # Deteksi Churn Event
    df['is_churn_event'] = (
        (df['prev_sentiment'].isin(['positive', 'neutral'])) & 
        (df['sentiment'] == 'negative')
    )
    
    # Hitung potensi loss
    df['potential_loss'] = df['is_churn_event'].apply(lambda x: 1200000 if x else 0)
    
    return df

def calculate_multi_clv(lifespan):
    """
    Menghitung estimasi CLV berdasarkan 3 skenario paket berlangganan Netflix:
    - Basic (Min): Rp 65.000
    - Standard (Avg): Rp 120.000
    - Premium (Max): Rp 186.000
    """
    # Hitung CLV untuk masing-masing skenario paket
    clv_min = 65000 * lifespan
    clv_avg = 120000 * lifespan
    clv_max = 186000 * lifespan
    
    return pd.Series([clv_min, clv_avg, clv_max])

def apply_clv_to_dataframe(df, sentiment_column='sentiment'):
    df_clv = df.copy()
    
    # 1. Persiapkan datetime, year, dan month
    df_clv['at_dt'] = pd.to_datetime(df_clv['at'], errors='coerce')
    df_clv['year'] = df_clv['at_dt'].dt.year
    df_clv['month'] = df_clv['at_dt'].dt.month
    df_clv['year_month'] = df_clv['at_dt'].dt.to_period('M')
    
    # 2. Pemetaan harga paket berdasarkan sentimen
    def get_price(sentiment):
        if pd.isna(sentiment): return 120000
        s = sentiment.lower()
        if s == 'positive': return 186000
        elif s == 'neutral': return 120000
        elif s == 'negative': return 65000
        return 120000

    df_clv['monthly_package_price'] = df_clv[sentiment_column].apply(get_price)
    
    # 3. Agregasi per user per bulan (ambil data pertama per bulan)
    df_sorted = df_clv.sort_values(by=['userName', 'at_dt'])
    df_monthly = df_sorted.drop_duplicates(subset=['userName', 'year', 'month'], keep='first')
    
    # 4. Hitung CLV per user per tahun
    user_yearly_stats = []
    
    for (user, year), group in df_monthly.groupby(['userName', 'year']):
        months_active = sorted(group['month'].tolist())
        sentiments = group.set_index('month')[sentiment_column].to_dict()
        prices = group.set_index('month')['monthly_package_price'].to_dict()
        
        # Actual spend selama bulan-bulan aktif
        actual_spend = sum(prices.values())
        
        last_month = months_active[-1]
        last_sentiment = sentiments[last_month].lower() if pd.notna(sentiments[last_month]) else 'neutral'
        last_price = prices[last_month]
        
        # Proyeksi untuk sisa bulan dalam tahun tersebut jika sentimen terakhir positif/netral
        projected_spend = 0
        if last_sentiment in ['positive', 'neutral']:
            remaining_months = 12 - last_month
            projected_spend = remaining_months * last_price
            
        total_clv = actual_spend + projected_spend
        
        # Hitung selisih bulan berlangganan untuk menentukan gap
        max_gap = 0
        if len(months_active) > 1:
            gaps = [months_active[i] - months_active[i-1] for i in range(1, len(months_active))]
            max_gap = max(gaps)
            
        user_yearly_stats.append({
            'userName': user,
            'year': year,
            'duration': len(months_active),
            'clv': total_clv,
            'max_gap': max_gap
        })
        
    df_yearly = pd.DataFrame(user_yearly_stats)
    
    if df_yearly.empty:
        df_clv['user_clv'] = 0
        df_clv['loyalty_status'] = 'Rentan Churn (Berhenti)'
        return df_clv
    
    # 5. Menghitung Rata-rata Uang dan Modus Lama Langganan per Tahun
    yearly_avg_clv = df_yearly.groupby('year')['clv'].mean().to_dict()
    
    # Fungsi pembantu untuk mengambil modus (bisa lebih dari satu, kita ambil yang pertama)
    def get_mode(x):
        m = x.mode()
        return m.iloc[0] if not m.empty else 1
        
    yearly_modus_duration = df_yearly.groupby('year')['duration'].apply(get_mode).to_dict()
    
    # Simpan atribut global untuk ditampilkan di dashboard (menggunakan data dari seluruh tahun atau tahun terakhir)
    # Sebagai representasi, kita bisa simpan rata-rata keseluruhan (opsional, tapi dashboard mungkin butuh fallback)
    df_clv.attrs['avg_expenditure_overall'] = int(df_yearly['clv'].mean())
    
    # 6. Menentukan Status Loyalitas
    def determine_loyalty(row):
        y = row['year']
        clv_val = row['clv']
        dur = row['duration']
        gap = row['max_gap']
        
        avg_exp = yearly_avg_clv[y]
        mod_dur = yearly_modus_duration[y]
        
        if clv_val >= avg_exp:
            return "Sangat Loyal"
        elif clv_val >= (avg_exp / 2):
            return "Cukup Loyal"
        else:
            # Skenario Fallback untuk pengguna dengan uang < (avg / 2) atau sentimen negatif
            if dur >= mod_dur:
                # Syarat: selisih (gap) <= 2 ATAU minimal jumlah berlangganan >= 4
                if gap <= 2 or dur >= 4:
                    return "Cukup Loyal"
            return "Rentan Churn (Berhenti)"
            
    df_yearly['loyalty_status'] = df_yearly.apply(determine_loyalty, axis=1)
    
    # 7. Merge kembali ke dataframe asli untuk keperluan analisis/visualisasi per row
    df_clv = pd.merge(df_clv, df_yearly[['userName', 'year', 'clv', 'loyalty_status', 'duration']], on=['userName', 'year'], how='left')
    
    # Ganti nama kolom 'clv' hasil merge menjadi 'user_clv' agar kompatibel dengan baris kode sebelumnya
    df_clv.rename(columns={'clv': 'user_clv'}, inplace=True)
    
    return df_clv