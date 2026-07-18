import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
from PIL import Image
from clv_calculator import apply_clv_to_dataframe
from clv_calculator import detect_churn_events

# Konfigurasi Halaman
st.set_page_config(page_title="Netflix Review Sentiment Analysis", layout="wide")

# Fungsi untuk memuat data
@st.cache_data
def load_data():
    df = pd.read_csv("netflix_reviews_labeled.csv")
    if 'at' in df.columns:
        df['at'] = pd.to_datetime(df['at'], errors='coerce')
        df = df[(df['at'].dt.year >= 2021) & (df['at'].dt.year <= 2026)]
    return df

@st.cache_data
def get_clv_data(df):
    return apply_clv_to_dataframe(df)

@st.cache_data
def get_churn_data(df):
    return detect_churn_events(df)

try:
    df = load_data()
except Exception as e:
    st.error(f"Gagal memuat data: {e}")
    st.stop()

# Sidebar Navigasi
st.sidebar.title("Navigasi")
page = st.sidebar.selectbox("Pilih Halaman", ["Ringkasan Data", "Pencarian Kata", "Visualisasi Model", "Analisis Kata (WordCloud & Top Words)","Analisis CLV & Loyalitas"])

# --- HALAMAN 1: RINGKASAN DATA ---
if page == "Ringkasan Data":
    st.title("📊 Ringkasan Sentimen Netflix Reviews")
    
    # Statistik Utama
    total_data = len(df)
    sentiment_counts = df['sentiment'].value_counts()
    most_frequent = sentiment_counts.idxmax()
    most_frequent_count = sentiment_counts.max()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Data Labeling", f"{total_data} reviews")
    col2.metric("Sentimen Terbanyak", most_frequent.capitalize())
    col3.metric("Jumlah Sentimen Terbanyak", f"{most_frequent_count} reviews")
    
    st.markdown("---")
    
    # Bar Chart Penyebaran Keseluruhan
    st.subheader("📈 Penyebaran Keseluruhan Sentimen")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(data=df, x='sentiment', palette='viridis', ax=ax)
    ax.set_title("Distribusi Label Sentimen")
    ax.set_xlabel("Sentimen")
    ax.set_ylabel("Jumlah")
    st.pyplot(fig)
    
    # Deskripsi
    st.info(f"""
    **Analisis Singkat:**
    Data labeling terdiri dari **{total_data}** ulasan Netflix. 
    Sentimen yang paling mendominasi adalah **{most_frequent}** dengan total **{most_frequent_count}** ulasan. 
    Hal ini menunjukkan kecenderungan pengguna dalam memberikan respon terhadap layanan Netflix di platform.
    """)

# --- HALAMAN 2: PENCARIAN KATA ---
elif page == "Pencarian Kata":
    st.title("🔍 Analisis Kata Kunci")
    st.markdown("Cari kata tertentu untuk melihat bagaimana sentimen pengguna terhadap topik tersebut.")
    
    query = st.text_input("Masukkan kata kunci (contoh: 'price', 'movie', 'bad'):", "")
    
    if query:
        # Filter data berdasarkan query
        filtered_df = df[df['content'].str.contains(query, case=False, na=False)]
        
        if not filtered_df.empty:
            st.success(f"Ditemukan {len(filtered_df)} ulasan mengandung kata '{query}'")
            
            # Distribusi Sentimen untuk Kata Kunci
            st.subheader(f"Penyebaran Sentimen untuk Kata: '{query}'")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.countplot(data=filtered_df, x='sentiment', palette='magma', order=['positive', 'neutral', 'negative'], ax=ax)
            st.pyplot(fig)
            
            # Tampilkan beberapa contoh review
            with st.expander("Lihat contoh ulasan"):
                st.dataframe(filtered_df[['content', 'sentiment']].head(10))
        else:
            st.warning(f"Tidak ditemukan ulasan yang mengandung kata '{query}'.")

# --- HALAMAN 3: VISUALISASI MODEL ---
elif page == "Visualisasi Model":
    st.title("🤖 Hasil Pemodelan (SVM & Random Forest)")
    
    viz_path = "visualizations"
    
    tab1, tab2, tab3 = st.tabs(["Perbandingan Model", "SVM Details", "Random Forest Details"])
    
    with tab1:
        st.subheader("Akurasi & Perbandingan")
        cols_comp = st.columns(2)
        comp_img_acc = os.path.join(viz_path, "model_comparison_accuracy.png")
        comp_img_gen = os.path.join(viz_path, "model_comparison.png")
        
        if os.path.exists(comp_img_acc):
            cols_comp[0].image(comp_img_acc, caption="Perbandingan Akurasi SVM vs RF")
        if os.path.exists(comp_img_gen):
            cols_comp[1].image(comp_img_gen, caption="Perbandingan Metrik Model")
            
    with tab2:
        st.subheader("SVM Performance Metrics")
        cols = st.columns(2)
        svm_files = {
            "Confusion Matrix (Folder)": os.path.join(viz_path, "svm_cm.png"),
            "Confusion Matrix (Root)": "cm_svm.png",
            "ROC Curve": os.path.join(viz_path, "svm_roc.png"),
            "Precision-Recall": os.path.join(viz_path, "svm_pr.png"),
            "Classification Report": os.path.join(viz_path, "svm_report.png"),
            "Feature Importance": os.path.join(viz_path, "svm_features.png")
        }
        
        for i, (label, path) in enumerate(svm_files.items()):
            if os.path.exists(path):
                cols[i % 2].image(path, caption=label)
                
    with tab3:
        st.subheader("Random Forest Performance Metrics")
        cols = st.columns(2)
        rf_files = {
            "Confusion Matrix (Folder)": os.path.join(viz_path, "rf_cm.png"),
            "Confusion Matrix (Root)": "cm_rf.png",
            "ROC Curve": os.path.join(viz_path, "rf_roc.png"),
            "Precision-Recall": os.path.join(viz_path, "rf_pr.png"),
            "Classification Report": os.path.join(viz_path, "rf_report.png"),
            "Feature Importance": os.path.join(viz_path, "rf_features.png")
        }
        
        for i, (label, path) in enumerate(rf_files.items()):
            if os.path.exists(path):
                cols[i % 2].image(path, caption=label)

# --- HALAMAN 4: ANALISIS KATA ---
elif page == "Analisis Kata (WordCloud & Top Words)":
    st.title("☁️ Analisis Kata Terbanyak")
    st.markdown("Visualisasi kata-kata dan sentimen yang paling ekspresif dengan memotong kata-kata umum (generic).")
    
    col1, col2, col3 = st.columns(3)
    
    wc_files = {
        "Positive": ("wordcloud_positive.png", "barchart_positive.png"),
        "Neutral": ("wordcloud_neutral.png", "barchart_neutral.png"),
        "Negative": ("wordcloud_negative.png", "barchart_negative.png")
    }
    
    for i, (label, (wc_file, bar_file)) in enumerate(wc_files.items()):
        current_col = col1 if label == "Positive" else (col2 if label == "Neutral" else col3)
        
        if os.path.exists(wc_file):
            current_col.image(wc_file, caption=f"WordCloud {label}")
        else:
            current_col.write(f"File {wc_file} tidak ditemukan.")
            
        if os.path.exists(bar_file):
            current_col.image(bar_file, caption=f"Top 5 Words {label}")
        else:
            current_col.write(f"File {bar_file} tidak ditemukan.")
            
# (Di dalam blok kode halaman Streamlit kamu...)
elif page == "Analisis CLV & Loyalitas":
    st.title("💎 Pendekatan Behavioral CLV (Analisis Multi-Skenario)")
    st.markdown("Analisis ini menggunakan variasi paket Netflix berdasarkan referensi penelitian terbaru untuk memetakan profit tahunan dan memprediksi retensi pengguna.")
    
    # Proses data
    df_clv = get_clv_data(df)
    
    # Memastikan rentang waktu secara ketat pada 2021-2026
    df_clv = df_clv[(df_clv['year'] >= 2021) & (df_clv['year'] <= 2026)]
    
    # Ambil atribut batas pengeluaran rata-rata keseluruhan (hanya untuk info awal)
    avg_expenditure_overall = df_clv.attrs.get('avg_expenditure_overall', 142000)
    cukup_loyal_threshold_exp = avg_expenditure_overall / 2
    
    # Menentukan paket langganan terpopuler
    package_map = {186000: 'Premium', 120000: 'Standard', 65000: 'Basic'}
    popular_package_price = df_clv['monthly_package_price'].mode()[0]
    popular_package = package_map.get(popular_package_price, 'Tidak Diketahui')
    
    st.info(f"""
    **Informasi Penentuan Loyalitas & CLV (Perhitungan Per Tahun):**
    - **Positif**: Paket Premium (Rp 186.000)
    - **Netral**: Paket Standard (Rp 120.000)
    - **Negatif**: Paket Basic (Rp 65.000)
    
    *Customer Lifetime Value* (CLV) kini dihitung **per tahun**. Jika pelanggan berhenti berkomentar dengan sentimen positif/netral, langganan akan diproyeksikan hingga akhir tahun. Jika pelanggan turun ke kategori "Rentan Churn" namun memiliki durasi langganan di atas modus tahunan (dengan gap maks 2 bulan atau total aktif $\ge$ 4 bulan), status akan diangkat menjadi "Cukup Loyal".
    """)
    
    # Ambil data unik per user dan per tahun agar penjumlahan tidak berlipat ganda
    unique_user_year_df = df_clv.drop_duplicates(subset=['userName', 'year'])
    total_users_yearly = len(unique_user_year_df)
    
    st.divider()
    
    # 1. TAMPILKAN METRIK TOTAL PROFIT NETFLIX
    st.subheader("💰 Metrik Historis Netflix (Akumulasi & Rata-rata)")
    
    # Perhitungan metrik tambahan
    total_clv_per_user = unique_user_year_df.groupby('userName')['user_clv'].sum()
    avg_expenditure_all_years = total_clv_per_user.mean()
    
    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)
    
    total_revenue = unique_user_year_df['user_clv'].sum()
    avg_revenue = unique_user_year_df['user_clv'].mean()
    
    # Rata-rata lama berlangganan pada seluruh rentang tahun (dibulatkan)
    total_duration_per_user = unique_user_year_df.groupby('userName')['duration'].sum()
    avg_duration_all_years = int(round(total_duration_per_user.mean()))
    
    col1.metric("Total Pendapatan Keseluruhan", f"Rp {total_revenue:,.0f}", help="Total pendapatan historis & proyeksi dari seluruh user")
    col2.metric("Rata-rata Pengeluaran User (Total)", f"Rp {avg_expenditure_all_years:,.0f}", help="Rata-rata total pengeluaran per user selama seluruh rentang tahun")
    col3.metric("Rata-rata Pengeluaran /Tahun", f"Rp {avg_revenue:,.0f}", help="Rata-rata pengeluaran user dalam satu tahun")
    
    col4.metric("Paket Paling Diminati", popular_package, help="Paket yang paling banyak digunakan secara akumulatif")
    col5.metric("Rata-rata Batas Sangat Loyal", f"Rp {avg_expenditure_overall:,.0f}", help="Rata-rata ambang batas sangat loyal dari seluruh rentang waktu")
    col6.metric("Rata-rata Lama Langganan", f"{avg_duration_all_years} bln", help="Rata-rata akumulasi bulan berlangganan per pengguna pada seluruh rentang tahun (dibulatkan)")
    
    st.divider()
    
    # 2. KOMPARASI STATISTIK ANTAR TAHUN
    st.subheader("📊 Komparasi Statistik Antar Tahun")
    
    yearly_stats = unique_user_year_df.groupby('year').agg(
        Total_Pendapatan=('user_clv', 'sum'),
        Rata_rata_Pengeluaran=('user_clv', 'mean'),
        Jumlah_User=('userName', 'count')
    ).reset_index()
    
    tab1, tab2, tab3 = st.tabs(["Total Pendapatan per Tahun", "Rata-rata Pengeluaran per Tahun", "Jumlah User Aktif per Tahun"])
    
    with tab1:
        fig_rev = px.bar(yearly_stats, x='year', y='Total_Pendapatan', text_auto='.2s', title="Total Pendapatan per Tahun (2021-2026)", color_discrete_sequence=['#3498db'], range_x=[2021, 2026])
        fig_rev.update_xaxes(dtick=1, tickmode='linear', tick0=2021)
        st.plotly_chart(fig_rev, use_container_width=True)
        
    with tab2:
        fig_avg = px.bar(yearly_stats, x='year', y='Rata_rata_Pengeluaran', text_auto='.2s', title="Rata-rata Pengeluaran per Tahun (2021-2026)", color_discrete_sequence=['#9b59b6'], range_x=[2021, 2026])
        fig_avg.add_scatter(x=yearly_stats['year'], y=yearly_stats['Rata_rata_Pengeluaran'], mode='lines+markers', name='Tren', line=dict(color='#e74c3c', width=3), marker=dict(size=8, color='#e74c3c'))
        fig_avg.update_xaxes(dtick=1, tickmode='linear', tick0=2021)
        st.plotly_chart(fig_avg, use_container_width=True)
        
    with tab3:
        fig_usr = px.line(yearly_stats, x='year', y='Jumlah_User', markers=True, title="Jumlah User Aktif per Tahun (2021-2026)", color_discrete_sequence=['#e67e22'], range_x=[2021, 2026])
        fig_usr.update_xaxes(dtick=1, tickmode='linear', tick0=2021)
        st.plotly_chart(fig_usr, use_container_width=True)
    
    st.divider()
    
    # 3. VISUALISASI: HISTOGRAM CHURN & LOYALITAS
    st.subheader("📊 Distribusi Loyalitas Pengguna (Akumulasi Semua Tahun)")
    col_chart1, col_chart2 = st.columns([2, 1])
    
    with col_chart1:
        # Menghitung jumlah per status loyalitas (berdasarkan (user, year))
        loyalty_counts = unique_user_year_df['loyalty_status'].value_counts().reset_index()
        loyalty_counts.columns = ['Status Loyalitas', 'Jumlah']
        
        # Membuat bar chart interaktif
        fig_bar = px.bar(
            loyalty_counts, 
            x='Status Loyalitas', 
            y='Jumlah',
            color='Status Loyalitas',
            color_discrete_map={
                "Sangat Loyal": "#2ecc71", 
                "Cukup Loyal": "#f1c40f", 
                "Rentan Churn (Berhenti)": "#e74c3c"
            },
            text_auto=True,
            title="Jumlah Insiden Pengguna Berdasarkan Kategori CLV"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_chart2:
        # Menampilkan persentase churn
        churn_count = loyalty_counts[loyalty_counts['Status Loyalitas'] == 'Rentan Churn (Berhenti)']['Jumlah'].sum()
        churn_rate = (churn_count / total_users_yearly) * 100 if total_users_yearly > 0 else 0
        
        st.markdown("### Ringkasan Churn")
        st.metric("Tingkat Risiko Churn", f"{churn_rate:.1f}%", delta_color="inverse")
        st.write("Persentase insiden (user-tahun) yang kemungkinan besar akan berhenti berlangganan.")

    st.divider()

    # 4. VISUALISASI: GRAFIK TREN PER TAHUN
    st.subheader("📈 Tren Loyalitas per Tahun")
    # Mengelompokkan data unik per tahun dan per user menggunakan dataframe yang sama
    trend_data = unique_user_year_df.groupby(['year', 'loyalty_status']).size().reset_index(name='Jumlah')
    
    fig_line = px.line(
        trend_data, 
        x='year', 
        y='Jumlah', 
        color='loyalty_status',
        markers=True,
        color_discrete_map={
            "Sangat Loyal": "#2ecc71", 
            "Cukup Loyal": "#f1c40f", 
            "Rentan Churn (Berhenti)": "#e74c3c"
        },
        title="Pergerakan Status Loyalitas Pengguna (2021-2026)",
        range_x=[2021, 2026]
    )
    # Memastikan sumbu X menampilkan tahun yang bulat
    fig_line.update_xaxes(dtick=1, tickmode='linear', tick0=2021) 
    st.plotly_chart(fig_line, use_container_width=True)

    st.divider()

    # 5. REKOMENDASI & KEPUTUSAN BISNIS (DECISION LOGIC)
    st.subheader("🎯 Rekomendasi Tindakan Bisnis")
    
    # Logika dinamis untuk memberikan saran berdasarkan data
    if churn_rate > 30:
        st.error(f"**🚨 PERINGATAN KRITIS:** Tingkat risiko churn sangat tinggi ({churn_rate:.1f}%).")
        st.markdown("""
        **Tindakan yang Direkomendasikan:**
        * **Investigasi Teknis Segera:** Sentimen negatif sangat mendominasi. Segera periksa keluhan utama di *Word Cloud* negatif (misal: *bug* aplikasi, *error* server).
        * **Kompensasi Pengguna:** Berikan penawaran khusus atau diskon perpanjangan untuk pengguna di kategori 'Rentan Churn' agar mereka tidak langsung beralih ke kompetitor.
        """)
    elif 15 <= churn_rate <= 30:
        st.warning(f"**⚠️ PERHATIAN:** Tingkat risiko churn berada di level menengah ({churn_rate:.1f}%).")
        st.markdown("""
        **Tindakan yang Direkomendasikan:**
        * **Strategi Retensi:** Kirimkan email promosi rekomendasi film/series yang dipersonalisasi kepada pengguna 'Cukup Loyal' agar mereka beralih menjadi 'Sangat Loyal'.
        * **Perbaikan Konten:** Analisis ulasan netral untuk melihat apa fitur atau tontonan yang dirasa kurang memuaskan oleh pengguna.
        """)
    else:
        st.success(f"**✅ KONDISI SEHAT:** Tingkat risiko churn cukup rendah ({churn_rate:.1f}%).")
        st.markdown("""
        **Tindakan yang Direkomendasikan:**
        * **Program Referral:** Maksimalkan pengguna 'Sangat Loyal' dengan memberikan promo jika mereka berhasil mengundang teman.
        * **Upselling:** Tawarkan paket berlangganan yang lebih tinggi (misal: dari Standard ke Premium 4K) kepada kelompok loyal ini karena tingkat kepercayaan mereka sedang tinggi.
        """)
        
    df_churn = get_churn_data(df)
    
    st.subheader("🚨 Analisis Transisi Sentimen (Churn Risk)")
    
    # Filter hanya untuk user yang baru saja berubah jadi negatif
    churn_users = df_churn[df_churn['is_churn_event'] == True]
    
    if not churn_users.empty:
        st.error(f"Ditemukan {len(churn_users)} pengguna yang berubah perilaku!")
        
        # Menampilkan tabel perbandingan agar mudah dibaca
        st.write("Daftar pengguna yang berubah dari Sentimen Positif/Netral ke Negatif:")
        
        # Membuat format tampilan agar perbandingan teks terlihat jelas
        display_cols = [
            'userName', 'prev_sentiment', 'sentiment', 
            'prev_content', 'content', 'potential_loss'
        ]
        st.dataframe(churn_users[display_cols], use_container_width=True)
        
        st.markdown("""
        **Cara Membaca Tabel:**
        * **prev_sentiment**: Status sebelum mereka kecewa.
        * **prev_content**: Apa yang mereka katakan saat masih 'bahagia' atau 'netral'.
        * **content**: Apa yang mereka katakan saat 'kecewa' (ini adalah alasan *churn*).
        """)
    else:
        st.success("Tidak ada perubahan sentimen drastis yang terdeteksi pada dataset ini.")