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

# =====================================================================
# FUNGSI PEMBANTU: TAMPILKAN DATASET TERPISAH (POSITIF, NETRAL, NEGATIF)
# =====================================================================
def display_separated_datasets(df_input, title_prefix=""):
    """
    Fungsi untuk menampilkan dataset terpisah berdasarkan label sentimen
    ke dalam 3 tab (Positif, Netral, Negatif).
    """
    st.subheader(f"📋 {title_prefix} Dataset Terpisah Berdasarkan Sentimen")
    
    if 'sentiment' not in df_input.columns:
        st.warning("Kolom 'sentiment' tidak ditemukan pada dataset ini.")
        return

    # Filter Dataframe
    df_pos = df_input[df_input['sentiment'] == 'positive']
    df_neu = df_input[df_input['sentiment'] == 'neutral']
    df_neg = df_input[df_input['sentiment'] == 'negative']

    tab_pos, tab_neu, tab_neg = st.tabs([
        f"🟢 Dataset Positif ({len(df_pos):,} data)", 
        f"🟡 Dataset Netral ({len(df_neu):,} data)", 
        f"🔴 Dataset Negatif ({len(df_neg):,} data)"
    ])

    with tab_pos:
        if not df_pos.empty:
            st.dataframe(df_pos, use_container_width=True)
        else:
            st.info("Tidak ada data dengan sentimen positif.")

    with tab_neu:
        if not df_neu.empty:
            st.dataframe(df_neu, use_container_width=True)
        else:
            st.info("Tidak ada data dengan sentimen netral.")

    with tab_neg:
        if not df_neg.empty:
            st.dataframe(df_neg, use_container_width=True)
        else:
            st.info("Tidak ada data dengan sentimen negatif.")


# =====================================================================
# 1. INISIALISASI DATASET MASTER (TERPUSAT DI SESSION STATE)
# =====================================================================
if 'df_master' not in st.session_state:
    try:
        # Load dataset bawaan saat pertama kali aplikasi dijalankan
        st.session_state.df_master = pd.read_csv("netflix_reviews_labeled.csv")
    except Exception as e:
        # Jika file csv bawaan tidak ada, buat dataframe kosong
        st.session_state.df_master = pd.DataFrame(columns=['userName', 'content', 'sentiment', 'at'])

# =====================================================================
# 2. PENGELOLAAN DATA TERPUSAT (SATU TEMPAT DI SIDEBAR)
# =====================================================================
st.sidebar.title("⚙️ Pengelolaan Data & Navigasi")

# Indikator Jumlah Data Saat Ini
st.sidebar.metric("Total Data Terdaftar", f"{len(st.session_state.df_master):,} ulasan")

# --- A. UNGGAH FILE (APPEND / GABUNG DATA) ---
st.sidebar.subheader("📁 Tambah Data via File")
uploaded_file = st.sidebar.file_uploader(
    "Unggah Excel / CSV untuk menggabungkan data:", 
    type=["csv", "xlsx", "xls"]
)

if uploaded_file is not None:
    if st.sidebar.button("➕ Gabungkan File ke Dataset"):
        try:
            if uploaded_file.name.endswith('.csv'):
                new_data = pd.read_csv(uploaded_file)
            else:
                new_data = pd.read_excel(uploaded_file)
            
            # Penggabungan data (Concat/Append)
            st.session_state.df_master = pd.concat([st.session_state.df_master, new_data], ignore_index=True)
            st.sidebar.success(f"Berhasil menambahkan {len(new_data):,} data baru!")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Gagal menggabungkan data: {e}")

# --- B. INPUT MANUAL (APPEND 1 DATA) ---
with st.sidebar.expander("➕ Tambah 1 Review Manual"):
    with st.form("add_single_review_form"):
        new_user = st.text_input("Nama User", "Pengguna Baru")
        new_content = st.text_area("Isi Ulasan", "")
        new_sentiment = st.selectbox("Sentimen", ["positive", "neutral", "negative"])
        new_date = st.date_input("Tanggal")
        submit_btn = st.form_submit_button("Tambah ke Dataset")
        
        if submit_btn:
            # Pastikan tanggal langsung dikonversi ke format pd.Timestamp
            parsed_date = pd.to_datetime(new_date)
            
            new_row = pd.DataFrame({
                'userName': [new_user],
                'content': [new_content],
                'sentiment': [new_sentiment],
                'at': [parsed_date]
            })
            
            # Penggabungan data
            st.session_state.df_master = pd.concat([st.session_state.df_master, new_row], ignore_index=True)
            st.sidebar.success("Review berhasil ditambahkan!")
            st.rerun()

# --- C. RESET DATASET (OPSIONAL) ---
if st.sidebar.button("🔄 Reset ke Data Bawaan Awal"):
    try:
        st.session_state.df_master = pd.read_csv("netflix_reviews_labeled.csv")
        st.sidebar.info("Dataset berhasil dikembalikan ke kondisi awal.")
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"Gagal mereset data: {e}")

st.sidebar.markdown("---")

# --- NAVIGASI HALAMAN ---
page = st.sidebar.selectbox("Pilih Halaman", [
    "Ringkasan Data", 
    "Pencarian Kata", 
    "Visualisasi Model", 
    "Analisis Kata (WordCloud & Top Words)",
    "Analisis CLV & Loyalitas"
])

# =====================================================================
# 3. PREPROCESSING UTAMA (DIGUNAKAN UNTUK SEMUA HALAMAN)
# =====================================================================
df = st.session_state.df_master.copy()

if not df.empty and 'at' in df.columns:
    df['at'] = pd.to_datetime(df['at'], errors='coerce')
    df_filtered = df[(df['at'].dt.year >= 2021) & (df['at'].dt.year <= 2026)]
    if not df_filtered.empty:
        df = df_filtered

# Cek ketersediaan Data
if df.empty:
    st.warning("⚠️ Belum ada data yang tersedia. Silakan unggah file atau tambah data di Sidebar.")
    st.stop()


# Cache fungsi untuk CLV & Churn
@st.cache_data
def get_clv_data(dataframe):
    return apply_clv_to_dataframe(dataframe)

@st.cache_data
def get_churn_data(dataframe):
    return detect_churn_events(dataframe)


# =====================================================================
# --- HALAMAN 1: RINGKASAN DATA ---
# =====================================================================
if page == "Ringkasan Data":
    st.title("📊 Ringkasan Sentimen Netflix Reviews")
    
    total_data = len(df)
    sentiment_counts = df['sentiment'].value_counts()
    most_frequent = sentiment_counts.idxmax() if not sentiment_counts.empty else "-"
    most_frequent_count = sentiment_counts.max() if not sentiment_counts.empty else 0
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Data Labeling", f"{total_data:,} reviews")
    col2.metric("Sentimen Terbanyak", str(most_frequent).capitalize())
    col3.metric("Jumlah Sentimen Terbanyak", f"{most_frequent_count:,} reviews")
    
    st.markdown("---")
    
    st.subheader("📈 Penyebaran Keseluruhan Sentimen")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.countplot(data=df, x='sentiment', palette='viridis', ax=ax)
    ax.set_title("Distribusi Label Sentimen")
    ax.set_xlabel("Sentimen")
    ax.set_ylabel("Jumlah")
    st.pyplot(fig)
    
    st.info(f"""
    **Analisis Singkat:**
    Data labeling terdiri dari **{total_data:,}** ulasan Netflix. 
    Sentimen yang paling mendominasi adalah **{most_frequent}** dengan total **{most_frequent_count:,}** ulasan.
    """)

    st.divider()
    display_separated_datasets(df, title_prefix="Ringkasan")

# =====================================================================
# --- HALAMAN 2: PENCARIAN KATA ---
# =====================================================================
elif page == "Pencarian Kata":
    st.title("🔍 Analisis Kata Kunci")
    st.markdown("Cari kata tertentu untuk melihat bagaimana sentimen pengguna terhadap topik tersebut.")
    
    query = st.text_input("Masukkan kata kunci (contoh: 'price', 'movie', 'bad'):", "")
    
    if query:
        filtered_df = df[df['content'].astype(str).str.contains(query, case=False, na=False)]
        
        if not filtered_df.empty:
            st.success(f"Ditemukan {len(filtered_df):,} ulasan mengandung kata '{query}'")
            
            st.subheader(f"Penyebaran Sentimen untuk Kata: '{query}'")
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.countplot(data=filtered_df, x='sentiment', palette='magma', order=['positive', 'neutral', 'negative'], ax=ax)
            st.pyplot(fig)
            
            st.divider()
            display_separated_datasets(filtered_df, title_prefix=f"Hasil Pencarian '{query}'")
        else:
            st.warning(f"Tidak ditemukan ulasan yang mengandung kata '{query}'.")
    else:
        st.divider()
        display_separated_datasets(df, title_prefix="Eksplorasi")

# =====================================================================
# --- HALAMAN 3: VISUALISASI MODEL ---
# =====================================================================
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

    st.divider()
    display_separated_datasets(df, title_prefix="Data Pengujian Pemodelan")

# =====================================================================
# --- HALAMAN 4: ANALISIS KATA ---
# =====================================================================
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

    st.divider()
    display_separated_datasets(df, title_prefix="Eksplorasi Kata Teks")

# =====================================================================
# --- HALAMAN 5: ANALISIS CLV & LOYALITAS ---
# =====================================================================
elif page == "Analisis CLV & Loyalitas":
    st.title("💎 Pendekatan Behavioral CLV (Analisis Multi-Skenario)")
    st.markdown("Analisis ini menggunakan variasi paket Netflix berdasarkan referensi penelitian terbaru untuk memetakan profit tahunan dan memprediksi retensi pengguna.")
    
    # Proses data CLV & Churn dari dataframe master yang aktif
    df_clv = get_clv_data(df)
    df_churn = get_churn_data(df)
    
    # Menentukan nama paket langganan
    package_map = {186000: 'Premium', 120000: 'Standard', 65000: 'Basic'}
    popular_package_price = 120000
    assumed_package = package_map.get(popular_package_price, 'Tidak Diketahui')
    
    # --- 1. KOTAK INFORMASI ---
    st.info("""
    **Informasi Penentuan Loyalitas & CLV (Perhitungan Per Tahun):**
    - **Positif**: Paket Premium (Rp 186.000)
    - **Netral**: Paket Standard (Rp 120.000)
    - **Negatif**: Paket Basic (Rp 65.000)
    
    *Customer Lifetime Value* (CLV) kini dihitung **per tahun**.
    """)
    
    # Preprocessing Data User & Tahun
    unique_user_year_df = df_clv.drop_duplicates(subset=['userName', 'year']) if 'userName' in df_clv.columns and 'year' in df_clv.columns else df_clv
    total_users_yearly = len(unique_user_year_df)
    
    # Hitung Nilai Financial
    total_revenue = unique_user_year_df['user_clv'].sum() if 'user_clv' in unique_user_year_df.columns else 0
    
    if 'potential_loss' in df_churn.columns and 'is_churn_event' in df_churn.columns:
        total_potential_loss = df_churn[df_churn['is_churn_event'] == True]['potential_loss'].sum()
    else:
        total_potential_loss = len(df_churn[df_churn['is_churn_event'] == True]) * 1200000 if 'is_churn_event' in df_churn.columns else 0

    st.markdown("<br>", unsafe_allow_html=True)

    # --- 2. METRIK UTAMA DENGAN BACKGROUND CARD & TEKS TENGAH ---
    st.subheader("💰 Ringkasan Metrik Utama Financial & CLV")
    
    card_col1, card_col2 = st.columns(2)
    
    with card_col1:
        st.markdown(f"""
        <div style="
            background-color: #ebf5fb; 
            border: 1px solid #aed6f1; 
            border-radius: 12px; 
            padding: 20px; 
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        ">
            <p style="margin: 0; font-size: 16px; font-weight: 600; color: #1b4f72;">💵 Total Pendapatan Keseluruhan</p>
            <h1 style="margin: 10px 0; font-size: 36px; font-weight: bold; color: #2e86c1;">Rp {total_revenue:,.0f}</h1>
            <p style="margin: 0; font-size: 12px; color: #5d6d7e;">Total akumulasi pendapatan historis & proyeksi dari seluruh user</p>
        </div>
        """, unsafe_allow_html=True)
        
    with card_col2:
        st.markdown(f"""
        <div style="
            background-color: #fdedec; 
            border: 1px solid #f5b7b1; 
            border-radius: 12px; 
            padding: 20px; 
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        ">
            <p style="margin: 0; font-size: 16px; font-weight: 600; color: #78281f;">🚨 Potensi Kerugian dari Churn Event</p>
            <h1 style="margin: 10px 0; font-size: 36px; font-weight: bold; color: #cb4335;">-Rp {total_potential_loss:,.0f}</h1>
            <p style="margin: 0; font-size: 12px; color: #78281f;">Estimasi potensi kerugian akibat pergeseran sentimen ke Negatif</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- 3. METRIK PENDUKUNG ---
    if 'userName' in unique_user_year_df.columns and 'user_clv' in unique_user_year_df.columns:
        total_clv_per_user = unique_user_year_df.groupby('userName')['user_clv'].sum()
        avg_expenditure_all_years = total_clv_per_user.mean() if not total_clv_per_user.empty else 0
        avg_revenue_yearly = unique_user_year_df['user_clv'].mean()
    else:
        avg_expenditure_all_years = 0
        avg_revenue_yearly = 0
        
    if 'userName' in unique_user_year_df.columns and 'duration' in unique_user_year_df.columns:
        total_duration_per_user = unique_user_year_df.groupby('userName')['duration'].sum()
        avg_duration_all_years = int(round(total_duration_per_user.mean())) if not total_duration_per_user.empty else 0
    else:
        avg_duration_all_years = 0

    sub_col1, sub_col2, sub_col3, sub_col4 = st.columns(4)
    sub_col1.metric("Paket yang Dijadikan Asumsi", assumed_package, help="Paket acuan berdasarkan asumsi utama")
    sub_col2.metric("Rata-rata Pengeluaran Total", f"Rp {avg_expenditure_all_years:,.0f}")
    sub_col3.metric("Rata-rata Pengeluaran /Tahun", f"Rp {avg_revenue_yearly:,.0f}")
    sub_col4.metric("Rata-rata Lama Langganan", f"{avg_duration_all_years} bln")
    
    st.divider()
    
    # --- 4. KOMPARASI STATISTIK ---
    st.subheader("📊 Komparasi Statistik Antar Tahun")
    if 'year' in unique_user_year_df.columns and 'user_clv' in unique_user_year_df.columns:
        yearly_stats = unique_user_year_df.groupby('year').agg(
            Total_Pendapatan=('user_clv', 'sum'),
            Rata_rata_Pengeluaran=('user_clv', 'mean'),
            Jumlah_User=('userName', 'count')
        ).reset_index()
        
        tab1, tab2, tab3 = st.tabs(["Total Pendapatan per Tahun", "Rata-rata Pengeluaran per Tahun", "Jumlah User Aktif per Tahun"])
        
        with tab1:
            fig_rev = px.bar(yearly_stats, x='year', y='Total_Pendapatan', text_auto='.2s', title="Total Pendapatan per Tahun", color_discrete_sequence=['#3498db'])
            st.plotly_chart(fig_rev, use_container_width=True)
            
        with tab2:
            fig_avg = px.bar(yearly_stats, x='year', y='Rata_rata_Pengeluaran', text_auto='.2s', title="Rata-rata Pengeluaran per Tahun", color_discrete_sequence=['#9b59b6'])
            fig_avg.add_scatter(x=yearly_stats['year'], y=yearly_stats['Rata_rata_Pengeluaran'], mode='lines+markers', name='Tren', line=dict(color='#e74c3c', width=3))
            st.plotly_chart(fig_avg, use_container_width=True)
            
        with tab3:
            fig_usr = px.line(yearly_stats, x='year', y='Jumlah_User', markers=True, title="Jumlah User Aktif per Tahun", color_discrete_sequence=['#e67e22'])
            st.plotly_chart(fig_usr, use_container_width=True)
    else:
        st.info("Kolom 'year' atau 'user_clv' tidak tersedia untuk menampilkan komparasi tahunan.")
    
    st.divider()
    
    # --- 5. CHURN & LOYALITAS ---
    st.subheader("📊 Distribusi Loyalitas Pengguna (Akumulasi Semua Tahun)")
    col_chart1, col_chart2 = st.columns([2, 1])
    
    with col_chart1:
        if 'loyalty_status' in unique_user_year_df.columns:
            loyalty_counts = unique_user_year_df['loyalty_status'].value_counts().reset_index()
            loyalty_counts.columns = ['Status Loyalitas', 'Jumlah']
            
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
        if 'loyalty_status' in unique_user_year_df.columns:
            churn_count = loyalty_counts[loyalty_counts['Status Loyalitas'] == 'Rentan Churn (Berhenti)']['Jumlah'].sum()
            churn_rate = (churn_count / total_users_yearly) * 100 if total_users_yearly > 0 else 0
            
            st.markdown("### Ringkasan Churn")
            st.metric("Tingkat Risiko Churn", f"{churn_rate:.1f}%")

    st.divider()

    # --- 6. CHURN RISK TRANSITION ---
    st.subheader("🚨 Analisis Transisi Sentimen (Churn Risk)")
    if 'is_churn_event' in df_churn.columns:
        churn_users = df_churn[df_churn['is_churn_event'] == True]
        if not churn_users.empty:
            st.error(f"Ditemukan {len(churn_users):,} pengguna yang berubah perilaku!")
            display_cols = ['userName', 'prev_sentiment', 'sentiment', 'prev_content', 'content', 'potential_loss']
            existing_display_cols = [c for c in display_cols if c in churn_users.columns]
            st.dataframe(churn_users[existing_display_cols], use_container_width=True)
        else:
            st.success("Tidak ada perubahan sentimen drastis yang terdeteksi pada dataset ini.")
    else:
        st.info("Data churn event tidak tersedia.")

    st.divider()
    display_separated_datasets(df, title_prefix="Rincian Sentimen CLV")