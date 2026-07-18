## 4.2 Implementasi Terintegrasi Berbasis Pemrograman (Python)

Sebagai bentuk otomasi komputasi tingkat lanjut yang adaptif, seluruh kerangka kerja algoritma diintegrasikan ke dalam satu alur eksekusi sekuensial (*end-to-end pipeline*) berbasis Python. Modul ini secara efisien memproses lebih dari 150 ribu baris data melalui pendekatan pemelajaran mesin (*Machine Learning*) yang *robust*, mulai dari ekstraksi awal hingga penentuan kerugian finansial pelanggan.

### 4.2.1 Data Mentah dan Data Olahan (Alur Dataset)

Tahap pertama dalam *pipeline* Python adalah memuat data ulasan mentah ke dalam sistem dan menyaringnya agar hanya data yang relevan dan valid yang diteruskan ke tahap analisis.

Untuk memastikan model belajar dari tren sentimen yang lebih modern dan relevan dengan kondisi aplikasi Netflix masa kini, sistem menerapkan pemotongan data (*cutting data*). Melalui fungsi konversi `pd.to_datetime()`, sistem mengisolasi ulasan sehingga hanya data sejak tahun 2021 hingga observasi terbaru yang digunakan dalam pemrosesan.

`[Tempat Screenshot Kode: Blok pemuatan pd.read_csv, filter tahun >= 2021, dan df.dropna(subset=['content']) pada file main_pipeline.py]`  
**Gambar 4.1 Potongan Kode Inisialisasi, Cutting Data, dan Pemuatan Dataset**

`[Tempat Screenshot Hasil: Output terminal/konsol yang menampilkan pesan "Total baris asli: ...", "Total baris setelah filter tahun (>= 2021): ..." dan "Total baris setelah cleaning: ..."]`  
**Gambar 4.2 Output Proses Cutting dan Pemuatan Dataset di Terminal**

Merujuk pada Gambar 4.1, eksplorasi data diinisialisasi menggunakan pustaka Pandas. Fungsi `pd.read_csv()` digunakan untuk memuat himpunan data Netflix ke dalam struktur DataFrame. Setelah itu, sistem menyaring stempel waktu pada kolom `at` dan hanya mempertahankan ulasan yang ditulis mulai dari tahun 2021 ke atas. Selanjutnya, sistem melakukan validasi struktural otomatis untuk mengidentifikasi dan membuang kekosongan data (*missing values*) melalui perintah `df.dropna(subset=['content'])`. Pemotongan rentang waktu dan penanganan data kosong ini esensial untuk memastikan bahwa model menghasilkan metrik prediksi dan estimasi kerugian finansial yang sesuai dengan dinamika perilaku pengguna Netflix pada era modern (pasca-2021). Keberhasilan penyaringan dan peringkasan baris data ini dapat dilihat secara gamblang pada kelancaran pemrosesan di output konsol di Gambar 4.2.

### 4.2.2 Pembersihan Data (Text Preprocessing)

Setelah data dimuat, teks ulasan yang masih kotor harus dinormalisasi agar mesin dapat membaca polanya dengan murni. 

`[Tempat Screenshot Kode: Blok fungsi def clean_text(text): yang memuat logika Regex dan modul emoji]`  
**Gambar 4.3 Fungsi Pembersihan Teks (Text Preprocessing)**

Merujuk pada Gambar 4.3, pembersihan teks diotomatisasi secara ketat melalui fungsionalitas Ekspresi Reguler (*Regex*). Melalui metode `re.sub()`, seluruh atribut konten dipindai untuk mengeliminasi anomali teks tidak terstruktur seperti tautan URL, sebutan akun (*mention*), serta tagar (*hashtag*). Selain itu, untuk meningkatkan presisi pemrosesan karakter murni, algoritma turut mengintegrasikan modul pihak ketiga yakni `emoji.replace_emoji()` guna menghapus seluruh piktograf. Proses ini diakhiri dengan *case folding* untuk menyeragamkan seluruh huruf menjadi huruf kecil, serta pemotongan imbuhan (*stemming*) dan penghapusan kata umum (*stopwords*) menggunakan *library* NLTK.

### 4.2.3 Pelabelan Sentimen

Langkah selanjutnya adalah memberikan label kebenaran dasar (*ground truth*) pada setiap baris ulasan agar model klasifikasi AI memiliki acuan untuk belajar.

`[Tempat Screenshot Kode: Blok fungsi def get_sentiment(row): yang mengilustrasikan logika fallback menggunakan TextBlob]`  
**Gambar 4.4 Logika Pelabelan Sentimen Sentimen**

`[Tempat Screenshot Hasil: Output terminal yang mencetak Distribusi Sentimen (jumlah ulasan Positif, Negatif, Netral)]`  
**Gambar 4.5 Distribusi Akhir Label Sentimen**

Merujuk pada Gambar 4.4, penetapan label target diproses menggunakan pendekatan hibrida (*hybrid approach*). Fungsi klasifikasi secara hierarkis mengevaluasi atribut `score` (rating bintang) terlebih dahulu (skor $\ge 4$ dinilai positif, $\le 2$ bernilai negatif). Namun, jika terjadi limitasi seperti ketiadaan skor rating (`NaN`), sistem secara cerdas melakukan rujukan mundur (*fallback*) dengan mengoperasikan modul komputasi linguistik **TextBlob**. Sentimen dievaluasi dari derajat polaritas teks, sehingga memastikan 150 ribu lebih baris ulasan terdistribusi merata ke dalam kelas sentimennya tanpa ada data yang terbuang, seperti yang dibuktikan pada hasil rekapitulasi jumlah label di Gambar 4.5.

### 4.2.4 Pelatihan Model AI (Ekstraksi Fitur & Klasifikasi)

Setelah data dibersihkan dan dilabeli, model diinstruksikan untuk mengenali pola teks dan memprediksi sentimen tersebut secara algoritmik.

`[Tempat Screenshot Kode: Bagian inisiasi TfidfVectorizer, train_test_split, dan perintah svm_model.fit(X_train, y_train)]`  
**Gambar 4.6 Pelatihan Model Support Vector Machine (SVM)**

`[Tempat Screenshot Hasil: Output Konsol/Terminal yang menunjukkan nilai Akurasi Model SVM dan Classification Report]`  
**Gambar 4.7 Evaluasi Akurasi dan Laporan Klasifikasi SVM**

Berdasarkan Gambar 4.6, ekstraksi fitur matematis disempurnakan menggunakan algoritma `TfidfVectorizer` untuk mengonversi representasi tekstual ke dalam matriks renggang (*sparse matrix*). Dataset kemudian dipecah menggunakan `train_test_split` ke dalam rasio pengujian 70:30 (Data Latih vs Data Uji) demi mencegah terjadinya kebocoran data (*data leakage*). Evaluasi kemudian dilakukan menggunakan algoritma *Support Vector Classifier* (SVC) bermetode kernel linear. Merujuk pada hasil di Gambar 4.7, pemilihan spesifikasi arsitektur ini membuktikan keandalannya dalam mencari hiperbidang pembatas (*hyperplane*) optimal untuk klasifikasi data teks multidimensi, dibuktikan dengan tingginya nilai presisi dan akurasi yang berhasil dicapai oleh model saat dihadapkan pada data pengujian.

### 4.2.5 Deteksi Churn, Perhitungan CLV, dan Status Loyalitas

Tahap puncak dari *pipeline* ini adalah menerjemahkan hasil klasifikasi sentimen linguistik menjadi valuasi dan proyeksi kerugian finansial bisnis perusahaan.

`[Tempat Screenshot Kode: Blok perhitungan komputasi CLV mulai dari def get_lifespan hingga pemanggilan fungsi shift(1)]`  
**Gambar 4.8 Komputasi Deteksi Churn dan Behavioral CLV**

`[Tempat Screenshot Hasil: Output terminal yang mencetak total potensi kerugian finansial (Total Potensi Kerugian Finansial: Rp ...)]`  
**Gambar 4.9 Hasil Agregasi Potensi Kerugian Finansial**

Sebagaimana diilustrasikan pada Gambar 4.8, penyelesaian masalah bisnis dituntaskan melalui integrasi komputasi *Behavioral CLV*. Modul Python dimanipulasi dengan mengonversi stempel waktu ke dalam struktur *datetime* untuk pengurutan kronologis yang linier. Model kemudian menerapkan metode pengelompokan data per entitas pengguna via fungsi `.groupby('userName')` yang dipadukan dengan operator `.shift(1)` untuk mengisolasi riwayat temporal masa lalu. 

Kondisional logika *boolean* lintas-baris lalu diterapkan guna mengekstrak variabel `is_churn_event`. Logika ini bertugas mengidentifikasi *churn* temporal, yakni penurunan loyalitas secara spesifik dari yang sebelumnya berstatus netral/positif lalu jatuh menjadi negatif. Pada akhirnya, sebagaimana ditunjukkan pada Gambar 4.9, sistem secara otonom memetakan dan merekap akumulasi penurunan kapital tersebut sebagai *Potential Loss* akhir yang disajikan dalam satuan moneter Rupiah.
