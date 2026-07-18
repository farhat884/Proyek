import pandas as pd

# 1. Preprocessing Data
data_prep = [
    ['D1', 'being forced to send an email so I can watch a show from a device that is logged into the account before on my home Wi-Fi is ridiculous. You almost killed piracy with this app...', 'forc send email watch show devic log account home wi-fi ridicul almost kill piraci applic much better kill applic'],
    ['D2', "won't let me delete my card information, and at random times, it generates a charge to my bank account.", "n't let delet card inform random time gener charg bank account"],
    ['D3', 'Have to completely reset the app and delete all downloads if you accidentally touch the brightness slider. It disables androids built in brightness control.', 'complet reset applic delet download accident touch bright slider disabl android built bright control'],
    ['D4', 'netflix is the best all-in-one entertsinment app. the interface is smooth, and the addition of high quality mobile games...', 'netflix best all-in-on entertsin applic interfac smooth addit high qualiti mobil game plu tab make easi find download save titl'],
    ['D5', "why is it too often for netflix to suddenly lagging, and I had to sign in so many times it's often failed... I bought it officially from netflix btw", 'often netflix suddenli lag sign mani time often fail need wait never succeed bought offici netflix way'],
    ['D6', "can't run on a Bootloader unlocked device.", "n't run bootload unlock devic"],
    ['D7', 'BRING BACK THE BTS PROFILE PICTURE PLEASE PLEASE PLEASE WE NEED IT ARMY NEEDS IT', 'bring back bt profil pictur pleas pleas pleas need armi need'],
    ['D8', "Rapidly becoming the worst streaming app available. New available content sucks... making it nearly impossible to find anything you actually WANT to watch...", "rapidli becom worst stream applic avail new avail content suck 're constantli tri forc bad origin content well alway increas cost justifi differ price tier add advertis constantli chang layout make nearli imposs find anyth actual want watch push thing want watch would give zero star option"],
    ['D9', 'I have no complaints about the quality it’s really good and perfect for a movie marathon. My only concern is that One Piece episodes are always delayed... ', 'no complaint qualiti realli good perfect movi marathon concern one piec episod alway delay arc also miss hope add complet arc futur keep updat prici din siya honest one piec thing wait'],
    ['D10', "I just Abit upset when there is not all movie is got to choose tamil option as audio it's really upsetting hope that Netflix could improve it tqq", 'abit upset not movi got choos tamil option audio realli upset hope netflix could improv tqq']
]
df_prep = pd.DataFrame(data_prep, columns=['ID', 'Raw Text', 'Cleaned & Stemmed Text'])

# 2. Labeling Data
data_label = [
    ['D1', 1, '1 <= 2', 'Negatif'],
    ['D2', 1, '1 <= 2', 'Negatif'],
    ['D3', 1, '1 <= 2', 'Negatif'],
    ['D4', 5, '5 >= 4', 'Positif'],
    ['D5', 2, '2 <= 2', 'Negatif'],
    ['D6', 1, '1 <= 2', 'Negatif'],
    ['D7', 5, '5 >= 4', 'Positif'],
    ['D8', 1, '1 <= 2', 'Negatif'],
    ['D9', 2, '2 <= 2', 'Negatif'],
    ['D10', 4, '4 >= 4', 'Positif']
]
df_label = pd.DataFrame(data_label, columns=['ID', 'Score', 'Aturan (Rule)', 'Label Asli'])

# 3. TF-IDF Calculation
data_tfidf = [
    ['applic', 4, 'ln(11/5) + 1', 1.788, 1, 1.788, '1.788/2.690', 0.664],
    ['netflix', 3, 'ln(11/4) + 1', 2.011, 1, 2.011, '2.011/2.690', 0.747],
    ['watch', 2, 'ln(11/3) + 1', 2.299, 0, 0.0, '0.0/2.690', 0.0],
    ['good', 1, 'ln(11/2) + 1', 2.704, 0, 0.0, '0.0/2.690', 0.0],
    ['movi', 2, 'ln(11/3) + 1', 2.299, 0, 0.0, '0.0/2.690', 0.0]
]
df_tfidf = pd.DataFrame(data_tfidf, columns=['Term', 'DF (Doc Freq)', 'Rumus IDF', 'Nilai IDF', 'TF (di D4)', 'Bobot Awal (TF*IDF)', 'Rumus Normalisasi L2', 'TF-IDF Final D4'])

# 4. SVM
data_svm = [
    ['applic', 1.0, 0.664, 0.664],
    ['netflix', 1.5, 0.747, 1.1205],
    ['watch', 0.5, 0.0, 0.0],
    ['good', 2.0, 0.0, 0.0],
    ['movi', 1.5, 0.0, 0.0],
    ['Bias (b)', 0.5, '-', 0.5],
    ['TOTAL f(x)', '-', '-', 2.2845]
]
df_svm = pd.DataFrame(data_svm, columns=['Term / Komponen', 'Bobot Model (W)', 'Nilai Fitur D4 (X)', 'Hasil (W * X)'])

# 5. Evaluasi
data_eval = [
    ['True Positive (TP)', 4, 'Accuracy', '(4+4)/10 = 0.80'],
    ['True Negative (TN)', 4, 'Precision', '4/5 = 0.80'],
    ['False Positive (FP)', 1, 'Recall', '4/5 = 0.80'],
    ['False Negative (FN)', 1, 'F1-Score', '2*(0.8*0.8)/(0.8+0.8) = 0.80']
]
df_eval = pd.DataFrame(data_eval, columns=['Confusion Matrix', 'Nilai', 'Metrik', 'Perhitungan'])

with pd.ExcelWriter('Simulasi_Manual_Netflix.xlsx') as writer:
    df_prep.to_excel(writer, sheet_name='1. Preprocessing', index=False)
    df_label.to_excel(writer, sheet_name='2. Labeling', index=False)
    df_tfidf.to_excel(writer, sheet_name='3. TF-IDF', index=False)
    df_svm.to_excel(writer, sheet_name='4. Klasifikasi SVM', index=False)
    df_eval.to_excel(writer, sheet_name='5. Evaluasi', index=False)

print('Excel generated successfully.')
