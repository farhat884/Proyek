import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
from sklearn.pipeline import Pipeline
import os


# INISIALISASI DIREKTORI OUTPUT
# Membuat folder 'visualizations' jika belum ada untuk menyimpan semua hasil grafik
output_dir = "visualizations"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# FUNGSI VISUALISASI: CONFUSION MATRIX
# Fungsi ini memvisualisasikan comfusion matrix untuk melihat 
# seberapa banyak prediksi yang benar vs salah pada setiap kategori.
def plot_confusion_matrix(y_true, y_pred, classes, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


# FUNGSI VISUALISASI: ROC CURVE (MULTICLASS)
# Fungsi ini membuat kurva ROC untuk setiap kelas (One-vs-Rest).
# Semakin besar area di bawah kurva (AUC), semakin baik model membedakan antar kelas.
def plot_roc_curve(y_test, y_score, classes, title, filename):
    y_test_bin = label_binarize(y_test, classes=classes)
    n_classes = len(classes)
    fpr, tpr, roc_auc = dict(), dict(), dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    plt.figure(figsize=(10, 8))
    colors = sns.color_palette("husl", n_classes)
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC {classes[i]} (AUC = {roc_auc[i]:0.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


# FUNGSI VISUALISASI: PRECISION-RECALL CURVE
# Fungsi ini berguna untuk mengevaluasi performa model pada dataset 
# yang mungkin tidak seimbang (imbalanced data).
def plot_precision_recall_curve(y_test, y_score, classes, title, filename):
    y_test_bin = label_binarize(y_test, classes=classes)
    n_classes = len(classes)
    precision, recall, average_precision = dict(), dict(), dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(y_test_bin[:, i], y_score[:, i])
    plt.figure(figsize=(10, 8))
    colors = sns.color_palette("husl", n_classes)
    for i, color in zip(range(n_classes), colors):
        plt.plot(recall[i], precision[i], color=color, lw=2, label=f'PR {classes[i]} (AP = {average_precision[i]:0.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


# FUNGSI VISUALISASI: FEATURE IMPORTANCE
# Fungsi ini mengekstrak kata-kata (fitur) yang paling berpengaruh 
# dalam pengambilan keputusan model untuk menentukan sentimen.
def plot_feature_importance(model, feature_names, title, filename, n_top=20):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_.toarray().mean(axis=0))
    else: return
    indices = np.argsort(importances)[-n_top:]
    plt.figure(figsize=(10, 8))
    plt.title(title)
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


# FUNGSI VISUALISASI: REPORT HEATMAP
# Mengonversi Classification Report (Precision, Recall, F1) 
# menjadi heatmap agar perbandingan antar kelas terlihat jelas secara visual.
def plot_classification_report_heatmap(y_true, y_pred, classes, title, filename):
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    report_df = pd.DataFrame(report).iloc[:-1, :len(classes)].T
    plt.figure(figsize=(10, 6))
    sns.heatmap(report_df, annot=True, cmap='RdYlGn', fmt='.2f')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


# ALUR UTAMA (MAIN PROCESS)
def main():
    input_file = "netflix_reviews_labeled.csv"
    if not os.path.exists(input_file):
        print(f"Error: {input_file} tidak ditemukan!")
        return

    # 1. Loading dan Preprocessing Awal
    print("Loading data...")
    df = pd.read_csv(input_file).dropna(subset=['content', 'sentiment'])
    X, y = df['content'], df['sentiment']
    classes = sorted(y.unique())
    
    # 2. Split Data (Mencegah Data Leakage secara MAKSIMAL)
    print("Splitting Data...")
    X_train_text, X_test_text, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # 3. Pelatihan Model 1: SVM dengan Pipeline & GridSearchCV
    print("\n--- Training SVM dengan GridSearchCV ---")
    svm_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 3))),
        ('clf', SVC(probability=True, class_weight='balanced', random_state=42))
    ])
    svm_param_grid = {'clf__C': [1], 'clf__kernel': ['linear']}
    svm_grid = GridSearchCV(svm_pipeline, svm_param_grid, cv=3, n_jobs=-1, verbose=1)
    svm_grid.fit(X_train_text, y_train)
    best_svm = svm_grid.best_estimator_
    y_pred_svm = best_svm.predict(X_test_text)
    y_score_svm = best_svm.predict_proba(X_test_text)
    
    # 4. Pelatihan Model 2: Random Forest dengan Pipeline & GridSearchCV
    print("\n--- Training Random Forest dengan GridSearchCV ---")
    rf_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 3))),
        ('clf', RandomForestClassifier(class_weight='balanced', random_state=42))
    ])
    rf_param_grid = {'clf__n_estimators': [100], 'clf__max_depth': [None], 'clf__min_samples_split': [2]}
    rf_grid = GridSearchCV(rf_pipeline, rf_param_grid, cv=3, n_jobs=-1, verbose=1)
    rf_grid.fit(X_train_text, y_train)
    best_rf = rf_grid.best_estimator_
    y_pred_rf = best_rf.predict(X_test_text)
    y_score_rf = best_rf.predict_proba(X_test_text)
    
    # 5. Generasi Seluruh Visualisasi Pengaruh
    print("\nGenerating Visualizations...")
    for model, y_pred, y_score, name in [(best_svm, y_pred_svm, y_score_svm, "SVM"), (best_rf, y_pred_rf, y_score_rf, "RF")]:
        print(f"Processing {name}...")
        plot_confusion_matrix(y_test, y_pred, classes, f"Confusion Matrix - {name}", f"{name.lower()}_cm.png")
        plot_roc_curve(y_test, y_score, classes, f"ROC Curve - {name}", f"{name.lower()}_roc.png")
        plot_precision_recall_curve(y_test, y_score, classes, f"PR Curve - {name}", f"{name.lower()}_pr.png")
        plot_classification_report_heatmap(y_test, y_pred, classes, f"Report - {name}", f"{name.lower()}_report.png")
        
        clf_model = model.named_steps['clf']
        tfidf_model = model.named_steps['tfidf']
        plot_feature_importance(clf_model, tfidf_model.get_feature_names_out(), f"Features - {name}", f"{name.lower()}_features.png")

    # 6. Perbandingan Akurasi Antar Model
    metrics = [
        {'Model': 'SVM', 'Accuracy': accuracy_score(y_test, y_pred_svm)},
        {'Model': 'Random Forest', 'Accuracy': accuracy_score(y_test, y_pred_rf)}
    ]
    df_metrics = pd.DataFrame(metrics)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df_metrics, x='Model', y='Accuracy')
    plt.ylim(0, 1.1)
    plt.title("Akurasi: SVM vs Random Forest")
    plt.savefig(os.path.join(output_dir, "model_comparison_accuracy.png"))
    
    # 7. Output Final Ringkasan
    print(f"\nSelesai! Semua visualisasi disimpan di folder '{output_dir}'.")
    print(f"Akurasi SVM: {accuracy_score(y_test, y_pred_svm):.4f}")
    print(f"Akurasi RF: {accuracy_score(y_test, y_pred_rf):.4f}")

if __name__ == "__main__":
    main()
