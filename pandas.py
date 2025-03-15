#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 12:24:35 2025
Modified on: 15 Mar 2025
Author: Tunahan

Bu proje, UCI Adult veri seti üzerinden;
- Veri yükleme, temizleme ve ön işleme,
- Kategorik değişkenlerin etiketlenmesi,
- Sayısal verilerin normalizasyonu,
- Veri keşfi (EDA) ve görselleştirme,
- GroupBy, filtering (query) ve merge/join işlemlerini
kapsamlı şekilde uygulayarak makine öğrenmesi (Logistic Regression) modeli oluşturmayı amaçlamaktadır.
"""

# Gerekli kütüphanelerin import edilmesi
from ucimlrepo import fetch_ucirepo 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve

# 1. Veri Setinin Yüklenmesi ve Ön İşleme
# Veri setini çekiyoruz
adult = fetch_ucirepo(id=2)
# Özellikler ve hedef değişken
X = adult.data.features.copy()
y = adult.data.targets.copy()

# Hedef değişkeni binary hale getirme
y = y.replace({">50K.": 1, ">50K": 1, "<=50K.": 0, "<=50K": 0}).infer_objects(copy=False)

# Metadata bilgisini yazdırma
print("Veri Seti Metadata:")
print(adult.metadata)

# 2. Veri Temizleme ve Dönüşümler

# Eksik ve hatalı değerler için: workclass, occupation, native-country sütunlarında "?" yerine NaN
for col in ["workclass", "occupation", "native-country"]:
    X[col] = X[col].replace("?", None).fillna(X[col].mode()[0])
    
# Örnek olarak marital-status sütunu: bazı kategorileri "other" ile gruplayıp etiketleme
X["marital-status"] = X["marital-status"].replace(
    ["Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"],
    "other"
)

# LabelEncoder örneği: cinsiyet değişkeni
lb = LabelEncoder()
X["sex"] = lb.fit_transform(X["sex"])

# marital-status sütunu etiketleme
X["marital-status"] = lb.fit_transform(X["marital-status"])

# workclass sütunu: bazı kategorik değerleri gruplandırma
X["workclass"] = X["workclass"].replace(
    {"Without-pay": "Other", "Never-worked": "Other",
     "Local-gov": "Gov", "State-gov": "Gov", "Federal-gov": "Gov",
     "Self-emp-not-inc": "Self-Employed", "Self-emp-inc": "Self-Employed"}
)
X["workclass"] = lb.fit_transform(X["workclass"])

# education sütunu: eğitim seviyelerini gruplandırma ve etiketleme
X["education"] = X["education"].replace(
    {
        "Preschool": "Primary",
        "1st-4th": "Primary",
        "5th-6th": "Primary",
        "7th-8th": "Primary",
        "9th": "High School",
        "10th": "High School",
        "11th": "High School",
        "12th": "High School",
        "HS-grad": "College",
        "Some-college": "College",
        "Assoc-voc": "College",
        "Assoc-acdm": "College",
        "Bachelors": "Bachelors",
        "Masters": "Graduate",
        "Prof-school": "Graduate",
        "Doctorate": "Graduate",
    }
)
X["education"] = lb.fit_transform(X["education"])

# occupation sütunu: meslek gruplarını basitleştirme ve etiketleme
X["occupation"] = X["occupation"].replace(
    {
        "Prof-specialty": "White-Collar-High",
        "Exec-managerial": "White-Collar-High",
        "Sales": "White-Collar-Low",
        "Adm-clerical": "White-Collar-Low",
        "Tech-support": "White-Collar-Low",
        "Craft-repair": "Blue-Collar",
        "Machine-op-inspct": "Blue-Collar",
        "Transport-moving": "Blue-Collar",
        "Handlers-cleaners": "Blue-Collar",
        "Farming-fishing": "Blue-Collar",
        "Protective-serv": "Service",
        "Other-service": "Service",
        "Priv-house-serv": "Service",
        "Armed-Forces": "Military"  
    }
)
X["occupation"] = lb.fit_transform(X["occupation"])

# Sayısal sütunlar için normalizasyon: hours-per-week ve education-num
scaler = MinMaxScaler()
X["hours-per-week"] = scaler.fit_transform(X[["hours-per-week"]])
X["education-num"] = scaler.fit_transform(X[["education-num"]])

# native-country sütunu: sadece "United-States" ve "Other" olarak gruplandırma
X["native-country"] = X["native-country"].apply(lambda x: "Other" if x != "United-States" else x)
# Bu sütunu daha sonra merge örneği için kullanabilmek adına kodun ilerleyen aşamasında etiketlemeden önce bir kopyasını alıyoruz:
X["native-country"] = lb.fit_transform(X["native-country"])

# relationship sütunu: bazı ilişkileri gruplandırma ve etiketleme
X["relationship"] = X["relationship"].replace(
    {
        "Husband": "Married",
        "Wife": "Married",
        "Own-child": "Child",
        "Not-in-family": "Independent",
        "Unmarried": "Single",
        "Other-relative": "Other"
    }
)
X["relationship"] = lb.fit_transform(X["relationship"])

# fnlwgt sütununu modelleme açısından gereksiz gördüğümüz için kaldırıyoruz
X.drop("fnlwgt", axis=1, inplace=True)

# race sütunu: sadece White ve Black olarak ikili gruplandırma
X["race"] = X["race"].replace(
    {
        "White": 1,
        "Black": 0,
    }
)
# Önce sayısal dönüşüm yapılmadan log dönüşümü uygulanması
X["capital-gain"] = np.log1p(X["capital-gain"])  
X["capital-loss"] = np.log1p(X["capital-loss"])

# race sütunu için tekrar uygulama (gerekirse)
X["race"] = X["race"].apply(lambda x: 1 if x == 1 else 0)

# age sütunu normalizasyonu
X["age"] = scaler.fit_transform(X[["age"]])

# Diğer kategorik sütunlarda da normalizasyon (sayısal hale getirilmiş etiketlerin ölçeklendirilmesi)
X["workclass"] = scaler.fit_transform(X[["workclass"]])
X["education"] = scaler.fit_transform(X[["education"]])
X["occupation"] = scaler.fit_transform(X[["occupation"]])
X["relationship"] = scaler.fit_transform(X[["relationship"]])

# 3. Veri Keşfi (EDA) ve Görselleştirme

# Özellikler (X) ve hedef (y) birleştirilerek tek DataFrame oluşturuluyor
df = pd.concat([X, y], axis=1)

print("\nVeri Setinin İlk 5 Satırı:")
print(df.head())

print("\nVeri Seti Hakkında Bilgi:")
print(df.info())

print("\nEksik Değerler:")
print(df.isnull().sum())

print("\nSayısal Değişkenlerin İstatistikleri:")
print(df.describe())
# Kategorik değişkenlerin dağılımlarını gözlemleyelim
print("\nKategorik Değişkenler:")
for col in df.select_dtypes(include=['object']).columns:
    print(f"\n{col} sütunu dağılımı:")
    print(df[col].value_counts())

# Görselleştirme örnekleri
sns.set(style="whitegrid")

# 3.1. Gelir Dağılımı
plt.figure()
sns.countplot(x="income", data=df)
plt.title("Gelir Dağılımı (<=50K vs >50K)")
plt.xlabel("Gelir (0: <=50K, 1: >50K)")
plt.ylabel("Sayı")
plt.show()

# 3.2. Yaş Dağılımı (Normalleştirilmiş olsa da genel dağılım incelenebilir)
plt.figure()
sns.histplot(df["age"], bins=30, kde=True)
plt.title("Yaş Dağılımı (Normalized)")
plt.xlabel("Normalized Age")
plt.show()

# 3.3. Eğitim ve Yaş İlişkisi (Boxplot)
plt.figure(figsize=(10,6))
sns.boxplot(x="education", y="age", hue="income", data=df)
plt.title("Eğitim ve Yaş İlişkisi (Income Bazında)")
plt.xlabel("Normalized Education Level")
plt.ylabel("Normalized Age")
plt.show()

# 3.4. GroupBy Örneği: Workclass Bazında Ortalama Çalışma Saatleri
print("\nWorkclass Bazında Ortalama 'hours-per-week':")
print(df.groupby("workclass")["hours-per-week"].mean())

# 3.5. Filtering Örneği: Yüksek çalışma saatleri ve yüksek gelir
# (hours-per-week 0.9'un üzeri, income 1 olan bireyler)
filtered_df = df.query("`hours-per-week` > 0.9 and income == 1")
print("\nYüksek çalışma saatlerine sahip (normalized > 0.9) ve yüksek gelirli bireyler:")
print(filtered_df.head())

# 3.6. Merge/Join Örneği: Dummy country bilgisi ekleme
# native-country sütununu etiketlemeden önceki kopyasını kullanarak, basit bir bölge bilgisi ekliyoruz.
# LabelEncoder uygulandıktan sonra native-country sütununda 0: Other, 1: United-States varsayılmaktadır.
country_info = pd.DataFrame({
    "native_country_encoded": [0, 1],
    "region": ["Other Region", "North America"]
})
# df içinde native-country sütunu, etiketlenmiş haliyle mevcut
df['native_country_encoded'] = df["native-country"]
# Merge işlemi
df = df.merge(country_info, left_on="native_country_encoded", right_on="native_country_encoded", how="left")
print("\nMerge Sonucu - İlk 5 Satır:")
print(df.head())

# 4. Makine Öğrenmesi Modelleme

# Modelleme için eğitim ve test setlerinin oluşturulması
y = y.values.ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# GridSearch için parametre ızgarası oluşturma
param_grid = {
    'penalty': ['l1', 'l2'],
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'saga']
}

grid = GridSearchCV(
    LogisticRegression(class_weight="balanced", max_iter=500),
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

# Model eğitimi
grid.fit(X_train, y_train)
best_model = grid.best_estimator_

# Test seti üzerinde tahminler
# Karar fonksiyonundan elde edilen skorlar belirli bir eşik değerinde sınıflandırılıyor
y_pred = (best_model.decision_function(X_test) > 0.63).astype(int)

# Performans metriklerini yazdırma
print(f"\nEn iyi parametreler: {grid.best_params_}")
print(f"En iyi doğruluk (Train): {grid.best_score_:.4f}")
print(f"Test Set Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}\n")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Precision-Recall eğrisi çizimi 
y_scores = best_model.decision_function(X_test)
precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores)
plt.figure()
plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision-Recall Curve")
plt.legend(loc="best")
plt.show()
