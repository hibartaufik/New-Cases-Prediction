# Project 3: New Cases Prediction [COVID-19 Indonesia Dataset (Machine Learning)]
Overview:
1. Membuat model machine learning menggunakan metode decision tree berdasarkan dataset yang berisi data COVID-19 di Indonesia
2. Analysis dan Visualisasi yang lebih lengkap dapat dilihat di Project 2 dengan dataset yang sama, project ini lebih berfokus pada pemodelan machine learning
3. Dataset berasal dari [kaggle.com](https://www.kaggle.com/) dengan nama COVID-19 Indonesia Dataset yang disusun oleh Hendratno yang mengambil data dari beberapa sumber yaitu [situs resmi pemerintah SATGAS COVID-19](https://covid19.go.id/), [Badan Pusat Statistik](https://www.bps.go.id/), dan [Hub InaCOVID-19](https://bnpb-inacovid19.hub.arcgis.com/)
4. Dataset disusun berdasarkan time series atau sususan waktu, tingkat nasional, dan tingkat provinsi, juga beserta data demografi dari lokasi/daerah tersebut
5. Dataset memiliki kolom
   - **'Date'** (Tanggal dilaporkan)
   - **'Location ISO Code'** (Kode lokasi berdasarkan standar ISO)
   - **'Location'** (Nama lokasi)
   - **'New Cases'** (Kasus positif harian)
   - **'New Deaths'** (Kasus kematian harian)
   - **'New Daily Recovered'** (Kasus kesembuhan harian)
   - **'New Active Cases'** (Kasus aktif harian)
   - **'Total Cases'** (Jumlah akumulatif kasus positif sampai waktu terkait)
   - **'Total Deaths'** (Jumlah akumulatif kasus kematian sampai waktu terkait)
   - **'Total Recovered'** (Jumlah akumulatif kasus kesembuhan sampai waktu terkait)
   - **'Total Active Cases'** (Jumlah akumulatif kasus aktif sampai waktu terkait)
   - **'Location Level'** (Tingkat lokasi regional atau nasional)
   - **'City or Regency'** (Nama kota atau wilayah)
   - **'Province'** (Nama provinsi lokasi)
   - **'Country'** (Nama negara lokasi)
   - **'Island'** (Nama pulau utama lokasi)
   - **'Time Zone'** (Zona waktu lokasi)
   - **'Special Status'** (Status istimewa lokasi)
   - **'Total Regencies'** (Jumlah kabupaten dalam lokasi terkait)
   - **'Total Cities'** (Jumlah kota dalam lokasi terkait)
   - **'Total Districts'** (Jumlah kecamatan dalam lokasi terkait)
   - **'Total Urban Village'** (Jumlah pedesaan dalam lokasi terkait)
   - **'Total Rural Village'** (Jumlah perkampungan dalam lokasi terkait)
   - **'Area (km2)'** (Area lokasi dalam kilometer persegi)
   - **'Population'** (Jumlah populasi dalam lokasi terkait)
   - **'Population Density'** (Kepadatan penduduk dalam lokasi terkait, rumus = Population / Area)
   - **'Longitude'** (Garis bujur lokasi)
   - **'Latitude'** (Garis lintang lokasi)
   - **'New Cases per Million'** (Rumus = (New Cases / Population) x 1.000.000)
   - **'Total Cases per Million'** (Rumus = (Total Cases / Population) x 1.000.000)
   - **'Total Deaths per Million'** (Rumus = (Total Deaths / Population) x 1.000.000)
   - **'Case Fatality Rate'** (Rumus = (Total Deaths / Total Cases) x 100)
   - **'Case Recovered Rate'** (Rumus = (Total Recovered / Total Cases) x 100)
   - **'Growth Factor of New Cases'** (Kurang dari 1 artinya menurun, 1 artinya tidak ada perubahan, lebih dari 1 artinya meningkat, rumus = Today New Cases / Yesterday New Cases)
   - **'Growth Factor of New Deaths'** (Kurang dari 1 artinya menurun, 1 artinya tidak ada perubahan, lebih dari 1 artinya meningkat, rumus = Today New Deaths / Yesterday New Deaths)
6. Tahapan membuat model machine learning terbagi ke dalam 6 tahap, yaitu:
   - Data Preparation
   - Exploratory Data Analysis
   - Data Preprocessing
   - Modeling
   - Model Evaluation
   - Predict Test Data
   
   Tahapan di atas merupakan acuan yang digunakan untuk membuat model Machine Learning, tahapan tidak baku, dapat disesuaikan berdasarkan karakteristik data dan studi kasus
7. Project menggunakan dataset berasal [kaggle](https://www.kaggle.com/), yang disusun oleh Hendratno.
   - Repository project Github dapat diakses [disini](https://github.com/hibartaufik/New-Cases-Prediction)
   - Dataset dapat diakses [disini](https://www.kaggle.com/hendratno/covid19-indonesia)

## 1. Data Preparation
### 1.1 Import Libraries
```
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
```
### 1.2 Import Dataset
Import dataset yang akan digunakan, lalu tampilkan dataset untuk mengecek apakah import berhasil atau tidak
```
df = pd.read_csv('./dataset/covid_19_indonesia_time_series_all.csv')
df = df.set_index('Location')
```
```
df.head()
```
![image](https://user-images.githubusercontent.com/74480780/126043463-4123b1dc-4d68-40a9-a012-cf87de2ac7e8.png)

## 2. Exploratory Data Analysis (EDA)
Menganalisa karakteristik data dengan perintah `info()`, `describe()`, `shape`, dan beberapa perintah lainnya agar menemukan insight yang dapat berguna dalam pengolahan data dan perancangan model machine learning. Lalu, mencatat segala macam penemuan pada dataset seperti data yang kosong, tidak lengkap, redundant, atau data yang perlu pengolahan lebih lanjut. Hal-hal yang sudah dicatat tersebut akan diolah dan dieksekusi pada tahapan Data Preprocessing.
```
# Melihat jumlah tipe data tiap kolom dalam dataset
df.info()
```
![image](https://user-images.githubusercontent.com/74480780/126043555-b7b82782-3302-4832-a8e0-afd50c7ba05a.png)
![image](https://user-images.githubusercontent.com/74480780/126043559-ae17dde2-b79c-4a48-bf5e-b6700e2d9aab.png)
Dataset memiliki 37 kolom, terdiri dari 12 kolom dengan tipe data object/string, 12 kolom dengan tipe data integer, dan 13 kolom dengan tipe data float.
```
# Melihat summary statistics
df.describe()
```
![image](https://user-images.githubusercontent.com/74480780/126043586-20ed44c0-ec66-47d4-bd93-c74b16854ef6.png)
```
# Melihat dimensi (baris, kolom) dataset
df.shape
```
![image](https://user-images.githubusercontent.com/74480780/126043594-8d9169d1-2982-417f-ac0b-0f16a1759ff7.png)

### 2.1 Menentukan Kolom yang Akan di-Imputasi
Melihat data null tiap kolom dalam bentuk presentase beserta tipe datanya
```
columns = list(df.columns)

for col in columns:
  print(f"{col}:{round((df[col].isnull().sum() / len(df) * 100), 3)}%\t\t{df[col].dtype}")
```
![image](https://user-images.githubusercontent.com/74480780/126043621-28a2c62a-225a-4c0b-b0bb-5fc14330b415.png)
![image](https://user-images.githubusercontent.com/74480780/126043633-9107a780-ea62-40b2-9852-02853be22e6d.png)
Membuat looping untuk memfilter setiap kolom dalam dataset dan mengelompokannya ke dalam beberapa list untuk menentukan kolom mana saja yang akan diimputasi (kolom yang memiliki jumlah data null < 40%) dan kolom mana yang harus di drop (kolom yang memiliki jumlah data null > 40%) nantinya.

- `cols_null_num` untuk kolom yang memiliki jumlah data null < 40% dan bertipe data numerik (integer & float)
- `cols_null_obj` untuk kolom yang memiliki jumlah data null < 40% dan bertipe data object (string)
- `cols_drop` untuk kolom yang memiliki jumlah data null > 40%

```
cols_null_num = []
cols_null_obj = []
cols_drop = []

for col in columns:
  if ((df[col].isnull().sum() / len(df) * 100) != 0) and ((df[col].isnull().sum() / len(df) * 100) < 40) and (df[col].dtype != 'object'):
    cols_null_num.append(col)
  elif ((df[col].isnull().sum() / len(df) * 100) != 0) and ((df[col].isnull().sum() / len(df) * 100) < 40) and (df[col].dtype == 'object'):
    cols_null_obj.append(col)
  elif ((df[col].isnull().sum() / len(df) * 100) != 0) and ((df[col].isnull().sum() / len(df) * 100) > 40):
    cols_drop.append(col)
```
```
print(f"Kolom numerik: {cols_null_num}")
print(f"Kolom object: {cols_null_obj}")
```
![image](https://user-images.githubusercontent.com/74480780/126043678-b3b09949-6d2d-4ed0-958a-ab90efa31ab4.png)
Kolom yang akan diimputasi:
- Kolom numerik: 'Total Cities', 'Total Urban Villages', 'Total Rural Villages', 'Growth Factor of New Cases', 'Growth Factor of New Deaths'
- Kolom object: 'Province', 'Island', 'Time Zone'

#### 2.1.1 Melihat Distribusi Data Untuk Menetukan Nilai Imputasi Tiap Kolom
- Kolom Numerik
```
for col in cols_null_num:
  sns.displot(df[col].value_counts(), kde=True)
```
![image](https://user-images.githubusercontent.com/74480780/126043744-768d7eca-f1e5-4740-bbb8-71b12772c6fb.png)
![image](https://user-images.githubusercontent.com/74480780/126043773-75b9ca42-6323-42e3-9b24-51fae1176e2b.png)
![image](https://user-images.githubusercontent.com/74480780/126043803-59567879-b64b-45ba-b767-36c5fad6071b.png)
![image](https://user-images.githubusercontent.com/74480780/126043826-292d94c3-e2cb-4eb0-87f1-2baf97acedbf.png)
![image](https://user-images.githubusercontent.com/74480780/126043839-477ab33d-3093-4ea8-b599-b80cd0bf2090.png)
Kita dapat menetukan nilai yang akan mengganti nilai kosong berdasarkan distribusi data tiap kolom. Untuk 'Total Rural Villages' memiliiki data berdistrbusi 'nyaris' normal, nilai modus merupakan pilihan tepat karena data yang bernilai bulat, mengapa tidak mean? karena nilai mean tidak terlalu bisa mewakili distribusi data yang tidak benar-benar normal/merata.

Untuk kolom numerik lain seperti 'Total Cities', 'Total Urban Villages', 'Growth Factor of New Cases', dan 'Growth Factor of New Deaths' memiliki data yang sama sekali tidak merata, sehingga nilai median merupakan nilai yang cocok untuk dapat mengisi data null si setiap kolom-nya.

- Kolom Object
```
for col in cols_null_obj:
  sns.displot(df[col].value_counts(), kde=True)
```
![image](https://user-images.githubusercontent.com/74480780/126043894-8f706f6a-074a-413d-9c4b-b82b4c1cecd8.png)
![image](https://user-images.githubusercontent.com/74480780/126043907-6fb440b6-a003-44b1-9058-d02306bc414a.png)
![image](https://user-images.githubusercontent.com/74480780/126043926-956818ae-9d8a-45ff-8cbb-a5002e816701.png)
Data null pada 'Province', 'Island', dan 'Time Zone' kurang baik jika diisi dengan modus data karena tidak bisa merepresentasikan data kolom terkait seperti provinsi atau pulau saat kasus COVID-19 ditemukan tidak akan dapat direpresentasikan dengan nama provinsi atau pulau yang paling banyak datanya.

Namun solusi drop kolom juga kurang baik karena data null memiliki komposisi yang sedikit. Yang akan dilakukan adalah mengisi nilai null tersebut dengan angka nol, meskipun nilai nol tersebut diisi pada kolom yang bertipe data object (string), namun pada akhirnya kolom tersebut juga akan diubah ke dalam bentuk numerik sehingga pengisian nilai nol tidak akan bermasalah.

### 2.2 Menentukan Kolom yang Akan di Drop
```
cols_drop
```
![image](https://user-images.githubusercontent.com/74480780/126043955-1c7f85ba-e6f9-478c-9864-626241242958.png)
Kolom yang akan di drop adalah kolom yang memiliki komposisi data null yang banyak (sekitar lebih dari 40% keseluruhan data kolom tersebut). Seluruh data kolom 'City or Regency' adalah null, sedangkan untuk kolom 'Special Status' memiliki 85.569% data null.

## 3. Data Preprocessing

Beberapa hal yang didapatkan dari proses EDA adalah:
- Handling Missing Value:
  
  - Imputasi: 
      
      - 'Total Rural Villages' -> mod
      - 'Total Urban Villages' -> median
      - 'Total Cities' -> median
      - 'Growth Factor of New Cases' -> median 
      - 'Growth Factor of New Deaths' -> median
      - 'Province' -> 0 
      - 'Island' -> 0
      - 'Time Zone' -> 0
  
  - Drop:

    - 'City or Regency'
    - 'Special Status'

- Mengubah Kolom Kategori Menjadi Numerik
### 3.1 Imputasi
#### 3.1.1 Imputasi Kolom Numerik
```
# Membuat dictionary untuk menampung nilai pengisi
filler_num = {
    'Total Rural Villages': stats.mode(df['Total Rural Villages'])[0][0],
    'Total Urban Villages': df['Total Urban Villages'].median(),
    'Total Cities': df['Total Cities'].median(),
    'Growth Factor of New Cases': df['Growth Factor of New Cases'].median(),
    'Growth Factor of New Deaths': df['Growth Factor of New Deaths'].median()
}
```
```
df.fillna(filler_num, inplace=True)
```
#### 3.1.2 Imputasi Kolom Kategorik
Mengisi data null di kolom 'Province', 'Island', 'Time Zone' dengan nilai nol
```
df[cols_null_obj] = df[cols_null_obj].replace([np.inf, -np.inf], np.nan)
df[cols_null_obj] = df[cols_null_obj].fillna(0)
```
### 3.2 Drop Kolom
Drop feature 'City or Regency' dan 'Special Status'
```
df.drop(columns=cols_drop, inplace=True)
```
Mengecek perubahan yang dilakukan
```
df.isnull().sum()
```
![image](https://user-images.githubusercontent.com/74480780/126044064-6592ccea-edd7-453e-929c-b947f3fc7c3a.png)
![image](https://user-images.githubusercontent.com/74480780/126044075-8735d61e-e17d-422f-8b59-ccca07f335c8.png)

### 3.3 Ubah Tipe Data Kolom Kategori Menjadi Numerik

Disini kita akan menggunakan teknik Label Encoding. Meskipun hampir semua kolom object dalam dataset tidak bersifat ordinal, namun jumlah kolom kategorik terlalu banyak sehingga akan terbentuk lebih banyak kolom saat kita melakukan Encoding dengan One-Hot Encoding yang tentu tidak efisien untuk dataset. Dengan pertimbangan tersebut, Label Encoding merupakan pilihan yang cocok.
```
# Membuat fungsi untuk mengubah kolom dengan metode Label Encoding
le = LabelEncoder()

def label_encoder(df):
  for col in df.columns:
    if df.dtypes[col] == 'object':
      le.fit(df[col].astype(str))
      df[col] = le.transform(df[col].astype(str))
  return df
```
```
# Terapkan fungsi pada dataset, lalu cek tipe data tiap kolom untuk melihat perubahannya
df = label_encoder(df)
df.info()
```
![image](https://user-images.githubusercontent.com/74480780/126044116-a3edabc7-f028-43ea-ade4-780c5270743d.png)
![image](https://user-images.githubusercontent.com/74480780/126044124-f57a6f9a-5cfc-4483-80f5-1b41efe38671.png)

## 4. Modeling
Membuat model machine learning yang akan memprediksi angka 'New Cases' dengan metode Decision Tree
### 4.1 Splitting Data
Melakukan splitting data dengan fungsi `train_test_split()` dari library Scikit-learn
```
X = df.drop(columns=['New Cases'])
y = df['New Cases']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
```
```
print(f"X train: {X_train.shape}")
print(f"y train: {y_train.shape}")
print(f"X test: {X_test.shape}")
print(f"y test: {y_test.shape}")
```
![image](https://user-images.githubusercontent.com/74480780/126044197-db25ef1c-f879-4551-82e7-9e586e393654.png)

### 4.2 Training Model
```
model_dt = DecisionTreeClassifier().fit(X_train, y_train)
```
```
# cek akurasi
score_model_dt= model_dt.score(X_test, y_test)
print(f"Akurasi Decision Tree Model:\t{round(score_model_dt * 100, 3)}%")
```
![image](https://user-images.githubusercontent.com/74480780/126044233-dbd017fe-8e9e-45ea-9e0c-bba9a598796c.png)

## 5 Model Evaluation
### 5.1 Improve Model with Feature Scaling
#### 5.1.1 Melakukan Feature scaling / Normalisasi Data dengan Fungsi `MinMaxScaler()` dari Library Skicit-learn
```
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```
```
# Ubah data yang telah dinormalisasi ke dalam bentuk Dataframe
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
```
![image](https://user-images.githubusercontent.com/74480780/126044266-76f220b5-33b9-4565-9e25-5d1fb0ebc724.png)
#### 5.1.2 Melakukan Pemodelan Kembali dengan Data yang Sudah di Normalisasi
```
model_dt = DecisionTreeClassifier().fit(X_train_scaled, y_train)
```
```
score_scaled_model_dt = model_dt.score(X_test_scaled, y_test)
print(f"Akurasi Decision Tree Model yang Sudah di Normalisasi:\t{round(score_scaled_model_dt * 100, 3)}%")
```
![image](https://user-images.githubusercontent.com/74480780/126044395-4dd032a9-581f-46cb-8eed-5415ae4cf107.png)
Akurasi model dengan data yang telah dinormalisasi meningkat meskipun hanya berselisih kurang lebih satu angka dengan akurasi model sebelumnya.

## 6. Predict Test Data
### 6.1 Melakukan Prediksi Dengan Data Test
```
Y_predict = model_dt.predict(X_test_scaled)
```
```
Y_predict
```
![image](https://user-images.githubusercontent.com/74480780/126044492-12e9de2d-7194-4095-b300-b5b1e40738f6.png)

### 6.2 Mengubah Data Hasil Prediksi ke dalam Bentuk Dataframe
```
new_cases_prediction = pd.DataFrame({'New Cases': Y_predict})
```
```
new_cases_prediction.head(10)
```
![image](https://user-images.githubusercontent.com/74480780/126044517-c0601052-1244-468e-9ab8-186a33c57d62.png)

### 6.3 Eksport DataFrame ke dalam bentuk file csv
```
filename = 'new_cases_prediction.csv'

new_cases_prediction.to_csv(filename, index=False)
print(f"File '{filename}' has been exported")
```
![image](https://user-images.githubusercontent.com/74480780/126044540-74bb1505-adff-413b-b903-b5f3ae5fd7bf.png)
