# Prediksi Hujan di Jakarta Berdasarkan Integrasi Data Cuaca, Kualitas Udara, dan Kondisi Laut Menggunakan Algoritma Random Forest dan Gradient Boosting

## Deskripsi Proyek

Project ini bertujuan untuk menganalisis kondisi lingkungan di Jakarta serta membangun model klasifikasi untuk memprediksi kondisi hujan atau tidak hujan berdasarkan integrasi tiga jenis data, yaitu data cuaca, kualitas udara, dan kondisi laut.

Proses ETL dilakukan menggunakan Apache Airflow. Data hasil ETL disimpan ke dalam database MySQL dalam tiga tabel terpisah, kemudian dianalisis lebih lanjut menggunakan Google Colab.

Model klasifikasi yang digunakan dalam project ini adalah Random Forest dan Gradient Boosting. Kedua model dibandingkan berdasarkan hasil evaluasi untuk menentukan model yang memberikan performa terbaik.

## Tujuan Proyek

Project ini bertujuan untuk:

1. Menganalisis karakteristik umum kondisi cuaca di Jakarta.
2. Menganalisis hubungan antara kondisi cuaca dan kualitas udara.
3. Menganalisis hubungan antara suhu permukaan laut dan kondisi cuaca.
4. Mengintegrasikan tiga dataset berdasarkan waktu pengamatan.
5. Membangun model klasifikasi hujan dan tidak hujan.
6. Membandingkan performa algoritma Random Forest dan Gradient Boosting.

## Pertanyaan Bisnis

1. Bagaimana karakteristik umum kondisi cuaca di Jakarta selama periode pengamatan berdasarkan suhu, kelembapan, curah hujan, kecepatan angin, tekanan udara, dan tutupan awan?

2. Bagaimana hubungan antara kondisi cuaca dengan kualitas udara di Jakarta, khususnya terhadap konsentrasi PM2.5, PM10, nitrogen dioksida, dan ozon?

3. Bagaimana hubungan suhu permukaan laut dengan kondisi cuaca di Jakarta, terutama suhu udara, kelembapan, tutupan awan, dan curah hujan?

4. Seberapa baik algoritma Random Forest dan Gradient Boosting dalam memprediksi kondisi hujan atau tidak hujan berdasarkan integrasi data cuaca, kualitas udara, dan kondisi laut, serta algoritma mana yang memberikan performa terbaik?

## Dataset

Project menggunakan tiga dataset dengan resolusi data per jam.

### 1. Weather Data

Dataset utama yang berisi:

- `time`
- `temperature_2m`
- `relative_humidity_2m`
- `precipitation`
- `wind_speed_10m`
- `surface_pressure`
- `cloud_cover`

### 2. Air Quality Data

Dataset pendukung pertama yang berisi:

- `time`
- `pm2_5`
- `pm10`
- `nitrogen_dioxide`
- `ozone`

### 3. Marine Data

Dataset pendukung kedua yang berisi:

- `time`
- `wave_height`
- `wave_period`
- `sea_surface_temperature`

Pada proses analisis, kolom `wave_height` dan `wave_period` tidak digunakan karena seluruh nilainya kosong. Variabel kondisi laut yang digunakan adalah `sea_surface_temperature`.

Ketiga dataset diintegrasikan berdasarkan kolom `time`.

## Alur Proyek

```text
Raw JSON
   ↓
Apache Airflow
   ↓
MySQL Database
   ↓
Google Colab
   ↓
Data Gathering
   ↓
Data Assessing
   ↓
Data Cleaning
   ↓
Re-Assessing
   ↓
Data Integration / Merging
   ↓
Exploratory Data Analysis
   ↓
Visualisasi
   ↓
Random Forest & Gradient Boosting
   ↓
Evaluasi Model
   ↓
Conclusion