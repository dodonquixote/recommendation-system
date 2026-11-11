# Laporan Proyek Machine Learning - Salman Alfarizi Taha

## Project Overview

Dengan semakin berkembangnya platform streaming seperti Netflix, Disney+, dan Amazon Prime, pengguna kini dihadapkan pada ribuan pilihan film. Namun, banyaknya pilihan ini justru menyulitkan pengguna dalam menemukan film yang sesuai dengan preferensi mereka. Sistem rekomendasi hadir untuk menyaring informasi ini dan membantu pengguna dalam pengambilan keputusan.

Dalam proyek ini, dibangun sistem rekomendasi film menggunakan pendekatan **Content-Based Filtering** yang mengandalkan atribut konten seperti genre, keyword, pemeran utama, dan sutradara untuk menghitung kemiripan antar film dan memberikan rekomendasi yang relevan. Pendekatan ini tidak membutuhkan data interaksi pengguna sehingga cocok digunakan untuk *cold-start users*.

### Referensi:
- Lops, P., Gemmis, M. D., & Semeraro, G. (2011). *Content-based Recommender Systems: State of the Art and Trends*. In Recommender Systems Handbook. Springer.
- [TMDB 5000 Dataset - Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

---

## Business Understanding

### Problem Statements
1. Bagaimana membantu pengguna menemukan film yang sesuai dengan preferensi mereka berdasarkan konten film?
2. Bagaimana mengukur kemiripan antar film menggunakan fitur konten seperti genre, cast, sutradara, dan keyword?
3. Bagaimana membangun sistem yang dapat memberikan rekomendasi yang relevan dan beragam?

### Goals
1. Membangun sistem rekomendasi berbasis konten dengan akurasi tinggi.
2. Menghasilkan rekomendasi yang relevan dengan skor kemiripan > 0.1.
3. Mencapai precision minimal 70% untuk film dengan rating di atas 7.0.

### Solution Statements
- **Pendekatan 1:** Content-Based Filtering menggunakan TF-IDF Vectorizer dan Cosine Similarity.
- **Pendekatan 2 (Alternatif):** Count Vectorizer sebagai pembanding performa model.

---

## Data Understanding

Dataset yang digunakan adalah **TMDB 5000 Movies Dataset** dari Kaggle, yang terdiri dari:

- `tmdb_5000_movies.csv` - metadata film (genre, overview, rating, dll)
- `tmdb_5000_credits.csv` - informasi cast dan crew film

**Jumlah film**: 4803  
**Sumber**: [https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

### Fitur utama:
- `genres`: kategori film
- `keywords`: kata kunci tema film
- `cast`: pemeran utama (top 3)
- `crew`: direktur film
- `vote_average`, `vote_count`: evaluasi popularitas dan kualitas

### Visualisasi:
- Distribusi rating film
- Tahun rilis film
- Korelasi runtime dengan rating
- Top 10 film dengan rating tertinggi

---

## Data Preparation

1. **Merging**: Gabungkan `movies_df` dan `credits_df` menggunakan `id`.
2. **Parsing JSON**: Ekstrak list dari kolom `genres`, `keywords`, `cast`, dan `crew`.
3. **Cleaning Text**: Lowercase dan hapus spasi antar kata.
4. **Content Soup**: Gabungkan semua fitur menjadi satu string sebagai basis input vektorisasi.

**Alasan**: Agar representasi tekstual bisa diekstrak menggunakan TF-IDF dan digunakan dalam penghitungan kemiripan antar film.

---

## Modeling

Sistem rekomendasi dibangun menggunakan langkah berikut:

- **TF-IDF Vectorization** untuk mentransformasikan content soup.
- **Cosine Similarity** untuk mengukur kemiripan antar film.
- Mapping index judul film untuk pencarian efisien.

### Output Contoh
Top-5 rekomendasi untuk film *Avatar*:
1. *Star Trek Into Darkness* – Similarity: 0.276
2. *Aliens*
3. *Clash of the Titans*
4. *Wrath of the Titans*
5. *Treasure Planet*

### Kelebihan:
- Tidak membutuhkan interaksi pengguna
- Explainable dan transparan

### Kekurangan:
- Rentan over-specialization
- Tidak personal untuk setiap pengguna

---

## Evaluation

### Metrik Evaluasi
Evaluasi sistem dilakukan menggunakan tiga metrik utama:
- **Precision@K**: Mengukur proporsi rekomendasi dengan rating ≥ 7.0.
- **Diversity**: Mengukur keragaman genre dari film-film rekomendasi.
- **Novelty**: Mengukur kebaruan film dari jumlah vote yang relatif kecil.

### Hasil Evaluasi (Contoh 5 Film Uji)

| Film             | Precision@K | Diversity | Novelty | Avg Similarity | Avg Rating |
|------------------|--------------|-----------|---------|----------------|-------------|
| Avatar           | 0.40         | 0.29      | 0.87    | 0.18           | 6.45        |
| The Dark Knight  | 0.40         | 0.23      | 0.77    | 0.31           | 6.38        |
| Titanic          | 0.10         | 0.17      | 0.98    | 0.17           | 6.24        |
| Toy Story        | 0.40         | 0.23      | 0.87    | 0.27           | 6.48        |
| The Godfather    | 0.20         | 0.28      | 0.96    | 0.24           | 5.80        |

### Rata-rata Performa Sistem
- **Average Precision@K**: 0.30
- **Average Diversity**: 0.24
- **Average Novelty**: 0.89
- **Average Similarity**: 0.23
- **Average Rating**: 6.3

---

## Interactive Interface

Sistem menyediakan fungsi interaktif berbasis terminal untuk mencoba rekomendasi film secara langsung. Pengguna dapat:

1. Memasukkan judul film.
2. Menentukan jumlah rekomendasi (default: 5).
3. Menentukan batas minimum similarity (default: 0.1).

Hasil yang ditampilkan mencakup:
- Judul film
- Tahun rilis
- Skor kemiripan
- Rating dan genre

Fungsi ini memudahkan pengguna dalam mengeksplorasi sistem secara langsung dan personal.

---
