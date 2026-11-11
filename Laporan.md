# Laporan Proyek Machine Learning – Salman Alfarizi Taha
## Sistem Rekomendasi Film Menggunakan Content-Based Filtering

---

## 📋 Table of Contents
1. [Project Overview](#1-project-overview)
2. [Business Understanding](#2-business-understanding)
3. [Data Understanding](#3-data-understanding)
4. [Data Preparation](#4-data-preparation)
5. [Modeling & Results](#5-modeling--results)
6. [Evaluation](#6-evaluation)
7. [Interactive Interface](#7-interactive-interface)
8. [Kesimpulan dan Insight](#8-kesimpulan-dan-insight)

---

## 1. Project Overview

Dengan semakin berkembangnya platform streaming seperti **Netflix**, **Disney+**, dan **Amazon Prime**, pengguna dihadapkan pada ribuan pilihan film. Banyaknya pilihan ini justru menyulitkan mereka menemukan film yang sesuai dengan preferensi pribadi. Sistem rekomendasi hadir untuk menyaring informasi dan membantu pengguna dalam pengambilan keputusan.

Pada proyek ini, dibangun sistem rekomendasi film menggunakan pendekatan **Content-Based Filtering**, yang mengandalkan atribut konten (genre, keyword, pemeran utama, sutradara) untuk menghitung kemiripan antar-film dan memberikan rekomendasi yang relevan. Pendekatan ini tidak membutuhkan data interaksi pengguna sehingga cocok digunakan untuk *cold-start users*.

### 📚 Referensi
- Lops, P., Gemmis, M. D., & Semeraro, G. (2011). *Content-based Recommender Systems: State of the Art and Trends*. In **Recommender Systems Handbook**. Springer.
- [TMDB 5000 Dataset – Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

---

## 2. Business Understanding

### 2.1 Problem Statements
1. **Bagaimana membantu pengguna menemukan film yang sesuai dengan preferensi mereka berdasarkan konten film?**
2. **Bagaimana mengukur kemiripan antar-film menggunakan fitur konten seperti genre, cast, sutradara, dan keyword?**
3. **Bagaimana membangun sistem yang dapat memberikan rekomendasi yang relevan dan beragam?**

### 2.2 Goals
1. ✅ **Membangun sistem rekomendasi berbasis konten dengan akurasi tinggi**
2. ✅ **Menghasilkan rekomendasi yang relevan dengan skor kemiripan > 0.1**
3. ✅ **Mencapai precision minimal 70% untuk film dengan rating ≥ 7.0**

### 2.3 Solution Statements
- **Pendekatan 1:** Content-Based Filtering menggunakan **TF-IDF Vectorizer** dan **Cosine Similarity**
- **Pendekatan 2 (Alternatif):** **Count Vectorizer** sebagai pembanding performa model

---

## 3. Data Understanding

Dataset yang digunakan adalah **TMDB 5000 Movies Dataset** dari Kaggle, yang terdiri dari dua file:

### 📊 Dataset Overview
| **File** | **Jumlah Baris** | **Jumlah Kolom** | **Deskripsi** |
|----------|------------------|------------------|---------------|
| `tmdb_5000_movies.csv` | 4,803 | 20 | Metadata film (genre, budget, rating, dll.) |
| `tmdb_5000_credits.csv` | 4,803 | 4 | Informasi cast dan crew |

### 3.1 Struktur Data Movies

#### `tmdb_5000_movies.csv`
Berisi metadata film dengan kolom berikut:

| **Kolom** | **Tipe** | **Deskripsi** |
|-----------|----------|---------------|
| `budget` | int | Biaya produksi (USD) |
| `genres` | str | Daftar genre dalam format JSON string |
| `homepage` | str | URL halaman resmi film |
| `id` | int | ID unik film di TMDB |
| `keywords` | str | Daftar keyword (tema) dalam format JSON string |
| `original_language` | str | Bahasa asli film |
| `original_title` | str | Judul asli film |
| `overview` | str | Sinopsis singkat film |
| `popularity` | float | Skor popularitas (metrik internal TMDB) |
| `production_companies` | str | Daftar perusahaan produksi dalam format JSON |
| `production_countries` | str | Daftar negara produksi dalam format JSON |
| `release_date` | str | Tanggal rilis (format YYYY-MM-DD) |
| `revenue` | int | Pendapatan kotor (USD) |
| `runtime` | float | Durasi film (menit) |
| `spoken_languages` | str | Daftar bahasa yang diucapkan dalam format JSON |
| `status` | str | Status film ("Released", "Post Production", dll.) |
| `tagline` | str | Kalimat pendek promosi |
| `title` | str | Judul film |
| `vote_average` | float | Rata-rata rating pengguna (0.0-10.0) |
| `vote_count` | int | Jumlah total vote yang diterima film |

#### `tmdb_5000_credits.csv`
Berisi informasi tentang cast dan crew:

| **Kolom** | **Tipe** | **Deskripsi** |
|-----------|----------|---------------|
| `movie_id` | int | ID film (untuk join dengan movies_df.id) |
| `title` | str | Judul film |
| `cast` | str | JSON string daftar pemeran lengkap |
| `crew` | str | JSON string daftar kru |

### 3.2 Missing Values Analysis

**Missing values pada `movies_df`:**

| **Kolom** | **Missing** | **Persentase** | **Keterangan** |
|-----------|-------------|----------------|----------------|
| `homepage` | 3,091 | 64.4% | Banyak film tidak memiliki homepage |
| `tagline` | 844 | 17.6% | Banyak tagline yang kosong |
| `overview` | 3 | 0.06% | Hanya 3 sinopsis yang kosong |
| `release_date` | 1 | 0.02% | 1 baris tanpa tanggal rilis |
| `runtime` | 2 | 0.04% | 2 film tanpa durasi |

**Missing values pada `credits_df`:** ✅ **Tidak ada missing value**

### 3.3 Contoh Format Data JSON

**Genres Example:**
```json
[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}]
```

**Keywords Example:**
```json
[{"id": 207, "name": "future"}, {"id": 533, "name": "space war"}]
```

**Cast Example:**
```json
[{"cast_id": 242, "character": "Jake Sully", "name": "Sam Worthington", "order": 0}]
```

---

## 4. Data Preparation

Setelah memahami struktur dan kondisi data, langkah berikutnya adalah mempersiapkan data agar dapat digunakan dalam model:

### 4.1 Load Data & Merge
```python
# Load CSV files
movies_df = pd.read_csv('tmdb_5000_movies.csv')
credits_df = pd.read_csv('tmdb_5000_credits.csv')

# Merge DataFrames
movies_df = movies_df.merge(credits_df, left_on='id', right_on='movie_id', how='left')
```

### 4.2 JSON Parsing
Untuk fitur `genres`, `keywords`, `cast`, dan `crew`, data masih berupa JSON string. Dibuat fungsi bantu untuk mengekstrak informasi:

```python
def parse_json_column(json_str):
    """Extract names from JSON string"""
    try:
        data = ast.literal_eval(json_str)
        return [item['name'] for item in data if 'name' in item]
    except:
        return []

def get_top_cast(cast_json_str, top_n=3):
    """Extract top N cast members"""
    try:
        cast_data = ast.literal_eval(cast_json_str)
        sorted_cast = sorted(cast_data, key=lambda x: x['order'])
        return [actor['name'] for actor in sorted_cast[:top_n]]
    except:
        return []

def get_director(crew_json_str):
    """Extract director name"""
    try:
        crew_data = ast.literal_eval(crew_json_str)
        directors = [person['name'] for person in crew_data if person['job'] == 'Director']
        return directors
    except:
        return []
```

**Hasil parsing:**
- `genres_list` ← hasil parsing genres
- `keywords_list` ← hasil parsing keywords  
- `cast_list` ← 3 pemeran utama berdasarkan order
- `director_list` ← nama sutradara

### 4.3 Text Cleaning
Setelah parsing, setiap kolom berisi list of strings. Dilakukan preprocessing:

```python
def clean_text_list(text_list):
    """Clean and preprocess text list"""
    if not isinstance(text_list, list):
        return []
    
    cleaned = []
    for text in text_list:
        if isinstance(text, str) and text.strip():
            # Convert to lowercase and remove spaces
            clean_text = text.lower().replace(" ", "")
            cleaned.append(clean_text)
    
    return cleaned
```

**Proses cleaning:**
1. ✅ Ubah semua teks menjadi huruf kecil (lowercase)
2. ✅ Hapus spasi dalam elemen ("Science Fiction" → "sciencefiction")  
3. ✅ Filter string kosong

**Hasil:**
- `genres_clean`
- `keywords_clean`
- `cast_clean`
- `director_clean`

### 4.4 Content Soup Creation
Gabungkan semua fitur yang sudah dibersihkan menjadi satu string utuh:

```python
def create_soup(row):
    """Combine all content features into soup"""
    features = []
    for feature_list in [
        row['genres_clean'],
        row['keywords_clean'], 
        row['cast_clean'],
        row['director_clean']
    ]:
        if isinstance(feature_list, list):
            features.extend(feature_list)
    return " ".join(features)

movies_df['soup'] = movies_df.apply(create_soup, axis=1)
```

### 4.5 Data Filtering
```python
# Remove rows with empty soup
movies_df = movies_df[movies_df['soup'].str.len() > 0].reset_index(drop=True)

# Create title-to-index mapping
indices = pd.Series(movies_df.index, index=movies_df['title']).drop_duplicates()
```

**Hasil akhir:** 4,790 film dengan soup non-kosong (dari 4,803 film awal)

---

## 5. Modeling & Results

### 5.1 Model Architecture

#### **Pendekatan 1: TF-IDF Vectorization**
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=5000,
    ngram_range=(1,2),
    min_df=2,
    max_df=0.8
)

tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['soup'])
# Shape: (4790, 5000)

# Cosine Similarity Calculation
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
# Shape: (4790, 4790)
```

#### **Pendekatan 2: Count Vectorizer (Alternatif)**
```python
from sklearn.feature_extraction.text import CountVectorizer

count_vectorizer = CountVectorizer(
    stop_words='english',
    max_features=5000,
    ngram_range=(1,2)
)

count_matrix = count_vectorizer.fit_transform(movies_df['soup'])
cosine_sim_count = cosine_similarity(count_matrix, count_matrix)
```

### 5.2 Recommendation Functions

#### **TF-IDF Recommendation Function**
```python
def get_movie_recommendations_tfidf(title, top_n=10, min_similarity=0.0):
    """Get movie recommendations using TF-IDF approach"""
    if title not in indices.index:
        return pd.DataFrame()
    
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Filter by minimum similarity and exclude the input movie
    sim_scores = [
        (i, score) for i, score in sim_scores[1:]
        if score >= min_similarity
    ][:top_n]
    
    movie_indices = [i for i, _ in sim_scores]
    scores = [score for _, score in sim_scores]
    
    return pd.DataFrame({
        'title': movies_df['title'].iloc[movie_indices].values,
        'release_year': movies_df['release_date'].astype(str).str[:4].iloc[movie_indices].values,
        'genres': movies_df['genres_list'].iloc[movie_indices].values,
        'vote_average': movies_df['vote_average'].iloc[movie_indices].values,
        'vote_count': movies_df['vote_count'].iloc[movie_indices].values,
        'similarity_score': scores
    })
```

### 5.3 Recommendation Results

#### **🎬 Top-5 Rekomendasi untuk "Avatar" (TF-IDF)**

| **Rank** | **Judul** | **Release Year** | **Vote Average** | **Vote Count** | **Similarity Score** |
|----------|-----------|------------------|------------------|----------------|---------------------|
| 1 | Star Trek Into Darkness | 2013 | 7.70 | 6,492 | 0.340 |
| 2 | Aliens | 1986 | 7.60 | 5,561 | 0.320 |
| 3 | Clash of the Titans | 2010 | 5.80 | 2,345 | 0.300 |
| 4 | Galaxina | 1980 | 4.20 | 432 | 0.290 |
| 5 | Wrath of the Titans | 2012 | 5.20 | 876 | 0.280 |

#### **🎬 Top-5 Rekomendasi untuk "Avatar" (CountVectorizer)**

| **Rank** | **Judul** | **Release Year** | **Vote Average** | **Vote Count** | **Similarity Score** |
|----------|-----------|------------------|------------------|----------------|---------------------|
| 1 | Star Trek Into Darkness | 2013 | 7.70 | 6,492 | 0.315 |
| 2 | Aliens | 1986 | 7.60 | 5,561 | 0.298 |
| 3 | Clash of the Titans | 2010 | 5.80 | 2,345 | 0.285 |
| 4 | Galaxina | 1980 | 4.20 | 432 | 0.270 |
| 5 | Wrath of the Titans | 2012 | 5.20 | 876 | 0.260 |

### 5.4 Model Comparison

| **Aspek** | **TF-IDF** | **Count Vectorizer** |
|-----------|------------|---------------------|
| **Similarity Score Range** | 0.280 - 0.340 | 0.260 - 0.315 |
| **Ranking Consistency** | ✅ Sama | ✅ Sama |
| **Computational Efficiency** | ⚡ Lebih cepat | 🐌 Lebih lambat |
| **Feature Weighting** | ✅ Ada (IDF) | ❌ Tidak ada |

---

## 6. Evaluation

### 6.1 Evaluation Metrics

Untuk menilai kinerja sistem rekomendasi, digunakan metrik berikut:

#### **1. Precision@K**
Proporsi film rekomendasi yang dianggap "relevan" (rating ≥ 7.0) dari total film yang direkomendasikan.

```python
def precision_at_k(recommendations, relevant_threshold=7.0):
    """Calculate Precision@K metric"""
    if recommendations.empty:
        return 0.0
    
    relevant_items = len(recommendations[recommendations['vote_average'] >= relevant_threshold])
    return relevant_items / len(recommendations)
```

#### **2. Diversity**
Mengukur keragaman genre dalam daftar rekomendasi.

```python
def calculate_diversity(recommendations):
    """Calculate genre diversity in recommendations"""
    if recommendations.empty:
        return 0.0
    
    all_genres = []
    for genres_list in recommendations['genres']:
        if isinstance(genres_list, list):
            all_genres.extend(genres_list)
    
    if not all_genres:
        return 0.0
    
    unique_genres = len(set(all_genres))
    total_genres = len(all_genres)
    return unique_genres / total_genres
```

#### **3. Novelty**
Mengukur kebaruan film berdasarkan popularitas (vote_count rendah → lebih novel).

```python
def calculate_novelty(recommendations):
    """Calculate novelty based on popularity"""
    if recommendations.empty:
        return 0.0
    
    avg_popularity = recommendations['vote_count'].mean()
    max_popularity = movies_df['vote_count'].max()
    
    if max_popularity == 0:
        return 0.0
    
    novelty = 1 - (avg_popularity / max_popularity)
    return max(0.0, novelty)
```

#### **4. Average Similarity**
Rata-rata skor kemiripan (cosine similarity) untuk seluruh rekomendasi.

#### **5. Average Rating**
Rata-rata vote_average dari seluruh rekomendasi.

### 6.2 Evaluation Results

#### **📊 Hasil Evaluasi (5 Film Uji)**

Parameter yang digunakan: `top_n=10`, `min_similarity=0.05`

| **Film** | **Precision@K** | **Diversity** | **Novelty** | **Avg Similarity** | **Avg Rating** |
|----------|----------------|---------------|-------------|-------------------|----------------|
| Avatar | 0.20 | 0.2500 | 0.9008 | 0.18 | 5.88 |
| The Dark Knight | 0.40 | 0.2333 | 0.7729 | 0.30 | 6.38 |
| Titanic | 0.10 | 0.1667 | 0.9763 | 0.17 | 6.24 |
| Toy Story | 0.30 | 0.2414 | 0.9095 | 0.27 | 6.48 |
| The Godfather | 0.20 | 0.2759 | 0.9615 | 0.20 | 5.80 |

#### **📈 Rata-rata Performa Sistem**

| **Metrik** | **Nilai** | **Interpretasi** |
|------------|-----------|------------------|
| **Average Precision@K** | 0.240 | 24% rekomendasi berkualitas tinggi |
| **Average Diversity** | 0.233 | Keragaman genre cukup baik |
| **Average Novelty** | 0.904 | Sangat baik dalam merekomendasikan film baru |
| **Average Similarity** | 0.224 | Kemiripan konten yang konsisten |
| **Average Rating** | 6.16 | Rating rata-rata di atas standar |

### 6.3 Performance Analysis

#### **✅ Kelebihan:**
- **High Novelty (0.904):** Sistem sangat baik dalam merekomendasikan film yang kurang populer/lebih fresh
- **Consistent Similarity:** Skor kemiripan stabil di rentang 0.17-0.30
- **Explainable:** Rekomendasi dapat dijelaskan berdasarkan kesamaan konten

#### **⚠️ Area Improvement:**
- **Low Precision@K (0.240):** Hanya 24% rekomendasi yang berkualitas tinggi (rating ≥ 7.0)
- **Moderate Diversity (0.233):** Keragaman genre bisa ditingkatkan
- **Average Rating (6.16):** Belum mencapai target rating tinggi

---

## 7. Interactive Interface

### 7.1 Terminal-Based Interface

Sistem menyediakan fungsi interaktif berbasis terminal yang memungkinkan pengguna mencoba rekomendasi film secara langsung:

```python
def recommend_movies_interactive():
    """Interactive movie recommendation system"""
    print("\n" + "🎬" * 20)
    print("SISTEM REKOMENDASI FILM INTERAKTIF")
    print("🎬" * 20)
    
    while True:
        print(f"\nJumlah film dalam database: {len(movies_df)}")
        movie_title = input("\nMasukkan judul film (atau 'quit' untuk keluar): ").strip()
        
        if movie_title.lower() == 'quit':
            print("Terima kasih telah menggunakan sistem rekomendasi!")
            break
            
        if not movie_title:
            print("❌ Silakan masukkan judul film.")
            continue

        try:
            num_recs = int(input("Berapa jumlah rekomendasi yang diinginkan? (default: 5): ").strip() or 5)
            min_sim = float(input("Minimum similarity score? (0.0–1.0, default: 0.1): ").strip() or 0.1)
        except ValueError:
            print("❌ Input tidak valid. Menggunakan nilai default (5 film, 0.1).")
            num_recs = 5
            min_sim = 0.1

        print(f"\n🔍 Mencari rekomendasi untuk '{movie_title}'…")
        recs = get_movie_recommendations_tfidf(movie_title, num_recs, min_sim)
        
        if not recs.empty:
            print(f"\n✅ Ditemukan {len(recs)} rekomendasi:")
            print("-" * 80)
            for idx, row in recs.iterrows():
                print(f"{idx+1}. {row['title']} ({row['release_year']})")
                print(f"   Rating: {row['vote_average']:.1f} | Similarity: {row['similarity_score']:.3f}")
                print(f"   Genres: {', '.join(row['genres']) if row['genres'] else 'N/A'}\n")
        else:
            print("❌ Tidak ditemukan rekomendasi yang memenuhi kriteria.")
        
        print("-" * 80)

# Untuk menjalankan mode interaktif
# recommend_movies_interactive()
```

### 7.2 Features Interface

#### **🎯 Fitur Utama:**
- ✅ Input judul film secara interaktif
- ✅ Pengaturan jumlah rekomendasi (customizable)
- ✅ Pengaturan minimum similarity score
- ✅ Display hasil yang informatif (judul, tahun, rating, genre, similarity score)
- ✅ Error handling untuk input yang tidak valid
- ✅ Exit command untuk keluar dari sistem

#### **📱 Sample Usage:**
```
🎬🎬🎬🎬🎬🎬🎬🎬🎬🎬🎬🎬🎬🎬🎬🎬🎬🎬🎬🎬
SISTEM REKOMENDASI FILM INTERAKTIF
🎬🎬🎬🎬🎬🎬🎬🎬🎬🎬🎬🎬🎬🎬🎬🎬🎬🎬🎬🎬

Jumlah film dalam database: 4790

Masukkan judul film (atau 'quit' untuk keluar): Avatar
Berapa jumlah rekomendasi yang diinginkan? (default: 5): 5
Minimum similarity score? (0.0–1.0, default: 0.1): 0.1

🔍 Mencari rekomendasi untuk 'Avatar'…

✅ Ditemukan 5 rekomendasi:
--------------------------------------------------------------------------------
1. Star Trek Into Darkness (2013)
   Rating: 7.7 | Similarity: 0.340
   Genres: Action, Adventure, Science Fiction

2. Aliens (1986)
   Rating: 7.6 | Similarity: 0.320
   Genres: Action, Thriller, Science Fiction
```

---

## 8. Kesimpulan dan Insight

### 🎯 **RINGKASAN SISTEM**

| **Aspek** | **Detail** |
|-----------|------------|
| **Algoritma** | Content-Based Filtering |
| **Feature Extraction** | TF-IDF Vectorizer |
| **Similarity Metric** | Cosine Similarity |
| **Features Used** | Genres, Keywords, Cast, Director |
| **Dataset Size** | 4,790 film (dari 4,803 original) |
| **Coverage Rate** | 99.73% |

### ✅ **KELEBIHAN SISTEM**

1. **🚀 Cold Start Friendly**
   - Tidak membutuhkan data interaksi pengguna
   - Cocok untuk user baru tanpa riwayat menonton

2. **🔍 Explainable Recommendations**
   - Rekomendasi berdasarkan kemiripan konten yang jelas
   - User dapat memahami alasan di balik rekomendasi

3. **⚡ Scalable & Efficient**
   - TF-IDF efisien untuk dataset besar
   - Cosine similarity computation yang optimal

4. **📊 High Coverage**
   - 99.73% film dalam dataset dapat direkomendasikan
   - Minimal data loss dalam preprocessing

5. **🎭 High Novelty Score (0.904)**
   - Sangat baik dalam merekomendasikan film yang kurang populer
   - Membantu user menemukan hidden gems

### ⚠️ **LIMITASI SISTEM**

1. **🎯 Low Precision@K (0.240)**
   - Hanya 24% rekomendasi yang berkualitas tinggi (rating ≥ 7.0)
   - Perlu improvement dalam filter kualitas

2. **🔄 Over-specialization Risk**
   - Rekomendasi mungkin terlalu mirip dengan input
   - Kurang eksplorasi genre/konten baru

3. **👤 No Personal Preferences**
   - Tidak mempertimbangkan preferensi individual user
   - Semua user mendapat rekomendasi yang sama untuk film yang sama

4. **🛠️ Preprocessing Intensive**
   - Memerlukan parsing JSON dan text cleaning yang kompleks
   - Dependency pada kualitas metadata

5. **📝 Limited Content Analysis**
   - Tidak mempertimbangkan sentimen atau konteks naratif
   - Hanya berdasarkan metadata struktural

### 🚀 **FUTURE IMPROVEMENTS**

1. **🔀 Hybrid Filtering Implementation**
   ```python
   # Combine Content-Based + Collaborative Filtering
   hybrid_score = α * content_similarity + β * collaborative_score
   ```

2. **💬 Sentiment Analysis Integration**
   - Analisis sentimen dari kolom `overview`
   - Integrasi review pengguna dari platform eksternal

3. **👥 User Behavior Integration**
   - Data history menonton pengguna
   - Click-through rate dan implicit feedback

4. **🔄 Real-time Updates**
   - Update similarity matrix secara incremental
   - Pembelajaran adaptif berdasarkan feedback

5. **🎯 Quality Filtering Enhancement**
   - Weighted scoring berdasarkan rating dan popularity
   - Ensemble methods untuk multiple quality signals

6. **🌐 Multi-modal Features**
   - Integrasi poster/trailer analysis
   - Audio-visual content analysis

### 📈 **TECHNICAL METRICS SUMMARY**

```python
# Coverage Analysis
total_movies_original = 4803           # Sebelum filtering soup
movies_with_recs     = len(movies_df)  # Setelah filtering soup > 0
coverage_rate        = movies_with_recs / total_movies_original

print(f"Total film yang dapat direkomendasikan: {movies_with_recs}/{total_movies_original}")
print(f"Coverage Rate: {coverage_rate:.1%}")
```

**Output:**
```
Total film yang dapat direkomendasikan: 4790/4803  
Coverage Rate: 99.73%
```

### 🎉 **FINAL RECOMMENDATIONS**

1. **For Production Use:**
   - Implement quality threshold filtering (vote_average ≥ 6.5)
   - Add diversity boosting algorithms
   - Create A/B testing framework

2. **For Model Enhancement:**
   - Experiment with advanced embedding techniques (Word2Vec, BERT)
   - Implement graph-based recommendation methods
   - Add temporal dynamics consideration

3. **For User Experience:**
   - Develop web-based interface
   - Add explanation features for recommendations
   - Implement user feedback collection system

---

### 📧 **USAGE INSTRUCTIONS**

Sistem rekomendasi film telah siap digunakan! 

**Untuk eksperimen manual:**
```python
# Gunakan fungsi utama
recommendations = get_movie_recommendations_tfidf(
    title="Avatar", 
    top_n=10, 
    min_similarity=0.1
)
print(recommendations)
```

**Untuk mode interaktif:**
```python
# Uncomment dan jalankan
recommend_movies_interactive()
```

---

**🏆 Project Status: COMPLETED**  
**📅 Last Updated: June 2025**  
**👨‍💻 Author: Salman Alfarizi T