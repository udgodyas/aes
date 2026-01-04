# -*- coding: utf-8 -*-
"""
Created on Sat Jan  3 18:49:44 2026

@author: kuzey
"""

import os
from docx import Document
import pandas as pd
import pickle
import joblib

# NLTK import'u - Windows'ta gmpy2/mpmath sorunları olabilir, graceful degradation
NLTK_AVAILABLE = False
try:
    import nltk
    from nltk.tokenize import sent_tokenize
    NLTK_AVAILABLE = True
    # İlk çalıştırmada gerekebilir - daha güvenli indirme
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        try:
            # NLTK veri indirme - hata durumunda sessizce devam et
            try:
                nltk.download('punkt', quiet=True, raise_on_error=False)
            except Exception as download_error:
                # İndirme hatası - NLTK'yı devre dışı bırak
                NLTK_AVAILABLE = False
                print(f"⚠ Uyarı: NLTK punkt verisi indirilemedi (cümle bölme için basit yöntem kullanılacak)")
        except Exception as e:
            NLTK_AVAILABLE = False
            print(f"⚠ Uyarı: NLTK yapılandırılamadı (cümle bölme için basit yöntem kullanılacak)")
except (ImportError, OSError, Exception) as e:
    # Windows'ta gmpy2/mpmath DLL hataları olabilir
    NLTK_AVAILABLE = False
    # Sessizce devam et - NLTK olmadan da çalışabilir
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠ XGBoost bulunamadı, bazı modeller kullanılamayacak.")
try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠ LightGBM bulunamadı, bazı modeller kullanılamayacak.")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Görselleştirme ayarları
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('default')
sns.set_palette("husl")

def natural_sort_key(filename):
    """
    Dosya adlarını doğal sıralama için anahtar fonksiyonu.
    A1, A2, A3... A10, A11 şeklinde sıralar.
    """
    # Dosya adından uzantıyı kaldır (A1.docx -> A1)
    name_without_ext = os.path.splitext(filename)[0]
    # Harf ve sayı kısımlarını ayır (A1 -> ('A', 1))
    match = re.match(r'([A-Za-z]+)(\d+)', name_without_ext)
    if match:
        letter_part = match.group(1)
        number_part = int(match.group(2))
        return (letter_part, number_part)
    return (name_without_ext, 0)

# Data dizinindeki tüm Word dosyalarını oku ve içeriklerini diziye aktar
def read_word_files(data_dir="data"):
    """
    Belirtilen dizindeki tüm .docx dosyalarını okur ve içeriklerini bir diziye aktarır.
    
    Args:
        data_dir (str): Word dosyalarının bulunduğu dizin yolu
    
    Returns:
        tuple: (documents_content, essay_nos, file_names) - İçerikler, essay no değerleri ve dosya adları
    """
    documents_content = []
    essay_nos = []
    file_names = []
    
    # Dizin içindeki tüm dosyaları listele
    if not os.path.exists(data_dir):
        print(f"Hata: '{data_dir}' dizini bulunamadı!")
        return documents_content, essay_nos, file_names
    
    # .docx dosyalarını bul ve doğal sıralama ile sırala
    word_files = [f for f in os.listdir(data_dir) if f.endswith('.docx')]
    word_files.sort(key=natural_sort_key)  # Doğal sıralama (A1, A2, A3... A10, A11...)
    
    print(f"Toplam {len(word_files)} Word dosyası bulundu.")
    
    # Her dosyayı oku
    for filename in word_files:
        file_path = os.path.join(data_dir, filename)
        # Dosya adını .docx uzantısı olmadan kaydet (A1, A2, vb.)
        file_name_without_ext = os.path.splitext(filename)[0]
        file_names.append(file_name_without_ext)
        
        try:
            doc = Document(file_path)
            # İlk satırı (essay no) atlayarak tüm paragrafları birleştir
            paragraphs = doc.paragraphs
            if len(paragraphs) > 0:
                # İlk paragrafı essay no olarak kaydet
                essay_no = paragraphs[0].text.strip()
                essay_nos.append(essay_no)
                # İlk paragrafı atla (essay no)
                text_content = '\n'.join([paragraph.text for paragraph in paragraphs[1:]])
            else:
                essay_nos.append("")
                text_content = ""
            documents_content.append(text_content)
            print(f"✓ {filename} okundu - Essay No: {essay_nos[-1] if essay_nos else 'N/A'} ({len(text_content)} karakter)")
        except Exception as e:
            print(f"✗ {filename} okunurken hata oluştu: {str(e)}")
            essay_nos.append("")
            documents_content.append("")  # Hata durumunda boş string ekle
    
    return documents_content, essay_nos, file_names

# Excel dosyasını oku ve içeriğini listeye aktar
def read_excel_file(file_path):
    """
    Excel dosyasını okur ve içeriğini DataFrame olarak döndürür.
    
    Args:
        file_path (str): Excel dosyasının yolu
    
    Returns:
        pandas.DataFrame: Excel dosyasının içeriği
    """
    try:
        df = pd.read_excel(file_path)
        print(f"✓ {os.path.basename(file_path)} okundu ({len(df)} satır, {len(df.columns)} sütun)")
        return df
    except Exception as e:
        print(f"✗ {os.path.basename(file_path)} okunurken hata oluştu: {str(e)}")
        return None

# Word dosyalarını oku
documents, essay_nos, file_names = read_word_files()

# Excel dosyasını oku
excel_file_path = os.path.join("data", "ALL_SCORES.xlsx")
scores_data = read_excel_file(excel_file_path)

# Sonuçları göster
#print(f"\nToplam {len(documents)} dosya içeriği diziye aktarıldı.")
#print(f"Toplam {len(essay_nos)} essay no değeri kaydedildi.")
#print(f"Toplam {len(file_names)} dosya adı kaydedildi.")
#print(f"İlk dosya adı: {file_names[0] if file_names else 'N/A'}")
#print(f"İlk dosyanın içeriği uzunluğu: {len(documents[0]) if documents else 0} karakter")
#print(f"İlk dosyanın essay no: {essay_nos[0] if essay_nos else 'N/A'}")
#if scores_data is not None:
    #print(f"\nExcel dosyası başarıyla okundu:")
    #print(f"  - Satır sayısı: {len(scores_data)}")
    #print(f"  - Sütun sayısı: {len(scores_data.columns)}")
    #print(f"  - Sütun adları: {list(scores_data.columns)}")
    #print(f"\nİlk 5 satır:")
    #print(scores_data.head())

# Rater ve kriter yapısını tanımla
RATERS = ['RA']  # Model eğitimi için sadece RA kullanılıyor (RB skorları mevcut ama kullanılmıyor)
CRITERIA = ['TITLE', 'THESIS', 'ORGANISATION', 'SUPPORT', 'ANALYSIS', 'SENTENCE', 'GRAMMAR']  # Puanlama kriterleri

# DataFrame oluştur
data = {"essay": documents}  # A1, A2, A3... sıralamasına göre essay içerikleri

# Her rater ve kriter için skorları ekle
for rater in RATERS:
    for criterion in CRITERIA:
        col_name = f"{rater}_{criterion}"
        if col_name in scores_data.columns:
            data[col_name] = scores_data[col_name]
        else:
            print(f"⚠ Uyarı: {col_name} sütunu bulunamadı!")

df = pd.DataFrame(data)

# ============================================================================
# SBERT ile Modelleme
# ============================================================================

def create_sbert_embeddings(texts, model_name='all-mpnet-base-v2', batch_size=32):
    """
    SBERT modeli kullanarak metinler için embedding'ler oluşturur.
    
    Args:
        texts (list): Embedding oluşturulacak metin listesi
        model_name (str): Kullanılacak SBERT model adı
        batch_size (int): Batch boyutu
    
    Returns:
        numpy.ndarray: Embedding'ler (n_samples, embedding_dim)
    """
    print(f"\nSBERT modeli yükleniyor: {model_name}")
    model = SentenceTransformer(model_name)
    
    print(f"{len(texts)} metin için embedding oluşturuluyor...")
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    
    print(f"Embedding'ler oluşturuldu. Şekil: {embeddings.shape}")
    return embeddings

def calculate_golden_vector_similarity(embeddings, scores, top_percentile=0.2):
    """
    Centroid Tekniği: En yüksek puanlı essay'lerin embedding'lerinin ortalamasını alır
    ("Altın Vektör") ve tüm essay'lerin bu vektöre olan cosine similarity'sini hesaplar.
    Analysis kriteri için konuya sadakat ölçümü sağlar.
    
    Args:
        embeddings (numpy.ndarray): Tüm essay embedding'leri (n_samples, embedding_dim)
        scores (numpy.ndarray): Essay skorları (n_samples,)
        top_percentile (float): En yüksek puanlı essay'lerin yüzdesi (varsayılan: %20)
    
    Returns:
        numpy.ndarray: Her essay'in Altın Vektör'e olan cosine similarity'si (n_samples,)
    """
    if len(embeddings) == 0 or len(scores) == 0:
        return np.zeros(len(embeddings))
    
    # En yüksek puanlı essay'leri bul
    threshold = np.percentile(scores, (1 - top_percentile) * 100)
    top_indices = np.where(scores >= threshold)[0]
    
    if len(top_indices) == 0:
        # Eğer hiç yüksek puanlı essay yoksa, en yüksek puanlı tek essay'i kullan
        top_indices = [np.argmax(scores)]
    
    # Altın Vektör: En yüksek puanlı essay'lerin embedding'lerinin ortalaması
    golden_vector = np.mean(embeddings[top_indices], axis=0)
    
    # Tüm essay'lerin Altın Vektör'e olan cosine similarity'sini hesapla
    similarities = cosine_similarity(embeddings, [golden_vector]).flatten()
    
    return similarities

def extract_coherence_features(text, model):
    """
    Ardışık cümleler arası anlamsal benzerliği ölçer (Tutarlılık analizi).
    Organisation ve Analysis kriterleri için önemlidir.
    
    Args:
        text (str): Analiz edilecek metin
        model: SBERT modeli (SentenceTransformer)
    
    Returns:
        list: [avg_coherence, min_coherence] - Ortalama ve minimum tutarlılık
    """
    if not text or len(text.strip()) == 0:
        return [0.0, 0.0]
    
    # Cümleleri ayır (NLTK kullanarak daha doğru bölme)
    if NLTK_AVAILABLE:
        try:
            sentences = sent_tokenize(text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 5]  # En az 5 karakter
        except:
            # NLTK başarısız olursa basit split kullan
            sentences = text.split('.')
            sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    else:
        # NLTK yoksa basit split kullan
        sentences = text.split('.')
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    
    if len(sentences) < 2:
        return [0.0, 0.0]  # Yeterli cümle yoksa
    
    try:
        # Cümleler için embedding'ler oluştur
        embeddings = model.encode(sentences, convert_to_numpy=True)
        
        # Ardışık cümleler arası benzerlikleri hesapla
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
            similarities.append(sim)
        
        avg_coherence = np.mean(similarities) if similarities else 0.0
        min_coherence = np.min(similarities) if similarities else 0.0  # En kopuk nokta
        
        return [avg_coherence, min_coherence]
    except Exception as e:
        # Hata durumunda sıfır döndür
        return [0.0, 0.0]

def extract_text_features(texts, sbert_model=None, golden_vector_similarity=None):
    """
    Metinlerden özellik mühendisliği ile ek özellikler çıkarır.
    Tutarlılık (coherence) ve Altın Vektör benzerliği özellikleri dahildir.
    
    Args:
        texts (list): Metin listesi
        sbert_model: SBERT modeli (tutarlılık analizi için gerekli, None ise atlanır)
        golden_vector_similarity (numpy.ndarray): Altın Vektör'e olan benzerlik skorları (None ise atlanır)
    
    Returns:
        numpy.ndarray: Ek özellikler (n_samples, n_features)
    """

def extract_text_features(texts, sbert_model=None, golden_vector_similarity=None):
    """
    Metinlerden özellik mühendisliği ile ek özellikler çıkarır.
    Tutarlılık (coherence) ve Altın Vektör benzerliği özellikleri dahildir.
    
    Args:
        texts (list): Metin listesi
        sbert_model: SBERT modeli (tutarlılık analizi için gerekli, None ise atlanır)
        golden_vector_similarity (numpy.ndarray): Altın Vektör'e olan benzerlik skorları (None ise atlanır)
    
    Returns:
        numpy.ndarray: Ek özellikler (n_samples, n_features)
    """
    features = []
    
    for i, text in enumerate(texts):
        if not text or len(text.strip()) == 0:
            text = ""
        
        # Temel metin özellikleri
        char_count = len(text)
        
        # Türkçe'ye özgü iyileştirilmiş tokenization (regex kullanarak)
        # Noktalama işaretlerinden ayrılmış kelimeleri bulur
        words = re.findall(r'\b\w+\b', text.lower())
        word_count = len(words) if words else 0
        
        # Cümle sayısı (noktalama işaretlerine göre)
        sentence_count = len([s for s in re.split(r'[.!?]+', text) if s.strip()]) if text else 0
        
        # Ortalama kelime uzunluğu
        avg_word_length = np.mean([len(word) for word in words]) if word_count > 0 else 0
        
        # Ortalama cümle uzunluğu
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        # Noktalama işaretleri
        punctuation_count = sum(1 for char in text if char in '.,!?;:')
        
        # Büyük harf oranı
        uppercase_ratio = sum(1 for char in text if char.isupper()) / char_count if char_count > 0 else 0
        
        # Rakam sayısı
        digit_count = sum(1 for char in text if char.isdigit())
        
        # Özel karakterler
        special_char_count = sum(1 for char in text if not char.isalnum() and char not in ' .,!?;:')
        
        # Kelime çeşitliliği (unique words / total words) - Türkçe için iyileştirilmiş
        unique_words = len(set(words))
        lexical_diversity = unique_words / word_count if word_count > 0 else 0
        
        # Temel özellikler listesi
        base_features = [
            char_count,
            word_count,
            sentence_count,
            avg_word_length,
            avg_sentence_length,
            punctuation_count,
            uppercase_ratio,
            digit_count,
            special_char_count,
            lexical_diversity
        ]
        
        # Tutarlılık (Coherence) özellikleri (Organisation ve Analysis için)
        if sbert_model is not None:
            coherence_features = extract_coherence_features(text, sbert_model)
            base_features.extend(coherence_features)
        else:
            # Model yoksa sıfır ekle
            base_features.extend([0.0, 0.0])
        
        # Altın Vektör Benzerliği (Analysis için - konuya sadakat ölçümü)
        if golden_vector_similarity is not None and i < len(golden_vector_similarity):
            base_features.append(golden_vector_similarity[i])
        else:
            base_features.append(0.0)
        
        features.append(base_features)
    
    return np.array(features)

def combine_features(embeddings, text_features, scaler=None, fit_scaler=False):
    """
    SBERT embedding'leri ile metin özelliklerini birleştirir.
    NOT: Scaling işlemi train_test_split'ten SONRA yapılmalı (veri sızıntısını önlemek için).
    
    Args:
        embeddings (numpy.ndarray): SBERT embedding'leri
        text_features (numpy.ndarray): Metin özellikleri
        scaler (StandardScaler): Önceden eğitilmiş scaler (None ise yeni oluşturulur)
        fit_scaler (bool): Scaler'ı fit et (True: fit_transform, False: transform)
    
    Returns:
        tuple: (birleştirilmiş özellikler, scaler)
    """
    # Scaler oluştur veya kullan
    if scaler is None:
        scaler = StandardScaler()
    
    # Özellikleri normalize et (fit_scaler=True ise fit_transform, False ise transform)
    if fit_scaler:
        text_features_scaled = scaler.fit_transform(text_features)
    else:
        text_features_scaled = scaler.transform(text_features)
    
    # Birleştir
    combined = np.hstack([embeddings, text_features_scaled])
    return combined, scaler

def calculate_sample_weights(y):
    """
    Hedef değişken dağılımına göre örnek ağırlıkları hesaplar.
    Az bulunan puanları (uç değerler) daha fazla ağırlandırır.
    
    Args:
        y (numpy.ndarray): Hedef değişken (skorlar)
    
    Returns:
        numpy.ndarray: Örnek ağırlıkları
    """
    # Skor dağılımını hesapla
    unique_scores, counts = np.unique(y, return_counts=True)
    
    # Her skor için ağırlık: toplam sayı / (sınıf sayısı * sınıf frekansı)
    # Bu formül, az bulunan skorları daha fazla ağırlandırır
    score_weights = {}
    total_samples = len(y)
    num_classes = len(unique_scores)
    
    for score, count in zip(unique_scores, counts):
        # Inverse frequency weighting: az bulunan skorlar daha yüksek ağırlık alır
        score_weights[score] = total_samples / (num_classes * count)
    
    # Her örnek için ağırlık ataması
    sample_weights = np.array([score_weights[score] for score in y])
    
    # Ağırlıkları normalize et (toplam = n_samples)
    sample_weights = sample_weights * len(y) / sample_weights.sum()
    
    return sample_weights

def create_stratified_folds(y, n_splits=5, random_state=42):
    """
    Regresyon için Stratified K-Fold oluşturur.
    Skorları kategorilere bölerek stratified split yapar.
    
    Args:
        y (numpy.ndarray): Hedef değişken
        n_splits (int): Fold sayısı
        random_state (int): Random seed
    
    Returns:
        StratifiedKFold veya KFold: Cross validation splitter
    """
    # Skorları kategorilere böl (5 kategori: çok düşük, düşük, orta, yüksek, çok yüksek)
    y_min, y_max = y.min(), y.max()
    y_range = y_max - y_min
    
    if y_range > 0:
        # Skorları 5 kategoriye böl
        bins = np.linspace(y_min, y_max, 6)  # 5 kategori için 6 bin edge
        y_categorical = np.digitize(y, bins) - 1
        y_categorical = np.clip(y_categorical, 0, 4)  # 0-4 arası kategoriler
        
        # Her kategoride en az 1 örnek varsa StratifiedKFold kullan
        unique_cats, counts = np.unique(y_categorical, return_counts=True)
        if len(unique_cats) >= n_splits and np.min(counts) >= n_splits:
            return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state), y_categorical
    
    # StratifiedKFold kullanılamazsa normal KFold kullan
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state), None

def train_multiple_models(X, y, test_size=0.2, random_state=42, cv_folds=5, use_stratified=True, use_sample_weights=True):
    """
    Birden fazla regresyon algoritması ile model eğitir ve en iyisini seçer.
    K-Fold Cross Validation ve örnek ağırlandırma kullanarak daha güvenilir performans ölçümü yapar.
    
    Args:
        X (numpy.ndarray): Özellikler
        y (numpy.ndarray): Hedef değişken
        test_size (float): Test seti oranı
        random_state (int): Random seed
        cv_folds (int): Cross validation kat sayısı
        use_stratified (bool): Stratified K-Fold kullan (skor dağılımını korur)
        use_sample_weights (bool): Örnek ağırlandırma kullan (az bulunan puanları ağırlandırır)
    
    Returns:
        dict: En iyi model ve tüm modellerin sonuçları
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Örnek ağırlıklarını hesapla (eğitim seti için)
    sample_weights_train = None
    if use_sample_weights:
        sample_weights_train = calculate_sample_weights(y_train)
        print(f"\n  Örnek ağırlandırma aktif: Az bulunan puanlar ağırlandırıldı")
        print(f"  Ağırlık aralığı: [{sample_weights_train.min():.4f}, {sample_weights_train.max():.4f}]")
    
    # Model listesi
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=random_state),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
        'AdaBoost': AdaBoostRegressor(n_estimators=50, random_state=random_state),
        'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1)
    }
    
    # XGBoost ve LightGBM ekle (varsa)
    if XGBOOST_AVAILABLE:
            models['XGBoost'] = XGBRegressor(n_estimators=100, random_state=random_state, n_jobs=1)
    if LIGHTGBM_AVAILABLE:
            models['LightGBM'] = LGBMRegressor(n_estimators=100, random_state=random_state, n_jobs=1)
    
    results = {}
    best_model = None
    best_cv_r2 = -np.inf
    best_name = None
    
    print(f"\n{'='*70}")
    print(f"FARKLI ALGORİTMALAR TEST EDİLİYOR ({cv_folds}-Fold Cross Validation)...")
    print(f"{'='*70}")
    
    # Cross Validation için splitter oluştur
    if use_stratified:
        cv_splitter, y_categorical = create_stratified_folds(y_train, n_splits=cv_folds, random_state=random_state)
        if isinstance(cv_splitter, StratifiedKFold):
            print(f"  Stratified K-Fold kullanılıyor (skor dağılımı korunuyor)")
            kfold = cv_splitter
            # StratifiedKFold için categorical labels gerekli
            cv_labels = y_categorical
        else:
            print(f"  Normal K-Fold kullanılıyor (StratifiedKFold için yeterli kategori yok)")
            kfold = cv_splitter
            cv_labels = None
    else:
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        cv_labels = None
        print(f"  Normal K-Fold kullanılıyor")
    
    for name, model in models.items():
        try:
            # K-Fold Cross Validation ile performans ölçümü
            if isinstance(kfold, StratifiedKFold) and cv_labels is not None:
                # StratifiedKFold için categorical labels kullan
                cv_scores_r2 = cross_val_score(model, X_train, y_train, cv=kfold.split(X_train, cv_labels), 
                                              scoring='r2', n_jobs=1)
                cv_scores_rmse = -cross_val_score(model, X_train, y_train, cv=kfold.split(X_train, cv_labels), 
                                                 scoring='neg_root_mean_squared_error', n_jobs=1)
                cv_scores_mae = -cross_val_score(model, X_train, y_train, cv=kfold.split(X_train, cv_labels), 
                                                 scoring='neg_mean_absolute_error', n_jobs=1)
            else:
                cv_scores_r2 = cross_val_score(model, X_train, y_train, cv=kfold, scoring='r2', n_jobs=1)
                cv_scores_rmse = -cross_val_score(model, X_train, y_train, cv=kfold, scoring='neg_root_mean_squared_error', n_jobs=1)
                cv_scores_mae = -cross_val_score(model, X_train, y_train, cv=kfold, scoring='neg_mean_absolute_error', n_jobs=1)
            
            # Cross validation ortalamaları
            cv_r2_mean = cv_scores_r2.mean()
            cv_r2_std = cv_scores_r2.std()
            cv_rmse_mean = cv_scores_rmse.mean()
            cv_mae_mean = cv_scores_mae.mean()
            
            # Modeli tüm eğitim verisi ile eğit (sample weights ile)
            if use_sample_weights and sample_weights_train is not None:
                # Model sample_weight destekliyorsa kullan
                if hasattr(model, 'fit') and 'sample_weight' in model.fit.__code__.co_varnames:
                    model.fit(X_train, y_train, sample_weight=sample_weights_train)
                else:
                    model.fit(X_train, y_train)
            else:
                model.fit(X_train, y_train)
            
            # Test seti üzerinde tahmin
            y_pred = model.predict(X_test)
            
            # Test seti metrikleri
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            test_mae = mean_absolute_error(y_test, y_pred)
            test_r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'cv_r2_mean': cv_r2_mean,
                'cv_r2_std': cv_r2_std,
                'cv_rmse_mean': cv_rmse_mean,
                'cv_mae_mean': cv_mae_mean,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'test_r2': test_r2,
                'y_pred': y_pred
            }
            
            print(f"\n{name}:")
            print(f"  Cross Validation ({cv_folds}-Fold):")
            print(f"    R²: {cv_r2_mean:.4f} (+/- {cv_r2_std*2:.4f})")
            print(f"    RMSE: {cv_rmse_mean:.4f}")
            print(f"    MAE: {cv_mae_mean:.4f}")
            print(f"  Test Seti:")
            print(f"    R²: {test_r2:.4f}")
            print(f"    RMSE: {test_rmse:.4f}")
            print(f"    MAE: {test_mae:.4f}")
            
            # En iyi modeli cross validation R² skoruna göre seç
            if cv_r2_mean > best_cv_r2:
                best_cv_r2 = cv_r2_mean
                best_model = model
                best_name = name
                
        except Exception as e:
            print(f"\n{name}: Hata - {str(e)}")
    
    print(f"\n{'='*70}")
    print(f"EN İYİ MODEL: {best_name} (CV R²={best_cv_r2:.4f})")
    print(f"{'='*70}")
    
    return {
        'best_model': best_model,
        'best_name': best_name,
        'best_cv_r2': best_cv_r2,
        'all_results': results,
        'X_test': X_test,
        'y_test': y_test
    }

def apply_hybrid_pca(X_train, X_test, embedding_dim=768, n_components=None):
    """
    Hibrit özellik yapısı: PCA sadece SBERT embedding'lerine uygulanır,
    text features (kelime sayısı, cümle uzunluğu vb.) olduğu gibi kalır.
    Bu, manuel özelliklerin kaybolmasını önler.
    
    Args:
        X_train (numpy.ndarray): Eğitim özellikleri [embeddings + text_features]
        X_test (numpy.ndarray): Test özellikleri [embeddings + text_features]
        embedding_dim (int): SBERT embedding boyutu (varsayılan: 768 - all-mpnet-base-v2)
        n_components (int): PCA bileşen sayısı (None ise otomatik belirlenir)
    
    Returns:
        tuple: (X_train_final, X_test_final, pca_model) - İşlenmiş veriler ve PCA modeli
    """
    n_samples, n_features = X_train.shape
    
    # Embedding ve text features'ı ayır
    # İlk embedding_dim sütunu embedding'ler, gerisi text features
    X_train_emb = X_train[:, :embedding_dim]
    X_train_text = X_train[:, embedding_dim:]
    
    X_test_emb = X_test[:, :embedding_dim]
    X_test_text = X_test[:, embedding_dim:]
    
    print(f"\n  Hibrit Özellik Yapısı:")
    print(f"    - Embedding özellikleri: {X_train_emb.shape[1]} (PCA uygulanacak)")
    print(f"    - Text özellikleri: {X_train_text.shape[1]} (PCA uygulanmayacak, korunacak)")
    
    # PCA sadece embedding'lere uygula
    if n_components is None:
        # Küçük veri setleri için optimal bileşen sayısı
        if n_samples < 50:
            n_components = min(15, n_samples - 1, embedding_dim)  # 59 veri için 15 ideal
        elif n_samples < 100:
            n_components = min(20, n_samples - 1, embedding_dim)
        else:
            n_components = min(30, n_samples - 1, embedding_dim)
    else:
        n_components = min(n_components, n_samples - 1, embedding_dim)
    
    if n_components < embedding_dim and n_samples > n_components:
        print(f"    - PCA uygulanıyor: {embedding_dim} embedding -> {n_components} bileşen")
        
        pca = PCA(n_components=n_components, random_state=42)
        X_train_emb_pca = pca.fit_transform(X_train_emb)
        X_test_emb_pca = pca.transform(X_test_emb)
        
        explained_variance = np.sum(pca.explained_variance_ratio_)
        print(f"    - Açıklanan varyans: {explained_variance:.4f} ({explained_variance*100:.2f}%)")
        
        # Embedding PCA ve text features'ı birleştir
        X_train_final = np.hstack([X_train_emb_pca, X_train_text])
        X_test_final = np.hstack([X_test_emb_pca, X_test_text])
        
        print(f"    - Final özellik sayısı: {X_train_final.shape[1]} (PCA embedding: {n_components} + Text: {X_train_text.shape[1]})")
        
        return X_train_final, X_test_final, pca
    else:
        print(f"    - PCA atlandı: Yeterli örnek yok veya embedding boyutu zaten küçük")
        return X_train, X_test, None

def apply_pca_if_needed(X_train, X_test, n_components=None, variance_threshold=0.95):
    """
    Boyut laneti (Curse of Dimensionality) sorununu çözmek için PCA uygular.
    Örnek sayısından fazla özellik varsa veya belirtilen varyansı korumak için PCA kullanır.
    NOT: Bu fonksiyon eski yöntemdir. Hibrit yapı için apply_hybrid_pca kullanılmalı.
    
    Args:
        X_train (numpy.ndarray): Eğitim özellikleri
        X_test (numpy.ndarray): Test özellikleri
        n_components (int): PCA bileşen sayısı (None ise otomatik belirlenir)
        variance_threshold (float): Korunacak varyans oranı (n_components=None ise kullanılır)
    
    Returns:
        tuple: (X_train_pca, X_test_pca, pca_model) - PCA uygulanmış veriler ve model
    """
    n_samples, n_features = X_train.shape
    
    # Eğer örnek sayısından fazla özellik varsa veya çok fazla özellik varsa PCA uygula
    if n_features > n_samples or n_features > 100:
        if n_components is None:
            # Varyansın %95'ini koruyacak bileşen sayısını bul
            pca_temp = PCA()
            pca_temp.fit(X_train)
            cumsum_variance = np.cumsum(pca_temp.explained_variance_ratio_)
            n_components = np.argmax(cumsum_variance >= variance_threshold) + 1
            # Minimum 10, maksimum n_samples-1 bileşen kullan
            n_components = max(10, min(n_components, n_samples - 1, 50))
        else:
            n_components = min(n_components, n_samples - 1, n_features)
        
        print(f"\n  PCA uygulanıyor: {n_features} özellik -> {n_components} bileşen")
        print(f"    (Örnek sayısı: {n_samples}, Özellik sayısı: {n_features})")
        
        pca = PCA(n_components=n_components, random_state=42)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        
        explained_variance = np.sum(pca.explained_variance_ratio_)
        print(f"    Açıklanan varyans: {explained_variance:.4f} ({explained_variance*100:.2f}%)")
        
        return X_train_pca, X_test_pca, pca
    else:
        print(f"\n  PCA atlandı: Örnek sayısı ({n_samples}) yeterli, özellik sayısı ({n_features}) makul")
        return X_train, X_test, None

def train_score_prediction_model_with_data(X_train, X_test, y_train, y_test, use_advanced=True, cv_folds=5, use_stratified=True, use_sample_weights=True, use_pca=True, use_gridsearch=False):
    """
    Zaten ayrılmış train/test verileri ile model eğitir.
    K-Fold Cross Validation, örnek ağırlandırma, PCA ve GridSearchCV kullanarak daha güvenilir performans ölçümü yapar.
    Veri sızıntısını önlemek için kullanılır.
    
    Args:
        X_train (numpy.ndarray): Eğitim özellikleri
        X_test (numpy.ndarray): Test özellikleri
        y_train (numpy.ndarray): Eğitim hedef değişkeni
        y_test (numpy.ndarray): Test hedef değişkeni
        use_advanced (bool): Gelişmiş mod kullan (birden fazla algoritma dener)
        cv_folds (int): Cross validation kat sayısı
        use_stratified (bool): Stratified K-Fold kullan
        use_sample_weights (bool): Örnek ağırlandırma kullan
        use_pca (bool): PCA kullan (boyut laneti için)
        use_gridsearch (bool): GridSearchCV kullan (hiperparametre optimizasyonu)
    
    Returns:
        tuple: (model, X_test, y_test, y_pred, pca_model) - Eğitilmiş model ve test sonuçları
    """
    print(f"\nModel eğitiliyor...")
    print(f"  Eğitim seti: {X_train.shape[0]} örnek")
    print(f"  Test seti: {X_test.shape[0]} örnek")
    print(f"  Özellik sayısı: {X_train.shape[1]}")
    
    # Skor dağılımını göster
    print(f"\n  Hedef değişken (Skor) dağılımı:")
    print(f"    - Min: {y_train.min():.2f}, Max: {y_train.max():.2f}")
    print(f"    - Ortalama: {y_train.mean():.2f}, Std: {y_train.std():.2f}")
    unique_scores, counts = np.unique(y_train, return_counts=True)
    print(f"    - Benzersiz skor sayısı: {len(unique_scores)}")
    if len(unique_scores) <= 10:
        print(f"    - Skor dağılımı: {dict(zip(unique_scores, counts))}")
    
    # Hibrit PCA uygula (PCA sadece embedding'lere, text features korunur)
    pca_model = None
    # Embedding boyutunu otomatik tespit et
    # Text features genelde 13 özellik (10 temel + 2 coherence + 1 golden_vector)
    # Eğer toplam özellik sayısı > 50 ise, ilk kısmı embedding olmalı
    if X_train.shape[1] > 50:
        # Embedding boyutunu tahmin et (text features genelde 13 özellik)
        estimated_embedding_dim = X_train.shape[1] - 13
        # 384, 512, 768 gibi yaygın embedding boyutlarına yakın olanı seç
        common_dims = [384, 512, 768]
        embedding_dim = min(common_dims, key=lambda x: abs(x - estimated_embedding_dim))
        
        print(f"  Tespit edilen embedding boyutu: {embedding_dim} (toplam özellik: {X_train.shape[1]})")
        
        if use_pca:
            # Hibrit yapı: PCA sadece embedding'lere
            X_train_processed, X_test_processed, pca_model = apply_hybrid_pca(
                X_train, X_test, embedding_dim=embedding_dim
            )
        else:
            X_train_processed, X_test_processed = X_train, X_test
    else:
        # Özellik sayısı azsa PCA gerekmez
        if use_pca:
            X_train_processed, X_test_processed, pca_model = apply_pca_if_needed(X_train, X_test)
        else:
            X_train_processed, X_test_processed = X_train, X_test
    
    # Küçük veri setleri için basit modellere öncelik ver
    n_samples = X_train_processed.shape[0]
    use_simple_models = n_samples < 100  # 100'den az örnek varsa basit modeller
    
    if use_advanced:
        # Küçük veri setleri için Linear modellere öncelik, Random Forest'ı kaldır
        if use_simple_models:
            print(f"\n  ⚠ Küçük veri seti ({n_samples} örnek): Basit (Linear) modellere öncelik veriliyor")
            models = {
                'Ridge Regression': Ridge(alpha=1.0),
                'Lasso Regression': Lasso(alpha=0.1),
                'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
                'Ridge (α=0.1)': Ridge(alpha=0.1),
                'Ridge (α=10)': Ridge(alpha=10.0),
                'ElasticNet (α=0.01)': ElasticNet(alpha=0.01, l1_ratio=0.5),
                'ElasticNet (α=1.0)': ElasticNet(alpha=1.0, l1_ratio=0.5),
            }
        else:
            # Büyük veri setleri için tüm modeller
            models = {
                'Ridge Regression': Ridge(alpha=1.0),
                'Lasso Regression': Lasso(alpha=0.1),
                'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1),
                'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'AdaBoost': AdaBoostRegressor(n_estimators=50, random_state=42),
                'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1)
            }
        
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = XGBRegressor(n_estimators=100, random_state=42, n_jobs=1)
        if LIGHTGBM_AVAILABLE:
            models['LightGBM'] = LGBMRegressor(n_estimators=100, random_state=42, n_jobs=1)
        
        # GridSearchCV parametreleri (küçük veri setleri için Linear modellere odaklan)
        param_grids = {
            'Ridge Regression': {
                'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
            },
            'Lasso Regression': {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
            },
            'ElasticNet': {
                'alpha': [0.001, 0.01, 0.1, 1.0],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
            },
            'Ridge (α=0.1)': {
                'alpha': [0.05, 0.1, 0.5]
            },
            'Ridge (α=10)': {
                'alpha': [5.0, 10.0, 20.0]
            },
            'ElasticNet (α=0.01)': {
                'alpha': [0.005, 0.01, 0.05],
                'l1_ratio': [0.3, 0.5, 0.7]
            },
            'ElasticNet (α=1.0)': {
                'alpha': [0.5, 1.0, 2.0],
                'l1_ratio': [0.3, 0.5, 0.7]
            }
        }
        
        # Büyük veri setleri için Random Forest ve Gradient Boosting parametreleri
        if not use_simple_models:
            param_grids.update({
                'Random Forest': {
                    'n_estimators': [100, 200] if X_train_processed.shape[0] > 50 else [100],
                    'max_depth': [None, 10, 20] if X_train_processed.shape[0] > 50 else [None, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'Gradient Boosting': {
                    'n_estimators': [100, 200] if X_train_processed.shape[0] > 50 else [100],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1]
                }
            })
        
        results = {}
        best_model = None
        best_cv_r2 = -np.inf
        best_name = None
        
        print(f"\n{'='*70}")
        print(f"FARKLI ALGORİTMALAR TEST EDİLİYOR ({cv_folds}-Fold Cross Validation)...")
        print(f"{'='*70}")
        
        # Örnek ağırlıklarını hesapla
        sample_weights_train = None
        if use_sample_weights:
            sample_weights_train = calculate_sample_weights(y_train)
            print(f"  Örnek ağırlandırma aktif: Az bulunan puanlar ağırlandırıldı")
            print(f"  Ağırlık aralığı: [{sample_weights_train.min():.4f}, {sample_weights_train.max():.4f}]")
        
        # Cross Validation için splitter oluştur
        if use_stratified:
            cv_splitter, y_categorical = create_stratified_folds(y_train, n_splits=cv_folds, random_state=42)
            if isinstance(cv_splitter, StratifiedKFold):
                print(f"  Stratified K-Fold kullanılıyor (skor dağılımı korunuyor)")
                kfold = cv_splitter
                cv_labels = y_categorical
            else:
                print(f"  Normal K-Fold kullanılıyor (StratifiedKFold için yeterli kategori yok)")
                kfold = cv_splitter
                cv_labels = None
        else:
            kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            cv_labels = None
            print(f"  Normal K-Fold kullanılıyor")
        
        for name, model in models.items():
            try:
                # K-Fold Cross Validation ile performans ölçümü (sadece train verisi üzerinde)
                if isinstance(kfold, StratifiedKFold) and cv_labels is not None:
                    cv_scores_r2 = cross_val_score(model, X_train_processed, y_train, cv=kfold.split(X_train_processed, cv_labels), 
                                                  scoring='r2', n_jobs=1)
                    cv_scores_rmse = -cross_val_score(model, X_train_processed, y_train, cv=kfold.split(X_train_processed, cv_labels), 
                                                     scoring='neg_root_mean_squared_error', n_jobs=1)
                    cv_scores_mae = -cross_val_score(model, X_train_processed, y_train, cv=kfold.split(X_train_processed, cv_labels), 
                                                     scoring='neg_mean_absolute_error', n_jobs=1)
                else:
                    cv_scores_r2 = cross_val_score(model, X_train_processed, y_train, cv=kfold, scoring='r2', n_jobs=1)
                    cv_scores_rmse = -cross_val_score(model, X_train_processed, y_train, cv=kfold, scoring='neg_root_mean_squared_error', n_jobs=1)
                    cv_scores_mae = -cross_val_score(model, X_train_processed, y_train, cv=kfold, scoring='neg_mean_absolute_error', n_jobs=1)
                
                # Cross validation ortalamaları
                cv_r2_mean = cv_scores_r2.mean()
                cv_r2_std = cv_scores_r2.std()
                cv_rmse_mean = cv_scores_rmse.mean()
                cv_mae_mean = cv_scores_mae.mean()
                
                # GridSearchCV ile hiperparametre optimizasyonu (eğer istenirse ve model destekliyorsa)
                if use_gridsearch and name in param_grids and X_train_processed.shape[0] > 30:
                    print(f"    GridSearchCV ile hiperparametre optimizasyonu yapılıyor...")
                    grid_search = GridSearchCV(
                        estimator=model,
                        param_grid=param_grids[name],
                        cv=min(3, cv_folds),  # Küçük veri setleri için daha az fold
                        scoring='r2',
                        n_jobs=1,
                        verbose=0
                    )
                    if use_sample_weights and sample_weights_train is not None:
                        grid_search.fit(X_train_processed, y_train, sample_weight=sample_weights_train)
                    else:
                        grid_search.fit(X_train_processed, y_train)
                    model = grid_search.best_estimator_
                    print(f"    En iyi parametreler: {grid_search.best_params_}")
                else:
                    # Modeli tüm eğitim verisi ile eğit (sample weights ile)
                    if use_sample_weights and sample_weights_train is not None:
                        # Model sample_weight destekliyorsa kullan
                        if hasattr(model, 'fit') and 'sample_weight' in model.fit.__code__.co_varnames:
                            model.fit(X_train_processed, y_train, sample_weight=sample_weights_train)
                        else:
                            model.fit(X_train_processed, y_train)
                    else:
                        model.fit(X_train_processed, y_train)
                
                # Test seti üzerinde tahmin
                y_pred = model.predict(X_test_processed)
                
                # Test seti metrikleri
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                test_mae = mean_absolute_error(y_test, y_pred)
                test_r2 = r2_score(y_test, y_pred)
                
                results[name] = {
                    'model': model,
                    'cv_r2_mean': cv_r2_mean,
                    'cv_r2_std': cv_r2_std,
                    'cv_rmse_mean': cv_rmse_mean,
                    'cv_mae_mean': cv_mae_mean,
                    'test_rmse': test_rmse,
                    'test_mae': test_mae,
                    'test_r2': test_r2,
                    'y_pred': y_pred
                }
                
                print(f"\n{name}:")
                print(f"  Cross Validation ({cv_folds}-Fold):")
                print(f"    R²: {cv_r2_mean:.4f} (+/- {cv_r2_std*2:.4f})")
                print(f"    RMSE: {cv_rmse_mean:.4f}")
                print(f"    MAE: {cv_mae_mean:.4f}")
                print(f"  Test Seti:")
                print(f"    R²: {test_r2:.4f}")
                print(f"    RMSE: {test_rmse:.4f}")
                print(f"    MAE: {test_mae:.4f}")
                
                # En iyi modeli cross validation R² skoruna göre seç
                if cv_r2_mean > best_cv_r2:
                    best_cv_r2 = cv_r2_mean
                    best_model = model
                    best_name = name
                    y_pred = y_pred
                    
            except Exception as e:
                print(f"\n{name}: Hata - {str(e)}")
        
        print(f"\n{'='*70}")
        print(f"EN İYİ MODEL: {best_name} (CV R²={best_cv_r2:.4f})")
        print(f"{'='*70}")
        model = best_model
    else:
        # Basit Random Forest (cross validation ile)
        print(f"\nRandom Forest eğitiliyor ({cv_folds}-Fold Cross Validation ile)...")
        model = RandomForestRegressor(n_estimators=200, max_depth=20, 
                                     min_samples_split=5, min_samples_leaf=2,
                                     random_state=42, n_jobs=1)
        
        # Örnek ağırlıklarını hesapla
        sample_weights_train = None
        if use_sample_weights:
            sample_weights_train = calculate_sample_weights(y_train)
            print(f"  Örnek ağırlandırma aktif")
        
        # Cross validation skorları
        if use_stratified:
            cv_splitter, y_categorical = create_stratified_folds(y_train, n_splits=cv_folds, random_state=42)
            if isinstance(cv_splitter, StratifiedKFold) and y_categorical is not None:
                kfold = cv_splitter
                cv_scores = cross_val_score(model, X_train, y_train, cv=kfold.split(X_train, y_categorical), 
                                           scoring='r2', n_jobs=1)
            else:
                kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                cv_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='r2', n_jobs=1)
        else:
            kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='r2', n_jobs=1)
        
        print(f"  Cross Validation R²: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        # Modeli eğit (sample weights ile)
        if use_sample_weights and sample_weights_train is not None:
            model.fit(X_train, y_train, sample_weight=sample_weights_train)
        else:
            model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    # Final metrikler
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\nFinal Model Performansı (Test Seti):")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R² Score: {r2:.4f}")
    
    return model, X_test_processed, y_test, y_pred, pca_model

def train_score_prediction_model(X, y, test_size=0.2, random_state=42, use_advanced=True):
    """
    Essay embedding'lerini kullanarak skor tahmin modeli eğitir.
    Gelişmiş mod: Birden fazla algoritma dener ve en iyisini seçer.
    
    Args:
        X (numpy.ndarray): Essay embedding'leri (veya birleştirilmiş özellikler)
        y (numpy.ndarray): Hedef skorlar
        test_size (float): Test seti oranı
        random_state (int): Random seed
        use_advanced (bool): Gelişmiş mod kullan (birden fazla algoritma dener)
    
    Returns:
        tuple: (model, X_test, y_test, y_pred) - Eğitilmiş model ve test sonuçları
    """
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"\nModel eğitiliyor...")
    print(f"  Eğitim seti: {X_train.shape[0]} örnek")
    print(f"  Test seti: {X_test.shape[0]} örnek")
    print(f"  Özellik sayısı: {X_train.shape[1]}")
    
    if use_advanced:
        # Birden fazla algoritma dene
        model_results = train_multiple_models(X, y, test_size, random_state)
        model = model_results['best_model']
        y_pred = model.predict(X_test)
        print(f"\n  Seçilen model: {model_results['best_name']}")
    else:
        # Basit Random Forest
        model = RandomForestRegressor(n_estimators=200, max_depth=20, 
                                     min_samples_split=5, min_samples_leaf=2,
                                     random_state=random_state, n_jobs=1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    # Metrikler (multi-output için her output için ayrı hesapla)
    if len(y.shape) > 1:
        print(f"\nModel Performansı (Her Output İçin):")
        for i in range(y.shape[1]):
            mse = mean_squared_error(y_test[:, i], y_pred[:, i])
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
            r2 = r2_score(y_test[:, i], y_pred[:, i])
            print(f"  Output {i+1}: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
        
        # Ortalama metrikler
        mse_all = mean_squared_error(y_test, y_pred, multioutput='uniform_average')
        rmse_all = np.sqrt(mse_all)
        mae_all = mean_absolute_error(y_test, y_pred, multioutput='uniform_average')
        r2_all = r2_score(y_test, y_pred, multioutput='uniform_average')
        print(f"\n  Ortalama: RMSE={rmse_all:.4f}, MAE={mae_all:.4f}, R²={r2_all:.4f}")
    else:
        # Tek output için
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"\nModel Performansı:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R² Score: {r2:.4f}")
    
    return model, X_test, y_test, y_pred

def save_model_results_to_txt(trained_models, model_results, df_clean, filename='model_results.txt'):
    """
    Model eğitimi sonuçlarını detaylı bir txt dosyasına kaydeder.
    
    Args:
        trained_models (dict): Eğitilmiş modeller dictionary'si
        model_results (dict): Model sonuçları dictionary'si
        df_clean (pd.DataFrame): Temizlenmiş veri DataFrame'i
        filename (str): Kaydedilecek dosya adı
    """
    from datetime import datetime
    
    output_lines = []
    output_lines.append("="*80)
    output_lines.append("SBERT MODEL EĞİTİMİ SONUÇ RAPORU")
    output_lines.append("="*80)
    output_lines.append(f"Oluşturulma Tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output_lines.append(f"Model: SBERT (all-mpnet-base-v2)")
    output_lines.append(f"Rater: RA (Sadece RA skorları kullanıldı)")
    output_lines.append("="*80)
    output_lines.append("")
    
    # Genel bilgiler
    output_lines.append("GENEL BİLGİLER")
    output_lines.append("-"*80)
    output_lines.append(f"Toplam Essay Sayısı: {len(df_clean)}")
    output_lines.append(f"Eğitilen Model Sayısı: {len(trained_models)}")
    if len(trained_models) > 0:
        output_lines.append(f"Eğitilen Kriterler: {', '.join(trained_models.keys())}")
    output_lines.append("")
    
    # Her kriter için detaylı sonuçlar
    if len(trained_models) > 0:
        output_lines.append("="*80)
        output_lines.append("KRİTER BAZINDA DETAYLI SONUÇLAR")
        output_lines.append("="*80)
        output_lines.append("")
        
        for criterion in CRITERIA:
            if criterion in model_results:
                result = model_results[criterion]
                y_test = result['y_test']
                y_pred = result['y_pred']
                
                # Metrikler
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                output_lines.append("-"*80)
                output_lines.append(f"KRİTER: {criterion} (RA)")
                output_lines.append("-"*80)
                output_lines.append(f"Örnek Sayısı: {result['n_samples']}")
                output_lines.append(f"Skor Sütunu: {result['ra_col']}")
                output_lines.append("")
                
                # Cross Validation sonuçları
                output_lines.append("Cross Validation (5-Fold) Sonuçları:")
                if result.get('cv_r2_mean') is not None:
                    output_lines.append(f"  R² Skoru: {result['cv_r2_mean']:.4f} (+/- {result['cv_r2_std']*2:.4f})")
                    output_lines.append(f"  RMSE: {result['cv_rmse_mean']:.4f}")
                    output_lines.append(f"  MAE: {result['cv_mae_mean']:.4f}")
                else:
                    output_lines.append("  Cross validation skorları mevcut değil")
                output_lines.append("")
                
                # Test seti sonuçları
                output_lines.append("Test Seti Performansı:")
                output_lines.append(f"  RMSE: {rmse:.4f}")
                output_lines.append(f"  MAE: {mae:.4f}")
                output_lines.append(f"  R² Skoru: {r2:.4f}")
                output_lines.append("")
                
                # İstatistikler
                output_lines.append("İstatistikler:")
                output_lines.append(f"  Gerçek RA Skorları:")
                output_lines.append(f"    - Ortalama: {y_test.mean():.4f}")
                output_lines.append(f"    - Standart Sapma: {y_test.std():.4f}")
                output_lines.append(f"    - Minimum: {y_test.min():.4f}")
                output_lines.append(f"    - Maksimum: {y_test.max():.4f}")
                output_lines.append(f"  Tahmin Edilen Skorlar:")
                output_lines.append(f"    - Ortalama: {y_pred.mean():.4f}")
                output_lines.append(f"    - Standart Sapma: {y_pred.std():.4f}")
                output_lines.append(f"    - Minimum: {y_pred.min():.4f}")
                output_lines.append(f"    - Maksimum: {y_pred.max():.4f}")
                output_lines.append("")
                
                # Feature importance (varsa)
                if result.get('feature_importance') is not None:
                    feature_importance = result['feature_importance']
                    top_indices = np.argsort(feature_importance)[-10:][::-1]
                    output_lines.append("En Önemli 10 Özellik:")
                    for i, idx in enumerate(top_indices, 1):
                        output_lines.append(f"  {i}. Özellik {idx+1}: {feature_importance[idx]:.4f}")
                    output_lines.append("")
                
                # PCA bilgisi (varsa)
                if result.get('pca_model') is not None:
                    pca = result['pca_model']
                    explained_variance = np.sum(pca.explained_variance_ratio_)
                    output_lines.append(f"PCA Bilgisi:")
                    output_lines.append(f"  Bileşen Sayısı: {pca.n_components_}")
                    output_lines.append(f"  Açıklanan Varyans: {explained_variance:.4f} ({explained_variance*100:.2f}%)")
                    output_lines.append("")
                
                output_lines.append("")
    else:
        output_lines.append("="*80)
        output_lines.append("UYARI: Hiçbir model eğitilemedi!")
        output_lines.append("="*80)
        output_lines.append("Olası nedenler:")
        output_lines.append("  1. Veri setinde yeterli örnek yok (her kriter için en az 10 örnek gerekli)")
        output_lines.append("  2. RA skor sütunları bulunamadı veya boş")
        output_lines.append("  3. Model eğitimi sırasında hata oluştu")
        output_lines.append("")
    
    # Özet tablo
    if len(trained_models) > 0:
        output_lines.append("="*80)
        output_lines.append("ÖZET TABLO")
        output_lines.append("="*80)
        output_lines.append(f"{'Kriter':<15} {'Örnek':<10} {'CV R²':<12} {'Test R²':<12} {'RMSE':<10} {'MAE':<10}")
        output_lines.append("-"*80)
        
        for criterion in CRITERIA:
            if criterion in model_results:
                result = model_results[criterion]
                y_test = result['y_test']
                y_pred = result['y_pred']
                
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                cv_r2 = result.get('cv_r2_mean', 0.0)
                if cv_r2 is None:
                    cv_r2 = 0.0
                
                output_lines.append(
                    f"{criterion:<15} {result['n_samples']:<10} "
                    f"{cv_r2:.4f}{'':<8} {r2:.4f}{'':<8} {rmse:.4f}{'':<6} {mae:.4f}"
                )
        
        output_lines.append("")
    
    # Dosyaya yaz
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))
        print(f"\n✓ Model sonuçları '{filename}' dosyasına kaydedildi.")
    except Exception as e:
        print(f"\n✗ Model sonuçları kaydedilirken hata oluştu: {str(e)}")

def save_trained_models(trained_models, model_results, models_dir='saved_models'):
    """
    Eğitilmiş modelleri ve gerekli tüm bileşenleri dosyaya kaydeder.
    
    Args:
        trained_models (dict): Eğitilmiş modeller dictionary'si
        model_results (dict): Model sonuçları dictionary'si
        models_dir (str): Modellerin kaydedileceği dizin
    """
    from datetime import datetime
    
    # Dizin oluştur
    os.makedirs(models_dir, exist_ok=True)
    
    # Her kriter için model ve gerekli bileşenleri kaydet
    saved_files = []
    
    for criterion in trained_models.keys():
        try:
            # Model ve ilgili bileşenleri birleştir
            model_data = {
                'model': trained_models[criterion],
                'criterion': criterion,
                'feature_scaler': globals().get('feature_scalers', {}).get(criterion),
                'golden_vector': globals().get('golden_vectors', {}).get(criterion),
                'pca_model': globals().get('pca_models', {}).get(criterion),
                'model_info': model_results.get(criterion, {}),
                'saved_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'sbert_model_name': 'all-mpnet-base-v2'
            }
            
            # Dosya adı
            filename = os.path.join(models_dir, f'model_{criterion}.pkl')
            
            # Modeli kaydet
            with open(filename, 'wb') as f:
                pickle.dump(model_data, f)
            
            saved_files.append(filename)
            print(f"  ✓ {criterion} modeli kaydedildi: {filename}")
            
        except Exception as e:
            print(f"  ✗ {criterion} modeli kaydedilirken hata: {str(e)}")
    
    # Genel bilgi dosyası
    info = {
        'criteria': list(trained_models.keys()),
        'saved_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_count': len(trained_models),
        'sbert_model_name': 'all-mpnet-base-v2'
    }
    
    info_filename = os.path.join(models_dir, 'models_info.pkl')
    with open(info_filename, 'wb') as f:
        pickle.dump(info, f)
    
    print(f"\n✓ Toplam {len(saved_files)} model kaydedildi: {models_dir}/")
    print(f"✓ Model bilgileri: {info_filename}")
    
    return saved_files

def augment_data_simple(texts, scores, augmentation_factor=5):
    """
    Basit veri çoğaltma (Data Augmentation) yöntemi.
    Küçük veri setleri için essay'leri çoğaltır.
    NOT: Daha gelişmiş augmentation için back-translation veya synonym replacement kullanılabilir.
    
    Args:
        texts (list): Essay metinleri
        scores (numpy.ndarray): Essay skorları
        augmentation_factor (int): Her essay için kaç kopya oluşturulacak (varsayılan: 5)
    
    Returns:
        tuple: (augmented_texts, augmented_scores) - Çoğaltılmış metinler ve skorlar
    """
    np.random.seed(42)  # Tekrarlanabilirlik için
    augmented_texts = list(texts)
    augmented_scores = list(scores)
    
    print(f"\n  Veri Çoğaltma (Data Augmentation):")
    print(f"    Orijinal veri sayısı: {len(texts)}")
    
    # Her essay için augmentation_factor kadar kopya oluştur
    for i, (text, score) in enumerate(zip(texts, scores)):
        for j in range(augmentation_factor - 1):
            augmented_text = text
            
            # 1. Cümleleri ayır ve sıralarını değiştir (rastgele)
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
            
            if len(sentences) > 1 and np.random.random() > 0.5:
                # İlk ve son cümleyi sabit tut, ortadakileri karıştır
                if len(sentences) > 2:
                    middle = sentences[1:-1]
                    np.random.shuffle(middle)
                    sentences = [sentences[0]] + middle + [sentences[-1]]
                augmented_text = '. '.join(sentences) + '.'
            
            # 2. Boşlukları normalize et
            augmented_text = ' '.join(augmented_text.split())
            
            # 3. Noktalama varyasyonları
            if np.random.random() > 0.7:
                # Bazı noktalama işaretlerini değiştir
                augmented_text = augmented_text.replace('!', '.').replace('?', '.')
            
            # 4. Metin uzunluğu kontrolü
            if len(augmented_text) < len(text) * 0.5 or len(augmented_text) > len(text) * 1.5:
                augmented_text = text  # Çok farklıysa orijinali kullan
            
            augmented_texts.append(augmented_text)
            augmented_scores.append(score)
    
    print(f"    Çoğaltılmış veri sayısı: {len(augmented_texts)}")
    print(f"    Artış oranı: {len(augmented_texts) / len(texts):.2f}x")
    
    return augmented_texts, np.array(augmented_scores)

def find_similar_essays(query_text, all_texts, embeddings, model, top_k=5):
    """
    Verilen bir metne en benzer essay'leri bulur.
    
    Args:
        query_text (str): Arama yapılacak metin
        all_texts (list): Tüm essay metinleri
        embeddings (numpy.ndarray): Tüm essay embedding'leri
        model (SentenceTransformer): SBERT modeli
        top_k (int): Döndürülecek en benzer essay sayısı
    
    Returns:
        list: En benzer essay'lerin indeksleri ve benzerlik skorları
    """
    # Query embedding'i oluştur
    query_embedding = model.encode([query_text], convert_to_numpy=True)[0]
    
    # Cosine similarity hesapla
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    
    # En benzer essay'leri bul
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            'index': idx,
            'similarity': float(similarities[idx]),
            'text': all_texts[idx][:200] + '...' if len(all_texts[idx]) > 200 else all_texts[idx]
        })
    
    return results

# SBERT Embedding'leri Oluştur
print("\n" + "="*70)
print("SBERT Embedding'leri Oluşturuluyor...")
print("="*70)
print("ℹ Windows uyumluluğu için paralel işleme devre dışı (n_jobs=1)")
print("  Bu, işlem süresini artırabilir ama Windows DLL hatalarını önler.")

# Essay metinlerini temizle (boş olanları filtrele)
valid_indices = [i for i, text in enumerate(df['essay']) if text and len(text.strip()) > 0]
df_clean = df.iloc[valid_indices].copy().reset_index(drop=True)

if len(df_clean) > 0:
    # SBERT modelini yükle (embedding ve tutarlılık analizi için)
    print("\nSBERT modeli yükleniyor: all-mpnet-base-v2")
    sbert_model = SentenceTransformer('all-mpnet-base-v2')
    
    # Embedding'leri oluştur
    essay_embeddings = create_sbert_embeddings(
        df_clean['essay'].tolist(),
        model_name='all-mpnet-base-v2'  # İngilizce model
    )
    
    # Özellik mühendisliği: Metin özelliklerini çıkar (tutarlılık özellikleri dahil)
    # NOT: Altın Vektör benzerliği her kriter için ayrı hesaplanacak (kriter bazında)
    print("\n" + "="*70)
    print("ÖZELLİK MÜHENDİSLİĞİ: Metin Özellikleri Çıkarılıyor...")
    print("="*70)
    print("Tutarlılık (Coherence) analizi yapılıyor (Organisation ve Analysis için)...")
    print("NOT: Altın Vektör benzerliği her kriter için ayrı hesaplanacak (konuya sadakat ölçümü)")
    text_features_base = extract_text_features(df_clean['essay'].tolist(), sbert_model=sbert_model, golden_vector_similarity=None)
    print(f"Temel metin özellikleri oluşturuldu. Şekil: {text_features_base.shape}")
    print(f"Özellikler:")
    print(f"  - Temel özellikler (10): Karakter sayısı, Kelime sayısı, Cümle sayısı, vb.")
    print(f"  - Tutarlılık özellikleri (2): Ortalama tutarlılık, Minimum tutarlılık (cümleler arası geçiş)")
    print(f"  - Altın Vektör benzerliği (1): Konuya sadakat ölçümü (Analysis için) - Her kriter için ayrı hesaplanacak")
    
    # NOT: Özellik birleştirme train_test_split'ten SONRA yapılacak (veri sızıntısını önlemek için)
    # Şimdilik sadece embedding'leri ve metin özelliklerini ayrı ayrı sakla
    print(f"Özellikler hazırlandı:")
    print(f"  - SBERT embedding'leri: {essay_embeddings.shape[1]} özellik")
    print(f"  - Temel metin özellikleri: {text_features_base.shape[1]} özellik")
    print(f"  - Toplam (Altın Vektör eklendikten sonra): {essay_embeddings.shape[1] + text_features_base.shape[1] + 1} özellik")
    print(f"\n⚠ ÖNEMLİ: Scaling işlemi train_test_split'ten SONRA yapılacak (veri sızıntısını önlemek için)")
    
    # Embedding'leri DataFrame'e ekle
    embedding_columns = [f'embedding_{i}' for i in range(essay_embeddings.shape[1])]
    df_embeddings = pd.DataFrame(essay_embeddings, columns=embedding_columns)
    df_with_embeddings = pd.concat([df_clean.reset_index(drop=True), df_embeddings], axis=1)
    
    print(f"\nEmbedding'ler DataFrame'e eklendi. Toplam sütun: {len(df_with_embeddings.columns)}")
    
    # Tüm skorlar için model eğitimi (Sadece RA skorları kullanılıyor)
    print("\n" + "="*70)
    print("Tüm Skorlar İçin Model Eğitimi Başlatılıyor...")
    print(f"Rater: RA (Sadece RA skorları kullanılıyor)")
    print(f"Kriterler: {', '.join(CRITERIA)}")
    print("="*70)
    
    # Eğitilmiş modelleri saklamak için dictionary: {criterion: model}
    trained_models = {}
    model_results = {}
    
    # Her kriter için (Sadece RA skorları kullanılıyor)
    for criterion in CRITERIA:
        print(f"\n{'='*70}")
        print(f"KRİTER: {criterion}")
        print(f"{'='*70}")
        
        # Sadece RA skorlarını kullan
        ra_col = f"RA_{criterion}"
        
        # RA sütununun mevcut olduğu satırları bul (boolean mask)
        valid_mask = df_clean[ra_col].notna()
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) > 10:  # Yeterli veri varsa
            # Embedding'leri ve metin özelliklerini ayrı ayrı al
            X_embeddings = essay_embeddings[valid_indices]
            X_text_features_base = text_features_base[valid_indices]
            
            # Sadece RA skorlarını al
            y_ra = df_clean.loc[valid_mask, ra_col].values
            
            print(f"  RA skorları kullanılıyor: {len(y_ra)} örnek")
            print(f"  Skor aralığı: [{y_ra.min():.2f}, {y_ra.max():.2f}]")
            
            try:
                # KRİTİK: ÖNCE train_test_split yap (veri sızıntısını önlemek için)
                # Altın Vektör'ü sadece eğitim verisi üzerinden hesaplayacağız
                # Essay metinlerini de sakla (data augmentation için)
                essay_texts_for_criterion = df_clean.loc[valid_mask, 'essay'].tolist()
                
                X_emb_train, X_emb_test, X_text_train_base, X_text_test_base, y_train, y_test, texts_train, texts_test = train_test_split(
                    X_embeddings, X_text_features_base, y_ra, essay_texts_for_criterion, 
                    test_size=0.2, random_state=42
                )
                
                # Veri çoğaltma (Data Augmentation) - Sadece train setine uygula
                # Veri sayısını 5 katına çıkar (her essay için 4 yeni kopya = toplam 5x)
                augmentation_factor = 5
                print(f"\n  Veri Çoğaltma Aktif: {len(y_train)} örnek -> {len(y_train) * augmentation_factor} örnek (5x)")
                
                if augmentation_factor > 1:
                        texts_train_aug, y_train_aug = augment_data_simple(
                            texts_train, y_train, augmentation_factor=augmentation_factor
                        )
                        # Augmented metinler için embedding ve text features oluştur
                        print(f"  Augmented metinler için embedding oluşturuluyor...")
                        X_emb_train_aug = sbert_model.encode(texts_train_aug[len(texts_train):], 
                                                             convert_to_numpy=True, 
                                                             show_progress_bar=False)
                        X_text_train_aug_base = extract_text_features(
                            texts_train_aug[len(texts_train):], 
                            sbert_model=sbert_model, 
                            golden_vector_similarity=None
                        )
                        
                        # Orijinal ve augmented verileri birleştir
                        X_emb_train = np.vstack([X_emb_train, X_emb_train_aug])
                        X_text_train_base = np.vstack([X_text_train_base, X_text_train_aug_base])
                        y_train = np.hstack([y_train, y_train_aug[len(y_train):]])
                        
                        print(f"  Augmentation sonrası train seti: {len(y_train)} örnek")
                
                # Altın Vektör'ü SADECE EĞİTİM SETİNDEN hesapla (veri sızıntısını önlemek için)
                print(f"\n  Altın Vektör (Centroid) hesaplanıyor (SADECE eğitim verisi kullanılıyor - veri sızıntısı önlendi)...")
                threshold = np.percentile(y_train, 80)  # Top %20
                top_indices_train = np.where(y_train >= threshold)[0]
                if len(top_indices_train) == 0:
                    top_indices_train = [np.argmax(y_train)]
                
                # Altın Vektör: Sadece eğitim setindeki en iyi essay'lerin ortalaması
                golden_vector = np.mean(X_emb_train[top_indices_train], axis=0)
                
                # Train ve test setleri için Altın Vektör benzerliğini hesapla
                X_train_golden_sim = cosine_similarity(X_emb_train, [golden_vector]).flatten()
                X_test_golden_sim = cosine_similarity(X_emb_test, [golden_vector]).flatten()
                
                print(f"  Altın Vektör benzerliği hesaplandı (Train ort: {X_train_golden_sim.mean():.4f}, Test ort: {X_test_golden_sim.mean():.4f})")
                
                # Altın Vektör benzerliğini metin özelliklerine ekle
                X_text_train = X_text_train_base.copy()
                X_text_train[:, -1] = X_train_golden_sim  # Son sütunu güncelle
                
                X_text_test = X_text_test_base.copy()
                X_text_test[:, -1] = X_test_golden_sim  # Son sütunu güncelle
                
                # SONRA scaling yap (sadece eğitim verisi üzerinde fit)
                feature_scaler = StandardScaler()
                X_text_train_scaled = feature_scaler.fit_transform(X_text_train)
                X_text_test_scaled = feature_scaler.transform(X_text_test)  # Sadece transform, fit değil!
                
                # Özellikleri birleştir
                X_train_combined = np.hstack([X_emb_train, X_text_train_scaled])
                X_test_combined = np.hstack([X_emb_test, X_text_test_scaled])
                
                print(f"\n  Veri sızıntısı önlendi: Scaling train_test_split'ten SONRA yapıldı")
                print(f"  Eğitim seti: {X_train_combined.shape[0]} örnek, {X_train_combined.shape[1]} özellik")
                print(f"  Test seti: {X_test_combined.shape[0]} örnek, {X_test_combined.shape[1]} özellik")
                
                # Gelişmiş model eğitimi (birden fazla algoritma dener, K-Fold Cross Validation ile)
                print(f"\n  Gelişmiş model eğitimi başlatılıyor (5-Fold Cross Validation, Stratified + Sample Weights + PCA + GridSearch ile)...")
                model_rf, X_test_final, y_test_final, y_pred, pca_model = train_score_prediction_model_with_data(
                    X_train_combined, X_test_combined, y_train, y_test, 
                    use_advanced=True, cv_folds=5, use_stratified=True, use_sample_weights=True,
                    use_pca=True, use_gridsearch=True
                )
                
                # PCA modelini sakla (tahmin için)
                if 'pca_models' not in globals():
                    globals()['pca_models'] = {}
                globals()['pca_models'][criterion] = pca_model
                
                # Scaler'ı ve golden_vector'ü sakla (tahmin fonksiyonları için)
                if 'feature_scalers' not in globals():
                    globals()['feature_scalers'] = {}
                globals()['feature_scalers'][criterion] = feature_scaler
                
                # Golden vector'ü sakla (tahmin için) - Zaten train setinden hesaplanmış
                if 'golden_vectors' not in globals():
                    globals()['golden_vectors'] = {}
                globals()['golden_vectors'][criterion] = golden_vector  # Zaten train setinden hesaplanmış
                
                # Cross validation skorlarını almak için train_multiple_models'i çağır
                # (sadece cross validation skorları için, model zaten eğitildi)
                print(f"\n  Cross Validation skorları hesaplanıyor...")
                cv_results = train_multiple_models(
                    X_train_combined, y_train, test_size=0.2, random_state=42, cv_folds=5
                )
                
                # En iyi modelin cross validation skorlarını al
                best_cv_info = None
                if cv_results['best_name'] in cv_results['all_results']:
                    best_cv_info = cv_results['all_results'][cv_results['best_name']]
                
                # Feature importance görselleştirmesi (eğer model destekliyorsa)
                feature_importance = None
                if hasattr(model_rf, 'feature_importances_'):
                    feature_importance = model_rf.feature_importances_
                    print(f"\n  En önemli 10 özellik:")
                    if pca_model is not None:
                        # PCA kullanıldıysa, özellik isimleri PCA bileşenleri
                        top_indices = np.argsort(feature_importance)[-10:][::-1]
                        for i, idx in enumerate(top_indices, 1):
                            print(f"    {i}. PCA Bileşen {idx+1}: {feature_importance[idx]:.4f}")
                    else:
                        # PCA kullanılmadıysa, özellik isimleri
                        top_indices = np.argsort(feature_importance)[-10:][::-1]
                        for i, idx in enumerate(top_indices, 1):
                            if idx < X_emb_train.shape[1]:
                                print(f"    {i}. SBERT Embedding {idx+1}: {feature_importance[idx]:.4f}")
                            else:
                                feature_names = ['char_count', 'word_count', 'sentence_count', 'avg_word_length', 
                                               'avg_sentence_length', 'punctuation_count', 'uppercase_ratio', 
                                               'digit_count', 'special_char_count', 'lexical_diversity',
                                               'avg_coherence', 'min_coherence', 'golden_vector_similarity']
                                feature_idx = idx - X_emb_train.shape[1]
                                if feature_idx < len(feature_names):
                                    print(f"    {i}. {feature_names[feature_idx]}: {feature_importance[idx]:.4f}")
                
                # Modeli ve sonuçları kaydet
                trained_models[criterion] = model_rf
                model_results[criterion] = {
                    'model': model_rf,
                    'X_test': X_test_final,
                    'y_test': y_test_final,  # Shape: (n_samples,) - RA skorları
                    'y_pred': y_pred,  # Shape: (n_samples,) - RA skor tahmini
                    'n_samples': len(X_embeddings),
                    'ra_col': ra_col,
                    'scaler': feature_scaler,
                    'golden_vector': golden_vector,
                    'pca_model': pca_model,
                    'feature_importance': feature_importance,
                    'cv_r2_mean': best_cv_info['cv_r2_mean'] if best_cv_info else None,
                    'cv_r2_std': best_cv_info['cv_r2_std'] if best_cv_info else None,
                    'cv_rmse_mean': best_cv_info['cv_rmse_mean'] if best_cv_info else None,
                    'cv_mae_mean': best_cv_info['cv_mae_mean'] if best_cv_info else None
                }
                
            except Exception as e:
                import traceback
                print(f"  ✗ Hata: {str(e)}")
                print(f"  Hata detayları:")
                traceback.print_exc()
        else:
            print(f"  ⚠ Yeterli veri yok ({len(valid_indices)} örnek), model eğitilemedi.")
            if ra_col not in df_clean.columns:
                print(f"    - {ra_col} sütunu bulunamadı")
            else:
                print(f"    - {ra_col} sütunu mevcut ama yeterli geçerli veri yok (en az 10 örnek gerekli)")
    
    # Özet rapor
    print("\n" + "="*70)
    print("MODEL EĞİTİMİ ÖZET RAPORU")
    print("="*70)
    
    if len(trained_models) > 0:
        print(f"\n✓ Toplam {len(trained_models)} model başarıyla eğitildi (Her kriter için RA skorları).\n")
        print(f"Eğitilen kriterler: {', '.join(trained_models.keys())}")
    else:
        print(f"\n⚠ UYARI: Hiçbir model eğitilemedi!")
        print(f"Olası nedenler:")
        print(f"  1. Veri setinde yeterli örnek yok (her kriter için en az 10 örnek gerekli)")
        print(f"  2. RA skor sütunları bulunamadı veya boş")
        print(f"  3. Model eğitimi sırasında hata oluştu (yukarıdaki hata mesajlarına bakın)")
        print(f"\nKontrol edin:")
        print(f"  - df_clean DataFrame'inde RA_* sütunları var mı?")
        print(f"  - Bu sütunlarda yeterli sayıda (>=10) geçerli veri var mı?")
        print(f"  - Veri okuma kısmı başarılı oldu mu?")
        print(f"\nMevcut sütunlar: {list(df_clean.columns)[:10]}...")  # İlk 10 sütunu göster
    
    # Her kriter için özet (sadece eğitilen modeller için)
    for criterion in CRITERIA:
        if criterion in model_results:
            result = model_results[criterion]
            y_test = result['y_test']  # RA skorları
            y_pred = result['y_pred']  # Tahmin edilen RA skorları
            
            print(f"\n{'='*70}")
            print(f"KRİTER: {criterion} (RA)")
            print(f"{'='*70}")
            print(f"  Örnek sayısı: {result['n_samples']}")
            
            # RA skorları için metrikler
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"\n  Cross Validation (5-Fold) - Daha Güvenilir Performans:")
            if result.get('cv_r2_mean') is not None:
                print(f"    - R²: {result['cv_r2_mean']:.4f} (+/- {result['cv_r2_std']*2:.4f})")
                print(f"    - RMSE: {result['cv_rmse_mean']:.4f}")
                print(f"    - MAE: {result['cv_mae_mean']:.4f}")
            else:
                print(f"    - Cross validation skorları mevcut değil")
            
            print(f"\n  Test Seti Performansı:")
            print(f"    - RMSE: {rmse:.4f}")
            print(f"    - MAE: {mae:.4f}")
            print(f"    - R²: {r2:.4f}")
            
            # Gerçek RA skorları ile tahmin edilen skorlar arasındaki fark
            print(f"\n  İstatistikler:")
            print(f"    - Gerçek RA ortalama: {y_test.mean():.4f}, std: {y_test.std():.4f}")
            print(f"    - Tahmin edilen ortalama: {y_pred.mean():.4f}, std: {y_pred.std():.4f}")
    
    # Model sonuçlarını txt dosyasına kaydet
    save_model_results_to_txt(trained_models, model_results, df_clean)
    
    # Modelleri dosyaya kaydet
    if len(trained_models) > 0:
        print("\n" + "="*70)
        print("MODELLERİN KAYDEDİLMESİ")
        print("="*70)
        save_trained_models(trained_models, model_results)
    
    # Örnek: Benzer essay bulma
    print("\n" + "="*70)
    print("Benzer Essay Arama Örneği...")
    print("="*70)
    
    if len(df_clean) > 0:
        # İlk essay'i query olarak kullan
        query_text = df_clean['essay'].iloc[0]
        print(f"\nQuery metni (ilk essay): {query_text[:200]}...")
        
        # Modeli yeniden yükle (veya önceki modeli kullan)
        sbert_model = SentenceTransformer('all-mpnet-base-v2')
        similar_essays = find_similar_essays(
            query_text, 
            df_clean['essay'].tolist(), 
            essay_embeddings,
            sbert_model,
            top_k=5
        )
        
        print(f"\nEn benzer {len(similar_essays)} essay:")
        for i, result in enumerate(similar_essays, 1):
            print(f"\n{i}. Benzerlik: {result['similarity']:.4f}")
            print(f"   Metin: {result['text']}")
    
    print("\n" + "="*70)
    print("SBERT Modelleme Tamamlandı!")
    print("="*70)
    print(f"\nKullanılabilir değişkenler:")
    print(f"  - df_with_embeddings: Embedding'lerle birlikte DataFrame")
    print(f"  - essay_embeddings: Tüm essay embedding'leri (numpy array)")
    print(f"  - X_combined: Birleştirilmiş özellikler (SBERT + metin özellikleri)")
    print(f"  - df_clean: Temizlenmiş veri DataFrame'i")
    print(f"  - trained_models: Eğitilmiş modeller dictionary'si (her kriter için tek model)")
    print(f"    Örnek: trained_models['TITLE'] -> TITLE kriteri için model")
    print(f"    Model çıktısı: RA skorları")
    print(f"  - model_results: Model sonuçları ve metrikleri dictionary'si")
    print(f"    Örnek: model_results['GRAMMAR'] -> GRAMMAR kriteri için sonuçlar")
    print(f"    - y_test: (n_samples,) -> RA skorları")
    print(f"    - y_pred: (n_samples,) -> Tahmin edilen RA skorları")
    print(f"  - feature_scaler: Metin özellikleri için StandardScaler")
    print(f"  - RATERS: Rater listesi ['RA'] (Sadece RA kullanılıyor)")
    print(f"  - CRITERIA: Kriter listesi ['TITLE', 'THESIS', ...]")
    
else:
    print("Hata: Embedding oluşturmak için yeterli veri yok!")

# ============================================================================
# Model Test Fonksiyonları
# ============================================================================

def predict_essay_score(essay_text, criterion='TITLE', sbert_model=None):
    """
    Yeni bir essay için RA skor tahmini yapar.
    
    Args:
        essay_text (str): Tahmin yapılacak essay metni
        criterion (str): Kriter adı (TITLE, THESIS, vb.)
        sbert_model: SBERT modeli (None ise yeniden yüklenir)
    
    Returns:
        float: Tahmin edilen RA skoru
    """
    if 'trained_models' not in globals() or criterion not in trained_models:
        print(f"Hata: {criterion} kriteri için model bulunamadı!")
        return None
    
    # SBERT modelini yükle (gerekirse)
    if sbert_model is None:
        sbert_model = SentenceTransformer('all-mpnet-base-v2')
    
    # Essay embedding'ini oluştur
    essay_embedding = sbert_model.encode([essay_text], convert_to_numpy=True)
    
    # Altın Vektör benzerliğini hesapla (eğitim sırasında saklanan golden_vector'ü kullan)
    golden_similarity = np.array([0.0])  # Varsayılan
    if 'golden_vectors' in globals() and criterion in globals()['golden_vectors']:
        golden_vector = globals()['golden_vectors'][criterion]
        golden_similarity = cosine_similarity(essay_embedding, [golden_vector]).flatten()
    else:
        print(f"⚠ Uyarı: {criterion} için golden_vector bulunamadı, 0.0 kullanılıyor.")
    
    # Metin özelliklerini çıkar (tutarlılık özellikleri ve golden_vector_similarity dahil)
    text_features = extract_text_features([essay_text], sbert_model=sbert_model, golden_vector_similarity=golden_similarity)
    
    # Kriter için scaler'ı al
    if 'feature_scalers' in globals() and criterion in globals()['feature_scalers']:
        feature_scaler = globals()['feature_scalers'][criterion]
        text_features_scaled = feature_scaler.transform(text_features)
    else:
        # Eğer scaler yoksa, yeni bir scaler oluştur (uyarı ver)
        print(f"⚠ Uyarı: {criterion} için scaler bulunamadı, yeni scaler oluşturuluyor.")
        scaler = StandardScaler()
        text_features_scaled = scaler.fit_transform(text_features)
    
    # Embedding ve metin özelliklerini birleştir
    combined_features = np.hstack([essay_embedding, text_features_scaled])
    
    # Model ile tahmin yap
    model = trained_models[criterion]
    predicted_score = model.predict(combined_features)[0]
    
    return predicted_score

def test_model_on_sample_essays(n_samples=5):
    """
    Test setindeki örnek essay'ler üzerinde model performansını gösterir.
    
    Args:
        n_samples (int): Gösterilecek örnek sayısı
    """
    if 'model_results' not in globals() or len(model_results) == 0:
        print("Hata: Model sonuçları bulunamadı!")
        return
    
    print("\n" + "="*70)
    print("MODEL TEST: Örnek Essay'ler Üzerinde Tahminler")
    print("="*70)
    
    for criterion in CRITERIA:
        if criterion in model_results:
            result = model_results[criterion]
            y_test = result['y_test']
            y_pred = result['y_pred']
            
            print(f"\n{'='*70}")
            print(f"KRİTER: {criterion}")
            print(f"{'='*70}")
            
            # Test setinden örnekler göster
            for i in range(min(n_samples, len(y_test))):
                print(f"\n  Örnek {i+1}:")
                print(f"    Gerçek Skor: {y_test[i]:.4f}")
                print(f"    Tahmin:      {y_pred[i]:.4f}")
                print(f"    Hata:        {abs(y_test[i] - y_pred[i]):.4f}")
            
            # Genel istatistikler
            errors = np.abs(y_test - y_pred)
            print(f"\n  Genel İstatistikler:")
            print(f"    Ortalama Hata: {errors.mean():.4f}")
            print(f"    Maksimum Hata: {errors.max():.4f}")
            print(f"    Minimum Hata: {errors.min():.4f}")
            print(f"    Standart Sapma: {errors.std():.4f}")

def test_model_all_criteria():
    """
    Tüm kriterler için model performansını karşılaştırmalı gösterir.
    """
    if 'model_results' not in globals() or len(model_results) == 0:
        print("Hata: Model sonuçları bulunamadı!")
        return
    
    print("\n" + "="*70)
    print("MODEL TEST: Tüm Kriterler Karşılaştırması")
    print("="*70)
    
    results_summary = []
    
    for criterion in CRITERIA:
        if criterion in model_results:
            result = model_results[criterion]
            y_test = result['y_test']
            y_pred = result['y_pred']
            
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results_summary.append({
                'Criterion': criterion,
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2,
                'Samples': result['n_samples']
            })
    
    # DataFrame olarak göster
    if results_summary:
        summary_df = pd.DataFrame(results_summary)
        print("\n" + summary_df.to_string(index=False))
        
        # En iyi ve en kötü performans
        best_r2 = summary_df.loc[summary_df['R²'].idxmax()]
        worst_r2 = summary_df.loc[summary_df['R²'].idxmin()]
        
        print(f"\n  En İyi Performans (R²): {best_r2['Criterion']} (R²={best_r2['R²']:.4f})")
        print(f"  En Kötü Performans (R²): {worst_r2['Criterion']} (R²={worst_r2['R²']:.4f})")

def predict_multiple_essays(essay_texts, criterion='TITLE'):
    """
    Birden fazla essay için toplu RA skor tahmini yapar.
    
    Args:
        essay_texts (list): Essay metinleri listesi
        criterion (str): Kriter adı
    
    Returns:
        numpy.ndarray: Tahmin edilen RA skorları
    """
    if 'trained_models' not in globals() or criterion not in trained_models:
        print(f"Hata: {criterion} kriteri için model bulunamadı!")
        return None
    
    # SBERT modelini yükle
    sbert_model = SentenceTransformer('all-mpnet-base-v2')
    
    # Essay embedding'lerini oluştur
    essay_embeddings = sbert_model.encode(essay_texts, convert_to_numpy=True)
    
    # Altın Vektör benzerliğini hesapla (eğitim sırasında saklanan golden_vector'ü kullan)
    golden_similarities = np.zeros(len(essay_texts))  # Varsayılan
    if 'golden_vectors' in globals() and criterion in globals()['golden_vectors']:
        golden_vector = globals()['golden_vectors'][criterion]
        golden_similarities = cosine_similarity(essay_embeddings, [golden_vector]).flatten()
    else:
        print(f"⚠ Uyarı: {criterion} için golden_vector bulunamadı, 0.0 kullanılıyor.")
    
    # Metin özelliklerini çıkar (tutarlılık özellikleri ve golden_vector_similarity dahil)
    text_features = extract_text_features(essay_texts, sbert_model=sbert_model, golden_vector_similarity=golden_similarities)
    
    # Kriter için scaler'ı al
    if 'feature_scalers' in globals() and criterion in globals()['feature_scalers']:
        feature_scaler = globals()['feature_scalers'][criterion]
        text_features_scaled = feature_scaler.transform(text_features)
    else:
        print(f"⚠ Uyarı: {criterion} için scaler bulunamadı, yeni scaler oluşturuluyor.")
        scaler = StandardScaler()
        text_features_scaled = scaler.fit_transform(text_features)
    
    # Embedding ve metin özelliklerini birleştir
    combined_features = np.hstack([essay_embeddings, text_features_scaled])
    
    # Model ile tahmin yap
    model = trained_models[criterion]
    predicted_scores = model.predict(combined_features)
    
    return predicted_scores

def predict_all_criteria(essay_text):
    """
    Bir essay için tüm kriterlerde RA skor tahmini yapar.
    
    Args:
        essay_text (str): Essay metni
    
    Returns:
        dict: Tüm kriterler için tahmin edilen RA skorları
    """
    if 'trained_models' not in globals():
        print("Hata: Eğitilmiş modeller bulunamadı!")
        return None
    
    # SBERT modelini yükle
    sbert_model = SentenceTransformer('all-mpnet-base-v2')
    
    # Essay embedding'ini oluştur
    essay_embedding = sbert_model.encode([essay_text], convert_to_numpy=True)
    
    # Her kriter için metin özelliklerini çıkar (her kriter için farklı golden_vector kullanılabilir)
    # İlk kriter için golden_vector'ü al (veya varsayılan olarak 0.0)
    golden_similarity = np.array([0.0])  # Varsayılan
    if 'golden_vectors' in globals() and len(globals()['golden_vectors']) > 0:
        # İlk bulunan golden_vector'ü kullan (tüm kriterler için aynı olabilir)
        golden_vector = list(globals()['golden_vectors'].values())[0]
        golden_similarity = cosine_similarity(essay_embedding, [golden_vector]).flatten()
    
    # Metin özelliklerini çıkar (tutarlılık özellikleri ve golden_vector_similarity dahil)
    text_features = extract_text_features([essay_text], sbert_model=sbert_model, golden_vector_similarity=golden_similarity)
    
    # Her kriter için aynı scaler kullanılabilir (ilk bulunan)
    if 'feature_scalers' in globals() and len(globals()['feature_scalers']) > 0:
        feature_scaler = list(globals()['feature_scalers'].values())[0]
        text_features_scaled = feature_scaler.transform(text_features)
    else:
        print(f"⚠ Uyarı: Scaler bulunamadı, yeni scaler oluşturuluyor.")
        scaler = StandardScaler()
        text_features_scaled = scaler.fit_transform(text_features)
    
    # Embedding ve metin özelliklerini birleştir
    combined_features = np.hstack([essay_embedding, text_features_scaled])
    
    predictions = {}
    for criterion in CRITERIA:
        if criterion in trained_models:
            model = trained_models[criterion]
            predicted_score = model.predict(combined_features)[0]
            predictions[criterion] = predicted_score
    
    return predictions

# Model testlerini çalıştır (eğer modeller eğitildiyse)
if 'trained_models' in globals() and len(trained_models) > 0:
    print("\n" + "="*70)
    print("MODEL TEST FONKSİYONLARI HAZIR")
    print("="*70)
    print("\nKullanılabilir test fonksiyonları:")
    print("  1. test_model_on_sample_essays(n_samples=5) - Test setindeki örnekleri göster")
    print("  2. test_model_all_criteria() - Tüm kriterler için karşılaştırma")
    print("  3. predict_essay_score(essay_text, criterion='TITLE') - Tek essay için tahmin")
    print("  4. predict_multiple_essays(essay_texts, criterion='TITLE') - Çoklu essay tahmini")
    print("  5. predict_all_criteria(essay_text) - Bir essay için tüm kriterlerde tahmin")
    print("\nÖrnek kullanım:")
    print("  # Test setindeki örnekleri göster")
    print("  test_model_on_sample_essays(n_samples=10)")
    print("\n  # Yeni bir essay için tahmin")
    print("  yeni_essay = 'Bu bir test essay metnidir...'")
    print("  skor = predict_essay_score(yeni_essay, criterion='TITLE')")
    print("  print(f'Tahmin edilen TITLE skoru: {skor:.2f}')")
    print("\n  # Tüm kriterler için karşılaştırma")
    print("  test_model_all_criteria()")
    print("\n  # Bir essay için tüm kriterlerde tahmin")
    print("  tahminler = predict_all_criteria(yeni_essay)")
    print("  for kriter, skor in tahminler.items():")
    print("      print(f'{kriter}: {skor:.2f}')")
    
    # Otomatik test çalıştır
    print("\n" + "="*70)
    print("OTOMATİK TEST ÇALIŞTIRILIYOR...")
    print("="*70)
    
    test_model_on_sample_essays(n_samples=5)
    test_model_all_criteria()
    
else:
    print("\n⚠ Model test fonksiyonları hazır, ancak henüz modeller eğitilmedi.")
    print("   Modelleri eğitmek için kodun üst kısmındaki model eğitimi bölümünün çalıştığından emin olun.")
    if 'trained_models' in globals():
        print(f"   Şu anda eğitilmiş model sayısı: {len(trained_models)}")
    else:
        print(f"   trained_models değişkeni tanımlı değil - model eğitimi hiç çalışmadı.")

# ============================================================================
# Model Performans Görselleştirme Fonksiyonları
# ============================================================================

def visualize_predictions(criterion='TITLE', save_path=None):
    """
    Gerçek vs Tahmin edilen skorları scatter plot olarak görselleştirir.
    
    Args:
        criterion (str): Kriter adı
        save_path (str): Kaydedilecek dosya yolu (None ise gösterir)
    """
    if 'model_results' not in globals() or criterion not in model_results:
        print(f"Hata: {criterion} kriteri için model sonuçları bulunamadı!")
        return
    
    result = model_results[criterion]
    y_test = result['y_test']
    y_pred = result['y_pred']
    
    # Metrikleri hesapla
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Mükemmel Tahmin')
    
    plt.xlabel('Gerçek Skorlar', fontsize=12, fontweight='bold')
    plt.ylabel('Tahmin Edilen Skorlar', fontsize=12, fontweight='bold')
    plt.title(f'{criterion} - Gerçek vs Tahmin Edilen Skorlar\nRMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Görsel kaydedildi: {save_path}")
    else:
        plt.show()
    plt.close()

def visualize_residuals(criterion='TITLE', save_path=None):
    """
    Residual (hata) dağılımını görselleştirir.
    
    Args:
        criterion (str): Kriter adı
        save_path (str): Kaydedilecek dosya yolu (None ise gösterir)
    """
    if 'model_results' not in globals() or criterion not in model_results:
        print(f"Hata: {criterion} kriteri için model sonuçları bulunamadı!")
        return
    
    result = model_results[criterion]
    y_test = result['y_test']
    y_pred = result['y_pred']
    
    residuals = y_test - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Residual scatter plot
    axes[0].scatter(y_pred, residuals, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0].set_xlabel('Tahmin Edilen Skorlar', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Residual (Gerçek - Tahmin)', fontsize=11, fontweight='bold')
    axes[0].set_title(f'{criterion} - Residual Plot', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Residual histogram
    axes[1].hist(residuals, bins=20, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Residual (Gerçek - Tahmin)', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Frekans', fontsize=11, fontweight='bold')
    axes[1].set_title(f'{criterion} - Residual Dağılımı', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Görsel kaydedildi: {save_path}")
    else:
        plt.show()
    plt.close()

def visualize_all_criteria_performance(save_path=None):
    """
    Tüm kriterler için performans metriklerini görselleştirir.
    
    Args:
        save_path (str): Kaydedilecek dosya yolu (None ise gösterir)
    """
    if 'model_results' not in globals() or len(model_results) == 0:
        print("Hata: Model sonuçları bulunamadı!")
        return
    
    # Metrikleri topla
    criteria_list = []
    rmse_list = []
    mae_list = []
    r2_list = []
    
    for criterion in CRITERIA:
        if criterion in model_results:
            result = model_results[criterion]
            y_test = result['y_test']
            y_pred = result['y_pred']
            
            criteria_list.append(criterion)
            rmse_list.append(np.sqrt(mean_squared_error(y_test, y_pred)))
            mae_list.append(mean_absolute_error(y_test, y_pred))
            r2_list.append(r2_score(y_test, y_pred))
    
    if len(criteria_list) == 0:
        print("Görselleştirilecek veri bulunamadı!")
        return
    
    # Subplot oluştur
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    x_pos = np.arange(len(criteria_list))
    
    # RMSE
    axes[0].bar(x_pos, rmse_list, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Kriterler', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('RMSE', fontsize=11, fontweight='bold')
    axes[0].set_title('RMSE (Root Mean Squared Error)', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(criteria_list, rotation=45, ha='right')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # MAE
    axes[1].bar(x_pos, mae_list, color='lightcoral', edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Kriterler', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('MAE', fontsize=11, fontweight='bold')
    axes[1].set_title('MAE (Mean Absolute Error)', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(criteria_list, rotation=45, ha='right')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # R²
    axes[2].bar(x_pos, r2_list, color='lightgreen', edgecolor='black', alpha=0.7)
    axes[2].set_xlabel('Kriterler', fontsize=11, fontweight='bold')
    axes[2].set_ylabel('R² Score', fontsize=11, fontweight='bold')
    axes[2].set_title('R² Score (Coefficient of Determination)', fontsize=12, fontweight='bold')
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(criteria_list, rotation=45, ha='right')
    axes[2].grid(True, alpha=0.3, axis='y')
    axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Görsel kaydedildi: {save_path}")
    else:
        plt.show()
    plt.close()

def visualize_prediction_comparison(criterion='TITLE', n_samples=20, save_path=None):
    """
    Gerçek ve tahmin edilen skorları yan yana karşılaştırır.
    
    Args:
        criterion (str): Kriter adı
        n_samples (int): Gösterilecek örnek sayısı
        save_path (str): Kaydedilecek dosya yolu (None ise gösterir)
    """
    if 'model_results' not in globals() or criterion not in model_results:
        print(f"Hata: {criterion} kriteri için model sonuçları bulunamadı!")
        return
    
    result = model_results[criterion]
    y_test = result['y_test']
    y_pred = result['y_pred']
    
    # İlk n_samples örneği al
    n_display = min(n_samples, len(y_test))
    indices = np.arange(n_display)
    
    x = np.arange(n_display)
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    bars1 = ax.bar(x - width/2, y_test[:n_display], width, label='Gerçek Skorlar', 
                   color='steelblue', alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, y_pred[:n_display], width, label='Tahmin Edilen Skorlar', 
                   color='coral', alpha=0.7, edgecolor='black')
    
    ax.set_xlabel('Örnek İndeksi', fontsize=11, fontweight='bold')
    ax.set_ylabel('Skor', fontsize=11, fontweight='bold')
    ax.set_title(f'{criterion} - Gerçek vs Tahmin Edilen Skorlar Karşılaştırması (İlk {n_display} Örnek)', 
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(indices)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Görsel kaydedildi: {save_path}")
    else:
        plt.show()
    plt.close()

def create_performance_dashboard(save_dir='visualizations'):
    """
    Tüm kriterler için performans dashboard'u oluşturur.
    
    Args:
        save_dir (str): Görsellerin kaydedileceği klasör
    """
    if 'model_results' not in globals() or len(model_results) == 0:
        print("Hata: Model sonuçları bulunamadı!")
        return
    
    # Klasör oluştur
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("PERFORMANS GÖRSELLEŞTİRMELERİ OLUŞTURULUYOR...")
    print(f"{'='*70}")
    
    # Tüm kriterler için genel performans
    print("\n1. Tüm kriterler için performans metrikleri...")
    visualize_all_criteria_performance(save_path=os.path.join(save_dir, 'all_criteria_performance.png'))
    
    # Her kriter için detaylı görselleştirmeler
    for criterion in CRITERIA:
        if criterion in model_results:
            print(f"\n{criterion} kriteri için görselleştirmeler oluşturuluyor...")
            
            # Gerçek vs Tahmin
            visualize_predictions(criterion, save_path=os.path.join(save_dir, f'{criterion}_predictions.png'))
            
            # Residual plot
            visualize_residuals(criterion, save_path=os.path.join(save_dir, f'{criterion}_residuals.png'))
            
            # Karşılaştırma
            visualize_prediction_comparison(criterion, n_samples=20, 
                                           save_path=os.path.join(save_dir, f'{criterion}_comparison.png'))
    
    print(f"\n{'='*70}")
    print(f"TÜM GÖRSELLEŞTİRMELER OLUŞTURULDU: {save_dir} klasörü")
    print(f"{'='*70}")

# Görselleştirme fonksiyonlarını çalıştır (eğer modeller eğitildiyse)
if 'trained_models' in globals() and len(trained_models) > 0:
    print("\n" + "="*70)
    print("GÖRSELLEŞTİRME FONKSİYONLARI HAZIR")
    print("="*70)
    print("\nKullanılabilir görselleştirme fonksiyonları:")
    print("  1. visualize_predictions(criterion='TITLE') - Gerçek vs Tahmin scatter plot")
    print("  2. visualize_residuals(criterion='TITLE') - Residual plot")
    print("  3. visualize_all_criteria_performance() - Tüm kriterler için performans")
    print("  4. visualize_prediction_comparison(criterion='TITLE', n_samples=20) - Karşılaştırma")
    print("  5. create_performance_dashboard(save_dir='visualizations') - Tüm görselleri oluştur")
    print("\nÖrnek kullanım:")
    print("  # Tek bir kriter için görselleştirme")
    print("  visualize_predictions('TITLE')")
    print("  visualize_residuals('GRAMMAR')")
    print("\n  # Tüm kriterler için performans")
    print("  visualize_all_criteria_performance()")
    print("\n  # Tüm görselleri oluştur ve kaydet")
    print("  create_performance_dashboard('visualizations')")
    
    # Otomatik görselleştirme oluştur
    print("\n" + "="*70)
    print("OTOMATİK GÖRSELLEŞTİRME OLUŞTURULUYOR...")
    print("="*70)
    
    # Tüm kriterler için genel performans
    visualize_all_criteria_performance()
    
    # Her kriter için görselleştirme (ilk 3 kriter)
    for criterion in list(CRITERIA)[:3]:
        if criterion in model_results:
            visualize_predictions(criterion)
            visualize_residuals(criterion)
    
else:
    print("\n⚠ Görselleştirme fonksiyonları hazır, ancak henüz modeller eğitilmedi.")

