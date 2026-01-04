# -*- coding: utf-8 -*-
"""
Streamlit Web Arayüzü - Essay Puanlama Sistemi
Kaydedilmiş modeller ile essay puanlaması yapar.
"""

import streamlit as st
import os
import sys
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle

# Sayfa yapılandırması
st.set_page_config(
    page_title="Essay Puanlama Sistemi",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Başlık
st.title("📝 Essay Puanlama Sistemi")
st.markdown("---")

# Sidebar - Model yükleme
st.sidebar.header("⚙️ Ayarlar")

models_dir = st.sidebar.text_input("Modeller Dizini", value="saved_models")
load_models = st.sidebar.button("Modelleri Yükle", type="primary")

# Kriterler
CRITERIA = ['TITLE', 'THESIS', 'ORGANISATION', 'SUPPORT', 'ANALYSIS', 'SENTENCE', 'GRAMMAR']

# Global değişkenler
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'loaded_models' not in st.session_state:
    st.session_state.loaded_models = {}
if 'sbert_model' not in st.session_state:
    st.session_state.sbert_model = None

# Model yükleme fonksiyonu (cache'lenmiş)
@st.cache_resource
def load_models_from_dir(models_dir):
    """Modelleri dizinden yükler (cache'lenmiş)"""
    loaded = {}
    errors = []
    
    for criterion in CRITERIA:
        model_path = os.path.join(models_dir, f'model_{criterion}.pkl')
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    loaded[criterion] = pickle.load(f)
            except Exception as e:
                errors.append(f"{criterion}: {str(e)}")
        else:
            errors.append(f"{criterion}: Dosya bulunamadı")
    
    return loaded, errors

# SBERT model yükleme (cache'lenmiş)
@st.cache_resource
def load_sbert_model():
    """SBERT modelini yükler (cache'lenmiş)"""
    try:
        return SentenceTransformer('all-mpnet-base-v2')
    except Exception as e:
        st.error(f"SBERT modeli yüklenemedi: {str(e)}")
        return None

# Model yükleme (otomatik veya buton ile)
if load_models or not st.session_state.models_loaded:
    with st.sidebar:
        with st.spinner("Modeller yükleniyor..."):
            try:
                # Modelleri yükle
                loaded, errors = load_models_from_dir(models_dir)
                
                if loaded:
                    st.session_state.loaded_models = loaded
                    st.session_state.models_loaded = True
                    
                    # SBERT modelini yükle
                    if st.session_state.sbert_model is None:
                        st.session_state.sbert_model = load_sbert_model()
                    
                    if st.session_state.sbert_model is not None:
                        st.sidebar.success(f"✓ {len(loaded)} model yüklendi")
                    else:
                        st.sidebar.warning("Modeller yüklendi ancak SBERT modeli yüklenemedi")
                else:
                    st.sidebar.error("Hiçbir model yüklenemedi!")
                    for error in errors:
                        st.sidebar.error(error)
            except Exception as e:
                st.sidebar.error(f"Model yükleme hatası: {str(e)}")
                st.session_state.models_loaded = False

# Ana içerik
if st.session_state.models_loaded and st.session_state.loaded_models:
    # Metin girişi
    st.header("📝 Essay Metni Girişi")
    
    essay_text = st.text_area(
        "Essay metnini buraya yazın:",
        height=200,
        placeholder="Essay metninizi buraya yapıştırın..."
    )
    
    # Puanlama butonu
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        score_button = st.button("🎯 Puanla", type="primary", use_container_width=True)
    
    with col2:
        clear_button = st.button("🗑️ Temizle", use_container_width=True)
    
    if clear_button:
        essay_text = ""
        st.rerun()
    
    # Puanlama fonksiyonu
    def extract_text_features_simple(text, sbert_model, golden_vector_similarity=None):
        """Basit metin özellikleri çıkarır (sbert.py'deki extract_text_features ile uyumlu)"""
        import re
        
        if not text or len(text.strip()) == 0:
            text = ""
        
        # Temel özellikler
        char_count = len(text)
        words = re.findall(r'\b\w+\b', text.lower())
        word_count = len(words) if words else 0
        sentence_count = len([s for s in re.split(r'[.!?]+', text) if s.strip()]) if text else 0
        avg_word_length = np.mean([len(word) for word in words]) if word_count > 0 else 0
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        punctuation_count = sum(1 for char in text if char in '.,!?;:')
        uppercase_ratio = sum(1 for char in text if char.isupper()) / char_count if char_count > 0 else 0
        digit_count = sum(1 for char in text if char.isdigit())
        special_char_count = sum(1 for char in text if not char.isalnum() and char not in ' .,!?;:')
        unique_words = len(set(words))
        lexical_diversity = unique_words / word_count if word_count > 0 else 0
        
        # Temel özellikler listesi
        base_features = [
            char_count, word_count, sentence_count, avg_word_length, avg_sentence_length,
            punctuation_count, uppercase_ratio, digit_count, special_char_count,
            lexical_diversity
        ]
        
        # Tutarlılık özellikleri (basitleştirilmiş)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        avg_coherence = 0.0
        min_coherence = 0.0
        if len(sentences) >= 2 and sbert_model is not None:
            try:
                embeddings = sbert_model.encode(sentences, convert_to_numpy=True, show_progress_bar=False)
                similarities = []
                for i in range(len(embeddings) - 1):
                    sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
                    similarities.append(sim)
                if similarities:
                    avg_coherence = np.mean(similarities)
                    min_coherence = np.min(similarities)
            except:
                pass
        
        base_features.extend([avg_coherence, min_coherence])
        
        # Altın Vektör benzerliği
        if golden_vector_similarity is not None:
            base_features.append(golden_vector_similarity)
        else:
            base_features.append(0.0)
        
        return np.array(base_features).reshape(1, -1)
    
    def predict_score(essay_text, criterion):
        """Kaydedilmiş model ile puan tahmini yapar"""
        if criterion not in st.session_state.loaded_models:
            return None
        
        if st.session_state.sbert_model is None:
            st.error("SBERT modeli yüklenmemiş!")
            return None
        
        model_data = st.session_state.loaded_models[criterion]
        sbert_model = st.session_state.sbert_model
        
        try:
            # Essay embedding'ini oluştur
            essay_embedding = sbert_model.encode([essay_text], convert_to_numpy=True, show_progress_bar=False)
            
            # Altın Vektör benzerliğini hesapla
            golden_vector = model_data.get('golden_vector')
            golden_similarity = None
            if golden_vector is not None:
                golden_similarity = cosine_similarity(essay_embedding, [golden_vector]).flatten()[0]
            
            # Metin özelliklerini çıkar
            text_features = extract_text_features_simple(
                essay_text, 
                sbert_model, 
                golden_vector_similarity=golden_similarity
            )
            
            # Text features'ı scale et
            feature_scaler = model_data.get('feature_scaler')
            if feature_scaler is not None:
                text_features_scaled = feature_scaler.transform(text_features)
            else:
                text_features_scaled = text_features
            
            # Embedding ve text features'ı birleştir
            combined_features = np.hstack([essay_embedding, text_features_scaled])
            
            # PCA uygula (varsa)
            pca_model = model_data.get('pca_model')
            if pca_model is not None:
                # Hibrit yapı: PCA sadece embedding'lere
                embedding_dim = len(essay_embedding[0])
                X_emb = combined_features[:, :embedding_dim]
                X_text = combined_features[:, embedding_dim:]
                
                X_emb_pca = pca_model.transform(X_emb)
                combined_features = np.hstack([X_emb_pca, X_text])
            
            # Tahmin yap
            model = model_data['model']
            prediction = model.predict(combined_features)[0]
            
            return float(prediction)
        except Exception as e:
            st.error(f"Hata ({criterion}): {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            return None
    
    # Puanlama
    if score_button and essay_text.strip():
        with st.spinner("Puanlama yapılıyor..."):
            results = {}
            
            # Her kriter için puanlama
            for criterion in CRITERIA:
                if criterion in st.session_state.loaded_models:
                    score = predict_score(essay_text, criterion)
                    if score is not None:
                        results[criterion] = score
            
            # Sonuçları göster
            if results:
                st.header("📊 Puanlama Sonuçları")
                
                # Metrikler
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    avg_score = np.mean(list(results.values()))
                    st.metric("Ortalama Puan", f"{avg_score:.2f}")
                
                with col2:
                    min_score = np.min(list(results.values()))
                    st.metric("En Düşük Puan", f"{min_score:.2f}")
                
                with col3:
                    max_score = np.max(list(results.values()))
                    st.metric("En Yüksek Puan", f"{max_score:.2f}")
                
                with col4:
                    total_score = sum(results.values())
                    st.metric("Toplam Puan", f"{total_score:.2f}")
                
                st.markdown("---")
                
                # Detaylı sonuçlar
                st.subheader("Kriter Bazında Puanlar")
                
                # İki sütunlu görünüm
                cols = st.columns(2)
                
                for idx, (criterion, score) in enumerate(results.items()):
                    with cols[idx % 2]:
                        # Progress bar ile görselleştirme
                        normalized_score = (score / 5.0) * 100  # 5 üzerinden normalizasyon
                        st.write(f"**{criterion}**")
                        st.progress(normalized_score / 100)
                        st.write(f"Puan: **{score:.2f}** / 5.0")
                        st.markdown("---")
                
                # Sonuçları JSON olarak indirme
                import json
                results_json = json.dumps(results, indent=2, ensure_ascii=False)
                st.download_button(
                    label="📥 Sonuçları İndir (JSON)",
                    data=results_json,
                    file_name="essay_scores.json",
                    mime="application/json"
                )
            else:
                st.error("Puanlama yapılamadı. Lütfen modellerin yüklü olduğundan emin olun.")
    
    elif score_button and not essay_text.strip():
        st.warning("⚠️ Lütfen essay metnini girin!")
    
    # Örnek metin
    with st.expander("📌 Örnek Metin"):
        st.code("""
Bu örnek bir essay metnidir. Burada essay'inizin nasıl puanlanacağını görebilirsiniz.
Essay metninizi yukarıdaki alana yapıştırıp "Puanla" butonuna tıklayın.
Tüm kriterler için puanlarınızı göreceksiniz.
        """)
    
    # Model bilgileri
    with st.expander("ℹ️ Model Bilgileri"):
        st.write("**Yüklü Modeller:**")
        for criterion in st.session_state.loaded_models.keys():
            model_data = st.session_state.loaded_models[criterion]
            saved_date = model_data.get('saved_date', 'Bilinmiyor')
            st.write(f"- **{criterion}**: {saved_date}")
        
        st.write(f"\n**Toplam Model Sayısı:** {len(st.session_state.loaded_models)}")
        st.write(f"**SBERT Modeli:** all-mpnet-base-v2")

else:
    # Model yüklenmemişse
    st.info("👈 Lütfen önce sidebar'dan 'Modelleri Yükle' butonuna tıklayın.")
    
    st.markdown("""
    ### Kullanım Talimatları:
    1. **Modelleri Yükle**: Sol taraftaki sidebar'dan "Modelleri Yükle" butonuna tıklayın
    2. **Essay Metni Gir**: Ana alana essay metninizi yazın veya yapıştırın
    3. **Puanla**: "Puanla" butonuna tıklayın
    4. **Sonuçları Gör**: Tüm kriterler için puanlarınızı görün
    
    ### Not:
    Modellerin kaydedilmiş olması gerekiyor. Eğer modeller kaydedilmemişse, önce `sbert.py` dosyasını çalıştırarak modelleri eğitin ve kaydedin.
    """)

