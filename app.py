# -*- coding: utf-8 -*-
"""
Streamlit Web Interface - Essay Scoring System
Scores essays using saved models.
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

# Page configuration
st.set_page_config(
    page_title="Essay Scoring System",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("📝 Essay Scoring System")
st.markdown("---")

# Sidebar - Model loading
st.sidebar.header("⚙️ Settings")

models_dir = st.sidebar.text_input("Models Directory", value="saved_models")
load_models = st.sidebar.button("Load Models", type="primary")

# Kriterler
CRITERIA = ['TITLE', 'THESIS', 'ORGANISATION', 'SUPPORT', 'ANALYSIS', 'SENTENCE', 'GRAMMAR']

# Global değişkenler
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'loaded_models' not in st.session_state:
    st.session_state.loaded_models = {}
if 'sbert_model' not in st.session_state:
    st.session_state.sbert_model = None

# Model loading function (cached)
@st.cache_resource
def load_models_from_dir(models_dir):
    """Loads models from directory (cached)"""
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
            errors.append(f"{criterion}: File not found")
    
    return loaded, errors

# SBERT model loading (cached)
@st.cache_resource
def load_sbert_model():
    """Loads SBERT model (cached)"""
    try:
        return SentenceTransformer('all-mpnet-base-v2')
    except Exception as e:
        st.error(f"Failed to load SBERT model: {str(e)}")
        return None

# Model loading (automatic or via button)
if load_models or not st.session_state.models_loaded:
    with st.sidebar:
        with st.spinner("Loading models..."):
            try:
                # Load models
                loaded, errors = load_models_from_dir(models_dir)
                
                if loaded:
                    st.session_state.loaded_models = loaded
                    st.session_state.models_loaded = True
                    
                    # Load SBERT model
                    if st.session_state.sbert_model is None:
                        st.session_state.sbert_model = load_sbert_model()
                    
                    if st.session_state.sbert_model is not None:
                        st.sidebar.success(f"✓ {len(loaded)} models loaded")
                    else:
                        st.sidebar.warning("Models loaded but SBERT model failed to load")
                else:
                    st.sidebar.error("No models could be loaded!")
                    for error in errors:
                        st.sidebar.error(error)
            except Exception as e:
                st.sidebar.error(f"Model loading error: {str(e)}")
                st.session_state.models_loaded = False

# Main content
if st.session_state.models_loaded and st.session_state.loaded_models:
    # Text input
    st.header("📝 Essay Text Input")
    
    essay_text = st.text_area(
        "Enter your essay text here:",
        height=200,
        placeholder="Paste your essay text here..."
    )
    
    # Scoring button
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        score_button = st.button("🎯 Score", type="primary", use_container_width=True)
    
    with col2:
        clear_button = st.button("🗑️ Clear", use_container_width=True)
    
    if clear_button:
        essay_text = ""
        st.rerun()
    
    # Scoring function
    def extract_text_features_simple(text, sbert_model, golden_vector_similarity=None):
        """Extracts simple text features (compatible with extract_text_features in sbert.py)"""
        import re
        
        if not text or len(text.strip()) == 0:
            text = ""
        
        # Basic features
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
        
        # Basic features list
        base_features = [
            char_count, word_count, sentence_count, avg_word_length, avg_sentence_length,
            punctuation_count, uppercase_ratio, digit_count, special_char_count,
            lexical_diversity
        ]
        
        # Coherence features (simplified)
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
        
        # Golden Vector similarity
        if golden_vector_similarity is not None:
            base_features.append(golden_vector_similarity)
        else:
            base_features.append(0.0)
        
        return np.array(base_features).reshape(1, -1)
    
    def predict_score(essay_text, criterion):
        """Makes score prediction using saved model"""
        if criterion not in st.session_state.loaded_models:
            return None
        
        if st.session_state.sbert_model is None:
            st.error("SBERT model is not loaded!")
            return None
        
        model_data = st.session_state.loaded_models[criterion]
        sbert_model = st.session_state.sbert_model
        
        try:
            # Create essay embedding
            essay_embedding = sbert_model.encode([essay_text], convert_to_numpy=True, show_progress_bar=False)
            
            # Calculate Golden Vector similarity
            golden_vector = model_data.get('golden_vector')
            golden_similarity = None
            if golden_vector is not None:
                golden_similarity = cosine_similarity(essay_embedding, [golden_vector]).flatten()[0]
            
            # Extract text features
            text_features = extract_text_features_simple(
                essay_text, 
                sbert_model, 
                golden_vector_similarity=golden_similarity
            )
            
            # Scale text features
            feature_scaler = model_data.get('feature_scaler')
            if feature_scaler is not None:
                text_features_scaled = feature_scaler.transform(text_features)
            else:
                text_features_scaled = text_features
            
            # Combine embedding and text features
            combined_features = np.hstack([essay_embedding, text_features_scaled])
            
            # Apply PCA (if available)
            pca_model = model_data.get('pca_model')
            if pca_model is not None:
                # Hybrid structure: PCA only on embeddings
                embedding_dim = len(essay_embedding[0])
                X_emb = combined_features[:, :embedding_dim]
                X_text = combined_features[:, embedding_dim:]
                
                X_emb_pca = pca_model.transform(X_emb)
                combined_features = np.hstack([X_emb_pca, X_text])
            
            # Make prediction
            model = model_data['model']
            prediction = model.predict(combined_features)[0]
            
            return float(prediction)
        except Exception as e:
            st.error(f"Error ({criterion}): {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            return None
    
    # Scoring
    if score_button and essay_text.strip():
        with st.spinner("Scoring in progress..."):
            results = {}
            
            # Score for each criterion
            for criterion in CRITERIA:
                if criterion in st.session_state.loaded_models:
                    score = predict_score(essay_text, criterion)
                    if score is not None:
                        results[criterion] = score
            
            # Show results
            if results:
                st.header("📊 Scoring Results")
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    avg_score = np.mean(list(results.values()))
                    st.metric("Average Score", f"{avg_score:.2f}")
                
                with col2:
                    min_score = np.min(list(results.values()))
                    st.metric("Minimum Score", f"{min_score:.2f}")
                
                with col3:
                    max_score = np.max(list(results.values()))
                    st.metric("Maximum Score", f"{max_score:.2f}")
                
                with col4:
                    total_score = sum(results.values())
                    st.metric("Total Score", f"{total_score:.2f}")
                
                st.markdown("---")
                
                # Detailed results
                st.subheader("Scores by Criterion")
                
                # Two-column view
                cols = st.columns(2)
                
                for idx, (criterion, score) in enumerate(results.items()):
                    with cols[idx % 2]:
                        # Visualization with progress bar
                        normalized_score = (score / 5.0) * 100  # Normalization out of 5
                        st.write(f"**{criterion}**")
                        st.progress(normalized_score / 100)
                        st.write(f"Score: **{score:.2f}** / 5.0")
                        st.markdown("---")
                
                # Download results as JSON
                import json
                results_json = json.dumps(results, indent=2, ensure_ascii=False)
                st.download_button(
                    label="📥 Download Results (JSON)",
                    data=results_json,
                    file_name="essay_scores.json",
                    mime="application/json"
                )
            else:
                st.error("Scoring failed. Please ensure models are loaded.")
    
    elif score_button and not essay_text.strip():
        st.warning("⚠️ Please enter essay text!")
    
    # Example text
    with st.expander("📌 Example Text"):
        st.code("""
This is an example essay text. Here you can see how your essay will be scored.
Paste your essay text in the field above and click the "Score" button.
You will see scores for all criteria.
        """)
    
    # Model information
    with st.expander("ℹ️ Model Information"):
        st.write("**Loaded Models:**")
        for criterion in st.session_state.loaded_models.keys():
            model_data = st.session_state.loaded_models[criterion]
            saved_date = model_data.get('saved_date', 'Unknown')
            st.write(f"- **{criterion}**: {saved_date}")
        
        st.write(f"\n**Total Number of Models:** {len(st.session_state.loaded_models)}")
        st.write(f"**SBERT Model:** all-mpnet-base-v2")

else:
    # If models are not loaded
    st.info("👈 Please click the 'Load Models' button in the sidebar first.")
    
    st.markdown("""
    ### Usage Instructions:
    1. **Load Models**: Click the "Load Models" button in the left sidebar
    2. **Enter Essay Text**: Type or paste your essay text in the main area
    3. **Score**: Click the "Score" button
    4. **View Results**: See your scores for all criteria
    
    ### Note:
    Models must be saved first. If models are not saved, first run the `sbert.py` file to train and save the models.
    """)

