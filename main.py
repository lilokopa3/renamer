import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from gensim.models import Word2Vec

import pickle
import nltk
from nltk.tokenize import word_tokenize
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt

@st.cache_resource
def load_models(): 
    device = torch.device('cpu')

    # Load IndoBERT
    indobert_tokenizer = AutoTokenizer.from_pretrained("indolem/indobert-base-uncased", do_lower_case=True)
    indobert_model = AutoModelForSequenceClassification.from_pretrained('indolem/indobert-base-uncased', num_labels=3).to(device)
    indobert_model.load_state_dict(torch.load('best_indobert_model1.pt', map_location=device))
    indobert_model.eval()  # Set model ke mode evaluasi

    # Load IndoRoBERTa
    indoroberta_tokenizer = AutoTokenizer.from_pretrained("LazarusNLP/simcse-indoroberta-base")
    indoroberta_model = AutoModelForSequenceClassification.from_pretrained("LazarusNLP/simcse-indoroberta-base", num_labels=3).to(device)
    indoroberta_model.load_state_dict(torch.load('best_model_fold4.pt', map_location=device))
    indoroberta_model.eval()  # Set model ke mode evaluasi

    # Load SVM model
    with open('.\svm_model.sav', 'rb') as f:
        svm_classifier = pickle.load(f)

    # Initialize Word2Vec (dummy initialization)
    word2vec_model = Word2Vec.load(".\w2vmodel.sav")

    return {
        'indobert': (indobert_tokenizer, indobert_model),
        'indoroberta': (indoroberta_tokenizer, indoroberta_model),
        'word2vec': word2vec_model,
        'svm': svm_classifier
    }

def preprocess_text(text):
    # Lowercase and clean text
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    return tokens

def get_word2vec_features(tokens, word2vec_model):  
    vectors = []
    for token in tokens: 
        if token in word2vec_model.wv: 
            vectors.append(word2vec_model.wv[token])

    return np.mean(vectors, axis=0) if vectors else np.zeros(word2vec_model.vector_size)

def analyze_sentiment_bert(text, tokenizer, model): 
    inputs = tokenizer(text, return_tensors="pt", padding='max_length', truncation=True, max_length=512)
 
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    outputs = model(input_ids, attention_mask=attention_mask)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return predictions.detach().numpy()[0]

def create_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

def analyze_sentiment(text, models):
    tokens = preprocess_text(text)
    preprocessed_text = ' '.join(tokens)

    # Predictions from IndoBERT and IndoRoBERTa
    indobert_pred = analyze_sentiment_bert(preprocessed_text, models['indobert'][0], models['indobert'][1])
    indoroberta_pred = analyze_sentiment_bert(preprocessed_text, models['indoroberta'][0], models['indoroberta'][1])

    # Word2Vec + SVM prediction
    word2vec_features = get_word2vec_features(tokens, models['word2vec'])
    svm_pred = models['svm'].predict_proba([word2vec_features])[0]

    return {
        'IndoBERT': indobert_pred,
        'IndoRoBERTa': indoroberta_pred,
        'SVM': svm_pred
    }

def main(): 
    # Download necessary NLTK data
    nltk.download('punkt_tab')

    # Set page config
    st.set_page_config(
        page_title="Analisis Sentimen Teks Berita",
        layout="wide"
    )

    # Custom CSS
    st.markdown("""
        <style>
        .main {
            padding: 20px;
        }
        .stButton>button {
            width: 100%;
            background-color: #4CAF50;
            color: white;
            padding: 10px;
            border-radius: 20px;
        }
        .footer {
            position: relative;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: transparent;
            padding: 20px;
            text-align: center;
            margin-top: 50px;
            animation: colorChange 5s infinite, marquee 20s linear infinite;
            font-weight: bold;
        }
        @keyframes colorChange {
            0% {color: red;}
            25% {color: blue;}
            50% {color: green;}
            75% {color: orange;}
            100% {color: red;}
        }
        @keyframes marquee {
            0% {transform: translateX(100%);}
            100% {transform: translateX(-100%);}
        }
        </style>
    """, unsafe_allow_html=True)

    # Title
    st.markdown("<h1 style='text-align: center;'>Analisis Sentimen Teks Berita</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Menggunakan Model IndoBERT, IndoRoBERTa, dan SVM</h3>", unsafe_allow_html=True)

    # Load models
    with st.spinner('Memuat model...'):
        models = load_models()

    # Input text area
    text_input = st.text_area("Masukkan Teks Berita", height=200)

    # Analyze button
    if st.button("Analisis Sentimen"):
        if text_input:
            with st.spinner("Menganalisis sentimen..."):
                results = analyze_sentiment(text_input, models)
                sentiments = ['Negatif', 'Netral', 'Positif']

                # Display results
                st.subheader("Hasil Analisis Sentimen")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.text("IndoBERT")
                    st.write(f"Prediksi: {sentiments[np.argmax(results['IndoBERT'])]}")
                    st.write(f"Confidence: {np.max(results['IndoBERT']):.2f}")
                with col2:
                    st.text("IndoRoBERTa")
                    st.write(f"Prediksi: {sentiments[np.argmax(results['IndoRoBERTa'])]}")
                    st.write(f"Confidence: {np.max(results['IndoRoBERTa']):.2f}")
                with col3:
                    st.text("SVM + Word2Vec")
                    st.write(f"Prediksi: {sentiments[np.argmax(results['SVM'])]}")
                    st.write(f"Confidence: {np.max(results['SVM']):.2f}")

                # Visualization
                st.subheader("Visualisasi Data")
                data_bar = pd.DataFrame({
                    'Model': ['IndoBERT', 'IndoRoBERTa', 'SVM'],
                    'Confidence': [
                        np.max(results['IndoBERT']),
                        np.max(results['IndoRoBERTa']),
                        np.max(results['SVM'])
                    ]
                })
                fig_bar = px.bar(data_bar, x='Model', y='Confidence', title='Confidence Score per Model')
                st.plotly_chart(fig_bar, use_container_width=True)

                # Word Cloud
                st.subheader("Word Cloud")
                st.pyplot(create_wordcloud(text_input))

    # Footer
    st.markdown("""
        <div class='footer'>
            <p>\u00a9 2024 Analisis Sentimen Teks Berita - Powered by IndoBERT, IndoRoBERTa, and SVM | @nsyw.rhdtl</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main() 