import base64
import io
from flask import Flask, render_template, request, send_file , jsonify
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import TFBertModel, BertTokenizer
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
import tensorflow as tf
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from PIL import Image
import pytesseract
import pandas as pd
from collections import Counter
import re
from sentence_transformers import SentenceTransformer
from scipy.stats import spearmanr, pearsonr

# Load BERT and SBERT models
bert_model = TFBertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Setup Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static'

# Pytesseract configuration for OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# File paths
CSV_FILES = {
    "correlation": "static/correlation.csv",
    "similarities": "static/similarities.csv",
    "word_frequencies": "static/word_frequencies.csv",
}

def Pearson_correlation(text1, text2):
    # Load the pre-trained model from sentence_transformers
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # Get embeddings for both texts
    embeddings1 = model.encode([text1])[0]
    embeddings2 = model.encode([text2])[0]
    # Pearson Correlation
    pearson_corr, _ = pearsonr(embeddings1, embeddings2)

    return pearson_corr

def Root_mean_square_error(text1, text2):
    # Load the pre-trained model from sentence_transformers
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # Get embeddings for both texts
    embeddings1 = model.encode([text1])[0]
    embeddings2 = model.encode([text2])[0]

    # RMS Calculation
    rms = np.sqrt(np.mean((embeddings1 - embeddings2) ** 2))

    return rms


def Spearmans_Rank_correlation(text1, text2):
    # Load the pre-trained model from sentence_transformers
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # Get embeddings for both texts
    embeddings1 = model.encode([text1])[0]
    embeddings2 = model.encode([text2])[0]

    # Spearman's Rank Correlation
    spearman_corr, _ = spearmanr(embeddings1, embeddings2)

    return spearman_corr


# Functions for similarity calculations

def calculate_cosine_similarity(text1, text2):
    """
    Calculate cosine similarity using TF-IDF vectors.
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    return float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])


def calculate_bert_similarity(text1, text2):
    """
    Calculate cosine similarity using BERT embeddings.
    """
    # Tokenize and get BERT embeddings
    inputs1 = tokenizer(text1, return_tensors='tf', truncation=True, padding=True, max_length=512)
    inputs2 = tokenizer(text2, return_tensors='tf', truncation=True, padding=True, max_length=512)

    outputs1 = bert_model(inputs1['input_ids'], attention_mask=inputs1['attention_mask'])
    outputs2 = bert_model(inputs2['input_ids'], attention_mask=inputs2['attention_mask'])

    # Use the [CLS] token embedding for similarity
    cls_embedding1 = outputs1.last_hidden_state[:, 0, :].numpy()
    cls_embedding2 = outputs2.last_hidden_state[:, 0, :].numpy()

    return float(cosine_similarity(cls_embedding1, cls_embedding2)[0][0])


def calculate_sbert_similarity(text1, text2):
    """
    Calculate cosine similarity using Sentence-BERT embeddings.
    """
    embeddings1 = sbert_model.encode([text1])[0]
    embeddings2 = sbert_model.encode([text2])[0]
    return float(cosine_similarity([embeddings1], [embeddings2])[0][0])


def calculate_lstm_similarity(text1, text2):
    """
    Dummy LSTM similarity calculation.
    """
    tokenizer_lstm = tf.keras.preprocessing.text.Tokenizer(num_words=10000)
    tokenizer_lstm.fit_on_texts([text1, text2])

    seq1 = tokenizer_lstm.texts_to_sequences([text1])[0]
    seq2 = tokenizer_lstm.texts_to_sequences([text2])[0]

    padded_seq1 = tf.keras.preprocessing.sequence.pad_sequences([seq1], maxlen=10)
    padded_seq2 = tf.keras.preprocessing.sequence.pad_sequences([seq2], maxlen=10)

    similarity = np.dot(padded_seq1, padded_seq2.T) / (np.linalg.norm(padded_seq1) * np.linalg.norm(padded_seq2))
    return float(similarity)


def extract_text_from_image(image_path):
    """
    Extract text from an image using Pytesseract OCR.
    """
    image = Image.open(image_path)
    return pytesseract.image_to_string(image)


def generate_word_frequencies(text1, text2):
    """
    Generate word frequencies from the combined text.
    """
    combined_text = text1 + ' ' + text2
    # Clean and tokenize the text
    words = re.findall(r'\w+', combined_text.lower())
    # Optionally, remove stopwords
    stopwords = set([
        'the', 'and', 'is', 'in', 'to', 'of', 'a', 'an', 'it', 'this', 'that',
        'with', 'for', 'on', 'at', 'by', 'from', 'as', 'are', 'be', 'or', 'if'
    ])  # Extend this list as needed
    filtered_words = [word for word in words if word not in stopwords]
    # Count word frequencies
    word_counts = Counter(filtered_words)
    return word_counts


def generate_word_frequency_csv(text1, text2):
    """
    Generate a word frequency CSV from the combined text.
    This function calls `generate_word_frequencies` to get word counts.
    """
    # Get word counts by calling the helper function
    word_counts = generate_word_frequencies(text1, text2)

    # Create DataFrame
    df = pd.DataFrame(word_counts.items(), columns=['word', 'frequency'])
    # Sort by frequency descending
    df = df.sort_values(by='frequency', ascending=False)
    # Save to CSV
    word_freq_filename = 'word_frequencies.csv'
    word_freq_filepath = os.path.join(app.config['STATIC_FOLDER'], word_freq_filename)
    df.to_csv(word_freq_filepath, index=False)

    return word_freq_filename

def generate_similarity_csv(cosine_sim, lstm_sim, bert_sim, sbert_sim):
    """
    Generate a CSV for similarity scores in a format that Tableau can visualize.
    """
    similarities = [
        {'Similarity': 'Cosine Similarity', 'Score': cosine_sim},
        {'Similarity': 'LSTM Similarity', 'Score': lstm_sim},
        {'Similarity': 'BERT Similarity', 'Score': bert_sim},
        {'Similarity': 'SBERT Similarity', 'Score': sbert_sim}
    ]

    # Create DataFrame for structured similarity data
    df_similarities = pd.DataFrame(similarities)

    # Save the DataFrame to CSV
    similarities_csv_filename = 'similarities.csv'
    similarities_csv_filepath = os.path.join(app.config['STATIC_FOLDER'], similarities_csv_filename)
    df_similarities.to_csv(similarities_csv_filepath, index=False)

    return similarities_csv_filename

def generate_correlation_csv(sp,pc,rmse):
    """
    Generate a CSV for similarity scores in a format that Tableau can visualize.
    """
    correlation = [
        {'correlation': 'Spearmans Rank correlation', 'Score': sp},
        {'correlation': 'Pearson correlation', 'Score': pc},
        {'correlation': 'Root mean square error', 'Score': rmse},
    ]

    # Create DataFrame for structured similarity data
    df_correlation = pd.DataFrame(correlation)

    # Save the DataFrame to CSV
    correlation_csv_filename = 'correlation.csv'
    correlation_csv_filepath = os.path.join(app.config['STATIC_FOLDER'], correlation_csv_filename)
    df_correlation.to_csv(correlation_csv_filepath, index=False)

    return correlation_csv_filename


# Function to generate graphs
def generate_graph(graph_type, dataset):
    try:
        # Load dataset
        file_path = f"static/{dataset}.csv"
        df = pd.read_csv(file_path)

        # Create figure
        fig, ax = plt.subplots()

        # Handle different chart types
        if graph_type == "line":
            df.plot(ax=ax)
        elif graph_type == "bar":
            df.plot(kind='bar', ax=ax)
        elif graph_type == "pie":
            df.iloc[:, 1].plot(kind='pie', ax=ax, autopct='%1.1f%%')
        elif graph_type == "histogram":
            df.hist(ax=ax)
        elif graph_type == "scatter":
            if len(df.columns) >= 2:
                df.plot.scatter(x=df.columns[0], y=df.columns[1], ax=ax)
        elif graph_type == "bubble":
            if len(df.columns) >= 3:
                df.plot.scatter(x=df.columns[0], y=df.columns[1], s=df[df.columns[2]] * 100, ax=ax, alpha=0.5)

        # Save plot as image
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        encoded_img = base64.b64encode(img.getvalue()).decode('utf-8')
        plt.close()

        return encoded_img
    except Exception as e:
        return str(e)


# Function to generate word cloud
def generate_wordcloud():
    try:
        df = pd.read_csv("static/word_frequencies.csv")
        word_dict = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))

        wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(word_dict)

        img = io.BytesIO()
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.savefig(img, format='png')
        img.seek(0)
        encoded_img = base64.b64encode(img.getvalue()).decode('utf-8')
        plt.close()

        return encoded_img
    except Exception as e:
        return str(e)

@app.route('/visualization')
def visualization():
    return render_template('visualization.html')


@app.route('/get_graph', methods=['POST'])
def get_graph():
    data = request.get_json()
    graph_type = data.get("graphType")
    dataset = data.get("dataset")

    if dataset == "word_frequencies":
        image = generate_wordcloud()  # Word cloud does not need graph type
    else:
        image = generate_graph(graph_type, dataset)

    return jsonify({"image": image})


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text1 = ''
        text2 = ''
        # Handle text or image input for Text 1
        if 'image1' in request.files and request.files['image1'].filename != '':
            image1 = request.files['image1']
            image1_path = os.path.join(app.config['UPLOAD_FOLDER'], image1.filename)
            image1.save(image1_path)
            text1 = extract_text_from_image(image1_path)
        else:
            text1 = request.form.get('text1', '')

        # Handle text or image input for Text 2
        if 'image2' in request.files and request.files['image2'].filename != '':
            image2 = request.files['image2']
            image2_path = os.path.join(app.config['UPLOAD_FOLDER'], image2.filename)
            image2.save(image2_path)
            text2 = extract_text_from_image(image2_path)
        else:
            text2 = request.form.get('text2', '')

        # Calculate similarities
        cosine_sim = calculate_cosine_similarity(text1, text2)
        lstm_sim = calculate_lstm_similarity(text1, text2)
        bert_sim = calculate_bert_similarity(text1, text2)
        sbert_sim = calculate_sbert_similarity(text1, text2)
        sp = Spearmans_Rank_correlation(text1,text2)
        pc = Pearson_correlation(text1,text2)
        rmse = Root_mean_square_error(text1,text2)

        # Generate word frequency CSV
        word_freq_filename = generate_word_frequency_csv(text1, text2)
        correlation_csv_filename = generate_correlation_csv(sp,pc,rmse)
        similarities_csv_filename = generate_similarity_csv(cosine_sim, lstm_sim, bert_sim, sbert_sim)


        return render_template('index.html',
                               text1=text1,
                               text2=text2,
                               sp = sp,
                               pc = pc,
                               rmse = rmse,
                               cosine_sim=cosine_sim,
                               lstm_sim=lstm_sim,
                               bert_sim=bert_sim,
                               sbert_sim=sbert_sim,
                               similarities_csv_filename=similarities_csv_filename,
                               correlation_csv_filename = correlation_csv_filename,
                               word_freq_csv_filename=word_freq_filename)

    return render_template('index.html')


@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['STATIC_FOLDER'], filename), as_attachment=True)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    if not os.path.exists(app.config['STATIC_FOLDER']):
        os.makedirs(app.config['STATIC_FOLDER'])
    app.run(debug=True)
