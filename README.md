# 🧠 NLP Text Similarity & Visualization Web App

This project is a powerful Flask-based web application that evaluates the similarity between two texts (or images containing text), leveraging various NLP techniques. It supports multiple similarity metrics, visual analytics, and word-level frequency insights. Ideal for research, academic comparisons, and document analysis.

## 🚀 Features

- ✅ Upload plain text or image (OCR-based text extraction)
- ✅ Calculate similarity using:
  - Cosine Similarity (TF-IDF)
  - BERT-based embeddings
  - Sentence-BERT (SBERT)
  - LSTM-based dummy similarity
- ✅ Statistical correlation metrics:
  - Pearson Correlation
  - Spearman's Rank Correlation
  - Root Mean Square Error (RMSE)
- ✅ Generate:
  - Word Frequency CSV
  - Similarity & Correlation CSVs
- ✅ Visualizations:
  - Word Cloud
  - Line, Bar, Pie, Scatter, Bubble, and Histogram charts

---

## 🛠️ Tech Stack

- **Backend**: Flask
- **NLP Models**:
  - `bert-base-uncased` (via HuggingFace Transformers)
  - `paraphrase-MiniLM-L6-v2` (SentenceTransformers)
- **Visualization**:
  - Matplotlib
  - Seaborn
  - Plotly
- **OCR**: Tesseract (via `pytesseract`)
- **Others**: Pandas, NumPy, Scikit-learn, PIL, WordCloud

---

## 📂 Folder Structure

project/ │ ├── app.py # Main Flask App ├── templates/ │ └── visualization.html # Visualization Page ├── static/ │ ├── *.csv # Auto-generated data files ├── uploads/ # Uploaded image files └── README.md # Project Documentation

---

## 🧪 Setup Instructions

### Clone the Repository


git clone https://github.com/ChAtulKumarPrusty/NLP-Text-Similarity-Visualization-WebApp.git
cd NLP-Text-Similarity-Visualization-WebApp\

Install Dependencies
pip install -r requirements.txt
If requirements.txt is missing, install manually:
pip install flask numpy pandas scikit-learn matplotlib seaborn plotly pytesseract pillow wordcloud tensorflow transformers sentence-transformers


### 🧰 Install Tesseract OCR

**For Windows Users:**

1. Download the Tesseract OCR installer from the official GitHub page:  
   👉 [https://github.com/tesseract-ocr/tesseract](https://github.com/tesseract-ocr/tesseract)
2. Install it and note the installation path (e.g., `C:\Program Files\Tesseract-OCR\tesseract.exe`)
3. Update your Python script (`app.py`) with the correct path:

import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

Run the Flask App
python app.py
Navigate to http://127.0.0.1:5000/visualization in your browser.

📈 Visualization Options
Graph Types: Line, Bar, Pie, Histogram, Scatter, Bubble
Data Sources: Word Frequencies, Similarities, Correlation Scores

📤 File Output
static/word_frequencies.csv – Word frequency count
static/similarities.csv – Cosine, BERT, SBERT, LSTM similarity
static/correlation.csv – Pearson, Spearman, RMSE correlations

👨‍💻 Authors
Atul Kumar Prusty  


Let me know if you want me to generate a `requirements.txt` file too, or add a GitHub Actions workflow for automatic deployment.
