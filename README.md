# Product Reviews Sentiment Analysis (Transformer-Based NLP)

### 📌 Overview



This project implements a **Transformer-based sentiment analysis system** for large-scale product reviews.

A fine-tuned **RoBERTa** model is used to classify customer reviews into:



* **Negative**
* **Neutral**
* **Positive**



The project covers the **full AI pipeline**:



Data preprocessing



Model fine-tuning



Evaluation using confusion matrix and classification report



CPU-friendly deployment using **Streamlit**



### 🚀 Features



Transformer-based sentiment classification (RoBERTa)



Robust text preprocessing for noisy user reviews



Single-review and batch CSV inference



Interactive **Streamlit web application**



CPU-only deployment compatible with **Streamlit Cloud**



Optional AI-powered review summarization using **Google Gemini**



### 📂 Project Structure

Product-Reviews-Sentiment-Analysis-Deployment/

│

├── app.py                     # Streamlit application

├── requirements.txt           # Python dependencies

├── config.toml                # Streamlit configuration

│

├── deploy\_bundle/

│   ├── model\_bundle\_cpu.pkl   # CPU-compatible model bundle (state\_dict + metadata)

│   └── tokenizer/             # HuggingFace tokenizer files

│

└── README.md                  # Project documentation

### 

### 🧠 Model Details



**Architecture**: RoBERTa-base



**Task**: 3-class sentiment classification



**Classes**:



* 0 → Negative
* 1 → Neutral
* 2 → Positive



**Max sequence length**: 128



**Frameworks**: PyTorch + HuggingFace Transformers



### 🧪 Evaluation



The model was evaluated on a held-out **test dataset** using:



Accuracy



Precision / Recall / F1-score



Confusion Matrix (visualized as a heatmap)



Most misclassifications occur between **neutral** and **positive**, which is consistent with natural language ambiguity in customer reviews.



### 🌐 Deployment (Streamlit)



The model is deployed using Streamlit, providing:



#### 🔹 Single Review Mode



Input a review text



View predicted sentiment



See cleaned text



Optional AI-generated summary (Gemini)



#### 🔹 Batch CSV Mode



Upload CSV file



Predict sentiment for all rows



Download results as CSV



### ⚙️ Installation \& Local Run

#### 1️⃣ Create environment

conda create -n sentiment\_app python=3.10

conda activate sentiment\_app



#### 2️⃣ Install dependencies

pip install -r requirements.txt



#### 3️⃣ Run Streamlit app

streamlit run app.py



### ☁️ Streamlit Cloud Deployment



Uses **CPU-only** inference



Model bundle is pre-converted from GPU → CPU



Tokenizer is loaded from *deploy\_bundle/tokenizer*





### 🧩 Key Insights



Transformer models significantly outperform classical ML approaches for sentiment analysis.



Proper tokenization and label normalization are critical for stability.



CPU-friendly deployment requires **state\_dict-based saving**, not raw GPU pickles.



Streamlit enables fast prototyping of production-ready NLP systems.



### 🔮 Future Work



Several extensions can be explored to further enhance the performance, robustness, and applicability of this sentiment analysis system:



#### Fine-Grained Sentiment Classification

The current model performs three-class sentiment classification (positive, neutral, negative). Future work could extend this task to **five-class prediction (1★ to 5★ ratings)**, enabling a more granular understanding of customer satisfaction and improving usefulness for recommendation and feedback systems.



#### Data Augmentation for Class Imbalance

Although the dataset is large, the neutral class remains underrepresented. A promising direction is to use **pretrained language generation models** to synthesize realistic neutral reviews, helping to balance the dataset and improve model generalization.



#### Evaluation of Larger Transformer Models

Performance could be further improved by experimenting with **larger and more expressive architectures**, such as **RoBERTa-large** or **DeBERTa**, which may capture deeper contextual and semantic relationships in customer reviews.



#### Hybrid Lexicon–Transformer Approaches

Exploring hybrid models that combine **lexicon-based sentiment analyzers (e.g., VADER)** with transformer-based architectures could provide complementary strengths, particularly for handling short, informal, or emotionally nuanced text.



These future directions open opportunities for improving accuracy, interpretability, and scalability, while extending the system toward more advanced real-world sentiment analysis applications.



### 👤 Authors



Abdallah ELKASHEF

Samsung Innovation Campus (SIC) – AI701 Internship

Department of Mechatronics \& Robotics Engineering

