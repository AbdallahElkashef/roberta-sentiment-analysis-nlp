import re
import pickle
import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------------------- Page Setup ----------------------------
st.set_page_config(
    page_title="Product Reviews Sentiment Analysis",
    layout="wide"
)

# ---------------------------- Text Cleaning ----------------------------
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^A-Za-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()

# ---------------------------- Load Model & Tokenizer ----------------------------
@st.cache_resource
def load_model():
    with open("deploy_bundle/model_bundle_cpu.pkl", "rb") as f:
        bundle = pickle.load(f)

    metadata = bundle["metadata"]
    state_dict = bundle["state_dict"]

    model = AutoModelForSequenceClassification.from_pretrained(
        metadata["model_base"],
        num_labels=metadata["num_labels"]
    )
    model.load_state_dict(state_dict)
    model.to("cpu")
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("deploy_bundle/tokenizer")

    id2label = metadata.get("id2label", {0: "negative", 1: "neutral", 2: "positive"})

    return model, tokenizer, id2label

model, tokenizer, ID2LABEL = load_model()
st.success("✅ Model and tokenizer loaded successfully on CPU")

# ---------------------------- Inference ----------------------------
def predict(text: str):
    text = clean_text(text)
    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )
    with torch.no_grad():
        logits = model(**enc).logits
        pred_id = int(torch.argmax(logits, dim=-1))
    return pred_id, ID2LABEL[pred_id]

# ---------------------------- UI ----------------------------
st.markdown("<h1 style='text-align:center;'>Product Reviews Sentiment Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Single review or batch CSV sentiment prediction</p>", unsafe_allow_html=True)
st.markdown("---")

mode = st.sidebar.radio("Input Mode", ["Single Review", "Batch CSV"])

# ---------------------------- Single Review ----------------------------
if mode == "Single Review":
    text = st.text_area("Enter a product review", height=150)
    if st.button("🔍 Predict"):
        if text.strip() == "":
            st.warning("Please enter text.")
        else:
            pred_id, label = predict(text)
            st.success("Prediction complete")
            st.write("**Predicted Sentiment:**", label)
            st.write("**Prediction ID:**", pred_id)
            st.write("**Cleaned Text:**")
            st.info(clean_text(text))

# ---------------------------- Batch CSV ----------------------------
if mode == "Batch CSV":
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
        if "Text" not in df.columns:
            st.error("CSV must contain a 'Text' column")
        else:
            if st.button("🚀 Run Batch Prediction"):
                results = []
                for t in df["Text"].astype(str):
                    if t.strip() == "":
                        results.append("empty")
                    else:
                        _, label = predict(t)
                        results.append(label)
                df["predicted_sentiment"] = results
                st.success("Batch prediction completed")
                st.dataframe(df.head())

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Results",
                    csv,
                    "sentiment_predictions.csv",
                    "text/csv"
                )

# ---------------------------- Footer ----------------------------
st.markdown("---")
st.caption("🚀 Fine-tuned RoBERTa sentiment classifier deployed on CPU using Streamlit")
