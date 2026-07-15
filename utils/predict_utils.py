import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ========================
# CONFIG
# ========================
MODEL_PATH = "best_model"  # pastikan folder model kamu di sini

# ========================
# LOAD MODEL (HANYA SEKALI)
# ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

model.to(device)
model.eval()


# ========================
# PREDICT PROBA (UNTUK LIME & UI)
# ========================
def predict_proba(texts):
    """
    Input:
        texts: str atau list[str]

    Output:
        numpy array (probabilities)
    """

    if isinstance(texts, str):
        texts = [texts]

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)

    return probs.cpu().numpy()


# ========================
# PREDICT LABEL + CONFIDENCE
# ========================
def predict_text(text: str) -> dict:
    """
    Output:
    {
        "label": "AI Generated" / "Human Written",
        "confidence": float,
        "probs": [human_prob, ai_prob]
    }
    """

    probs = predict_proba(text)[0]
    pred_class = int(np.argmax(probs))

    label = "AI Generated" if pred_class == 1 else "Human Written"
    confidence = float(probs[pred_class])

    return {
        "label": label,
        "confidence": confidence,
        "probs": probs,
        "class_index": pred_class
    }