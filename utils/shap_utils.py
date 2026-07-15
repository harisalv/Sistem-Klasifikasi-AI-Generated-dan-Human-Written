import shap
import numpy as np
import torch

from utils.predict_utils import tokenizer, model


# ========================
# MODEL CPU (WAJIB UNTUK SHAP)
# ========================
model_cpu = model.cpu()


# ========================
# NORMALIZE INPUT
# ========================
def normalize_texts(texts):

    if isinstance(texts, str):
        return [texts]

    if isinstance(texts, np.ndarray):
        texts = texts.tolist()

    cleaned = []

    for t in texts:

        if isinstance(t, list):
            t = " ".join(map(str, t))

        cleaned.append(str(t))

    return cleaned


# ========================
# PREDICT PROBA (CPU VERSION)
# ========================
def predict_proba_cpu(texts):

    texts = normalize_texts(texts)

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )

    with torch.no_grad():

        outputs = model_cpu(**inputs)

        probs = torch.nn.functional.softmax(
            outputs.logits,
            dim=1
        )

    return probs.numpy()


# ========================
# INIT SHAP EXPLAINER
# ========================
masker = shap.maskers.Text(tokenizer)

explainer_shap = shap.Explainer(
    predict_proba_cpu,
    masker,
    algorithm="partition"
)


# ========================
# CONSTANT LABEL
# ========================
AI_CLASS = 1
HUMAN_CLASS = 0


# ========================
# GENERATE SHAP
# ========================
def generate_shap(text: str, class_idx: int):
    """
    Generate SHAP explanation.

    Visualisasi selalu menjelaskan kelas AI
    sehingga:

    Merah = mendukung AI
    Biru  = mendukung Human
    """

    text = str(text)

    shap_values = explainer_shap(
        [text],
        max_evals=300
    )

    # ==================================================
    # DEFAULT SHAP
    # Menjelaskan kelas yang diprediksi model
    # ==================================================
    explanation = shap.Explanation(
    values=shap_values.values[0][:, class_idx],
    base_values=shap_values.base_values[0][class_idx],
    data=shap_values.data[0]
)

    # ========================
    # TOP WORDS
    # ========================
    values = explanation.values
    tokens = explanation.data

    values = values / (np.sum(np.abs(values)) + 1e-8)

    top_idx = np.argsort(np.abs(values))[-30:]

    words = [tokens[i] for i in top_idx]

    # ========================
    # SHAP HTML
    # ========================
    shap_html = f"""
    <head>
        {shap.getjs()}
    </head>

    <body>
        {shap.plots.text(
            explanation,
            display=False
        )}
    </body>
    """

    return {
        "html": shap_html,
        "words": words
    }