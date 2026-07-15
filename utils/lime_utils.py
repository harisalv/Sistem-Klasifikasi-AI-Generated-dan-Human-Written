from lime.lime_text import LimeTextExplainer
from utils.predict_utils import predict_proba


# =====================================================
# CONSTANT LABEL
# =====================================================

HUMAN_CLASS = 0
AI_CLASS = 1


# =====================================================
# INIT EXPLAINER
# =====================================================

explainer_lime = LimeTextExplainer(
    class_names=[
        "Human Written",
        "AI Generated"
    ]
)


# =====================================================
# GENERATE LIME
# =====================================================

def generate_lime(
    text: str,
    num_features: int = 30,
    num_samples: int = 500
):
    """
    LIME explanation

    Konsisten dengan SHAP:

    Merah = AI
    Biru  = Human

    Selalu menjelaskan kelas AI
    """

    text = str(text)

    explanation = explainer_lime.explain_instance(
        text,
        predict_proba,
        labels=(AI_CLASS,),
        num_features=num_features,
        num_samples=num_samples
    )

    # ==========================================
    # TOP WORDS BERDASARKAN KELAS AI
    # ==========================================

    words = [
        word
        for word, score in explanation.as_list(
            label=AI_CLASS
        )
    ]

    # ==========================================
    # HTML BERDASARKAN KELAS AI
    # ==========================================

    html = explanation.as_html(
        labels=[AI_CLASS]
    )

    return {
        "html": html,
        "words": words
    }