import streamlit as st

# ========================
# IMPORT MODULE
# ========================
from utils.text_utils import validate_text
from utils.predict_utils import predict_text
from utils.lime_utils import generate_lime
from utils.shap_utils import generate_shap

from components.lime_component import render_lime
from components.shap_component import render_shap


# ========================
# CONFIG PAGE
# ========================
st.set_page_config(
    page_title="AI vs Human Text Detector",
    layout="wide"
)

# ========================
# CUSTOM CSS
# ========================
st.markdown("""
<style>

/* Background utama */
.stApp {
    background-color: white;
}

/* Warna text area */
textarea {
    background-color: white !important;
    color: black !important;
    caret-color: black !important; /* WARNA KURSOR */
}

/* Input text */
input {
    color: black !important;
    caret-color: black !important;
}

/* Tombol */
.stButton>button {
    background-color: white !important;
    color: black !important;
    border: 1px solid #cccccc !important;
}

/* Cursor fokus */
textarea:focus, input:focus {
    border: 1px solid #999999 !important;
    box-shadow: none !important;
}

</style>
""", unsafe_allow_html=True)

# ========================
# WHITE BACKGROUND + LIGHT BUTTON STYLE
# ========================
st.markdown("""
    <style>
    .stApp {
        background-color: white;
        color: black;
    }

    textarea {
        background-color: white !important;
        color: black !important;
    }

    .stTextArea textarea {
        background-color: white !important;
        color: black !important;
    }

    div[data-testid="stMarkdownContainer"] p {
        color: black;
    }

    h1, h2, h3 {
        color: black;
    }

    /* BUTTON STYLE */
    .stButton > button {
        background-color: #f5f5f5 !important;
        color: black !important;
        border: 1px solid #dcdcdc !important;
        border-radius: 10px !important;
    }

    .stButton > button:hover {
        background-color: #eaeaea !important;
        color: black !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 AI vs Human Text Detection with XAI")
st.markdown(
    "Text classification system for detecting **AI-generated** and **human-written** text using **BERT + LIME + SHAP**."
)

# ========================
# SAMPLE TEXT
# ========================
sample_ai = """
The rain started just after midnight, tapping softly against the windows of the small apartment overlooking the river. Daniel sat at his desk with a cup of cold coffee beside him, trying to finish a report that he had postponed for nearly two weeks. Outside, the city looked unusually calm. The usual sounds of traffic had faded, replaced by the occasional rumble of thunder in the distance. For some reason, nights like this always made him think more clearly. He began typing again, slowly at first, then faster as ideas finally started to connect in his mind.
Earlier that day, he had visited a bookstore hidden between two old buildings downtown. The shop smelled of dust and paper, and every shelf seemed packed beyond capacity. An elderly woman behind the counter recommended a novel that Daniel had never heard of before. She claimed it was the kind of story that changed the way people viewed ordinary life. Although he doubted her dramatic description, he bought the book anyway. By the time he returned home, he found himself strangely excited to start reading it, even more excited than he was about completing his work.
Meanwhile, across the river, a small group of university students gathered in a café that stayed open all night. They were preparing for final exams, though most of their time was spent debating random topics unrelated to their studies. One of them insisted that technology was making people less patient, while another argued that it simply changed the way humans interacted with information. Their discussion continued for hours, interrupted only by laughter and the arrival of fresh cups of coffee. Despite their different opinions, they all shared the same exhaustion that comes near the end of a long semester.
In another part of the city, a stray orange cat wandered through narrow alleyways searching for shelter from the rain. It eventually curled up beneath the awning of a closed bakery, protected from the cold wind. Early the next morning, the bakery owner arrived and noticed the animal sleeping quietly near the door. Instead of chasing it away, he brought out a small bowl of water and a piece of bread. The cat remained cautious at first, but hunger eventually overcame fear. From that day on, the bakery owner often found the cat waiting outside each morning before sunrise.
""".replace("\n", " ")

sample_human = """
On Tuesday, Americans stood in line[s] that stretched around schools and churches in numbers this nation has never seen. It didn’t matter who they were or where they came from or what they looked like or what Party they belonged to; they came out and cast their ballot because they believed that in this country our destiny is not written for us, but by us. We should all take pride in the fact that we once again displayed for the world the power of our democracy, and reaffirmed that the great American ideal that this is a nation where anything is possible.
This week, I spoke with President Bush, who graciously offered his full support and assistance in this period of transition. Michelle and I look forward to meeting with him and the First Lady on Monday to begin that process.
This speaks to a fundamental recognition that here in America we can compete vigorously in elections and challenge each other's ideas, yet come together in service of a common purpose once the voting is done. And that is particularly important at a moment when we face the most serious challenges of our lifetime.
Yesterday, we woke up to more sobering news about the state of our economy. The 240,000 jobs lost in October marks the tenth consecutive month that our economy has shed jobs. In total, we've lost nearly 1.2 million jobs this year; and more than 10 million Americans are now unemployed. Tens of millions of families are struggling to figure out how to pay the bills and stay in their homes. Their stories are an urgent reminder that we are facing the greatest economic challenge of our lifetime; and we must act swiftly to resolve them.
In the wake of these disturbing reports, I met with members of my Transition Economic Advisory Board who will help guide the work of my transition team in developing a strong set of policies to respond to this crisis; though we must recognize that we only have one President at a time, and that President Bush is the leader of our government. I want to ensure that we hit the ground running on January 20th because we don't have a moment to lose.
We discussed several of the most immediate challenges facing our economy and key priorities on which to focus in the days and weeks ahead to ease the credit crisis, help hard-working families, and restore growth and prosperity. First, we need a rescue plan for the middle class that invests in immediate efforts to create jobs and provides relief to families that are watching their paychecks shrink and their life savings disappear. Then, we'll address the spreading impact to the financial crisis on other sectors of our economy and ensure that the rescue plan that passed Congress is working to stabilize financial markets while protecting tax payers, helping home owners, and not unduly rewarding the management of financial firms that are receiving government assistance.
Finally, we will move forward with a set of policies that will grow our middle class and strengthen our economy in the long term. We can't afford to wait on moving forward on the key priorities that I identified during the campaign, including clean energy, health care, education, and tax relief for middle-class families.
Let me close by saying I do not underestimate the enormity of the task that lies ahead. We've taken some major actions to date and we will need further actions during this transition and subsequent months. Some of those choices will be difficult, but America is a strong and resilient country. I know that we will succeed, if we put aside partisanship and work together as one nation. And that is what I intend to do.
""".replace("\n"," ")
col1, col2 = st.columns(2)

with col1:
    if st.button("Use AI Sample"):
        st.session_state.text_input = sample_ai

with col2:
    if st.button("Use Human Sample"):
        st.session_state.text_input = sample_human


# ========================
# INPUT TEXT
# ========================
text_input = st.text_area(
    "Enter text (minimum 3 words):",
    value=st.session_state.get("text_input", ""),
    height=200
)


# ========================
# ANALYZE BUTTON
# ========================
if st.button("🔍 Analyze"):
    # ========================
    # INPUT VALIDATION
    # ========================
    validation = validate_text(text_input)

    if not validation["valid"]:
        st.error(validation["error"])
        st.stop()

    # ========================
    # ENGLISH WARNING
    # ========================
    if validation["warning"]:
        st.warning(
            "The entered text is not in English. "
            "Prediction results may be inaccurate."
        )

    clean_text = validation["clean_text"]

    # ========================
    # PROCESSING
    # ========================
    with st.spinner("Processing model prediction..."):

        # ========================
        # PREDICTION
        # ========================
        result = predict_text(clean_text)

        label = result["label"]
        confidence = result["confidence"]
        class_idx = result["class_index"]

        st.subheader("📊 Prediction Result")
        st.success(f"**Predicted Label:** {label}")
        st.info(f"**Confidence Score:** {confidence * 100:.2f}%")

        # ========================
        # LIME
        # ========================
        lime_result = generate_lime(clean_text)
        render_lime(lime_result["html"])

        # ========================
        # SHAP
        # ========================
        shap_result = generate_shap(clean_text, class_idx)
        render_shap(shap_result["html"]),