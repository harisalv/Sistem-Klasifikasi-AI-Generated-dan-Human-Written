import re
from langdetect import detect, LangDetectException


def clean_text(text: str) -> str:
    """
    Membersihkan teks dari spasi berlebih
    """
    return text.strip()


def count_words(text: str) -> int:
    """
    Menghitung jumlah kata (bukan karakter)
    """
    words = re.findall(r'\b\w+\b', text)
    return len(words)


def validate_min_words(text: str, min_words: int = 3):
    """
    Validasi jumlah minimal kata
    """
    word_count = count_words(text)

    if word_count < min_words:
        return False, f"Input must contain at least {min_words} words."

    return True, ""


def detect_language(text: str) -> str:
    """
    Deteksi bahasa menggunakan langdetect
    """
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def validate_text(text: str) -> dict:
    """
    Fungsi utama validasi input

    Return format:
    {
        "valid": bool,
        "error": str,
        "warning": str,
        "clean_text": str
    }
    """

    result = {
        "valid": True,
        "error": "",
        "warning": "",
        "clean_text": ""
    }

    # 1. Bersihkan teks
    text = clean_text(text)

    if text == "":
        result["valid"] = False
        result["error"] = "Input text cannot be empty."
        return result

    # 2. Validasi jumlah kata
    is_valid, error_msg = validate_min_words(text)
    if not is_valid:
        result["valid"] = False
        result["error"] = error_msg
        return result

    # 3. Deteksi bahasa
    lang = detect_language(text)

    if lang != "en":
        result["warning"] = "The entered text is not written in English. Prediction results may be less accurate."

    # 4. Simpan teks bersih
    result["clean_text"] = text

    return result