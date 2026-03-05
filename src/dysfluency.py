"""L1 (word-level) + L2 (phoneme-level) dysfluency generation via Gemini.

Extracted from main.py for reuse and readability.
"""

from src.phoneme import phonemize_text
from src.prompts import get_prompts

GEMINI_MODEL = "gemini-3-flash-preview"


def generate_dysfluent_ipa(client, ref_text: str, severity: int) -> tuple[str, str]:
    """Run L1 (word-level) and L2 (phoneme-level) dysfluency generation.

    Args:
        client: google.genai.Client instance
        ref_text: fluent reference text
        severity: 0=mild, 1=moderate, 2=severe

    Returns:
        (word_level_text, dysfluent_ipa)
    """
    system_prompt_l1, system_prompt_l2 = get_prompts(severity)

    # L1: word-level dysfluency
    response_l1 = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[ref_text],
        config={"system_instruction": system_prompt_l1},
    )
    word_dys = response_l1.text.replace("...", "")

    # L2: phoneme-level dysfluency
    user_l2 = (
        f"DYS_REF_TEXT:\n{word_dys}\n\n"
        f"IPA_CORRECT:\n{phonemize_text(word_dys)}\n"
    )
    response_l2 = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[user_l2],
        config={"system_instruction": system_prompt_l2},
    )
    dysfluent_ipa = response_l2.text.strip()

    return word_dys, dysfluent_ipa
