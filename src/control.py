"""Control speech generation: single-layer natural filler insertion via Gemini.

Parallel to dysfluency.py — provides generate_control_ipa() for use in main.py.
"""

from src.phoneme import phonemize_text
from src.prompts import get_control_prompt

GEMINI_MODEL = "gemini-3-flash-preview"


def generate_control_ipa(client, ref_text: str) -> tuple[str, str]:
    """Run single-layer filler insertion, then phonemize.

    Args:
        client: google.genai.Client instance
        ref_text: fluent reference text

    Returns:
        (filler_text, ipa_string)
    """
    system_prompt = get_control_prompt()

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[ref_text],
        config={"system_instruction": system_prompt},
    )
    filler_text = response.text.strip()

    ipa = phonemize_text(filler_text)
    return filler_text, ipa
