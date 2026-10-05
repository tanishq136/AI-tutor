"""
offline/cleaner.py
------------------
Knowledge-aware transcript cleaner using Azure OpenAI.

One LLM call does two things simultaneously:
  1. Cleans the raw transcript (grammar, fillers, flow)
  2. Extracts structured metadata (title, topics, key terms, difficulty, summary)

Returns: (cleaned_text: str, metadata: dict)
Falls back to (raw_text, {}) if the Azure call fails or keys are missing.
"""

import json
from ..config import get_azure_client, emit

_SYSTEM_PROMPT = """\
You are a lecture transcript processor.
Given a raw lecture transcript, return a JSON object with exactly these keys:

{
  "cleaned_text": "<the full transcript with grammar fixed and improved readability. Preserve all content.>",
  "metadata": {
    "lecture_title":  "<inferred title or Unknown>",
    "topics":         ["<topic_1>", "<topic_2>"],
    "key_terms":      ["<term_1>", "<term_2>"],
    "difficulty":     "beginner or intermediate or advanced",
    "summary":        "<2 sentences describing the main content>"
  }
}

Return ONLY valid JSON. No markdown. No extra keys. No explanation."""


def clean_and_extract(raw_text: str, source_name: str = "source") -> tuple[str, dict]:
    """
    Clean transcript and extract metadata via Azure OpenAI.
    Falls back to (raw_text, {}) gracefully if Azure is unavailable.
    """
    result = get_azure_client()
    if result is None:
        emit(f"[{source_name}] Azure keys missing — indexing raw text.", "error")
        return raw_text, {}
    client, deployment = result

    try:
        emit(f"[{source_name}] Cleaning via Azure OpenAI ({deployment})…")
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": raw_text[:12_000]},
            ],
            temperature=0.1,
            max_tokens=4096,
            response_format={"type": "json_object"},
        )
        payload  = json.loads(response.choices[0].message.content)
        cleaned  = payload.get("cleaned_text", raw_text)
        metadata = payload.get("metadata", {})
        emit(f"[{source_name}] Cleaned + metadata extracted.", "success")
        return cleaned, metadata

    except json.JSONDecodeError as e:
        emit(f"[{source_name}] JSON parse error ({e}) — using raw text.", "error")
        return raw_text, {}
    except Exception as e:
        emit(f"[{source_name}] Azure cleaner error ({e}) — using raw text.", "error")
        return raw_text, {}
