import requests
from .config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from .prompts import SYSTEM_PROMPT, build_user_prompt

def generate_answer(question, contexts):
    if not LLM_API_KEY or not LLM_MODEL:
        raise RuntimeError("LLM_API_KEY and LLM_MODEL must be configured in .env")
    response = requests.post(
        f"{LLM_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": LLM_MODEL,
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(question, contexts)},
            ],
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()
