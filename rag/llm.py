import logging
import time

import requests

from .config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from .prompts import SYSTEM_PROMPT, build_user_prompt

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_BACKOFF = 2.0  # seconds


def generate_answer(question: str, contexts: list[dict]) -> str:
    if not LLM_API_KEY or not LLM_MODEL:
        raise RuntimeError(
            "LLM_API_KEY and LLM_MODEL must be set in your .env file."
        )

    payload = {
        "model": LLM_MODEL,
        "temperature": 0.1,
        "max_tokens": 1024,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(question, contexts)},
        ],
    }
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    url = f"{LLM_BASE_URL}/chat/completions"

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            if response.status_code == 429:
                wait = _RETRY_BACKOFF * attempt
                logger.warning("Rate limited by LLM API. Retrying in %.1fs…", wait)
                time.sleep(wait)
                continue
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                f"Could not connect to LLM API at {LLM_BASE_URL}. "
                "Check LLM_BASE_URL in your .env file."
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise RuntimeError("LLM API request timed out after 60 seconds.") from exc
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code
            body = exc.response.text[:300]
            raise RuntimeError(
                f"LLM API returned HTTP {status}: {body}"
            ) from exc

    raise RuntimeError(f"LLM API rate limit exceeded after {_MAX_RETRIES} retries.")
