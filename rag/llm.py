import logging
import time

import requests

from .config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from .prompts import SYSTEM_PROMPT, build_user_prompt

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_BACKOFF = 2.0  # seconds
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


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
    last_error: Exception | None = None

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)

            if response.status_code in _RETRYABLE_STATUS:
                wait = _RETRY_BACKOFF * attempt
                logger.warning(
                    "LLM API returned %d (attempt %d/%d). Retrying in %.1fs…",
                    response.status_code, attempt, _MAX_RETRIES, wait,
                )
                time.sleep(wait)
                last_error = RuntimeError(
                    f"LLM API returned HTTP {response.status_code}: {response.text[:200]}"
                )
                continue

            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()

        except requests.exceptions.ConnectionError:
            wait = _RETRY_BACKOFF * attempt
            logger.warning("Connection error (attempt %d/%d). Retrying in %.1fs…",
                           attempt, _MAX_RETRIES, wait)
            time.sleep(wait)
            last_error = RuntimeError(
                f"Could not connect to LLM API at {LLM_BASE_URL}. "
                "Check LLM_BASE_URL in your .env file."
            )
        except requests.exceptions.Timeout:
            wait = _RETRY_BACKOFF * attempt
            logger.warning("Request timed out (attempt %d/%d). Retrying in %.1fs…",
                           attempt, _MAX_RETRIES, wait)
            time.sleep(wait)
            last_error = RuntimeError("LLM API request timed out after 60 seconds.")
        except requests.exceptions.HTTPError as exc:
            # Non-retryable HTTP error (4xx except 429)
            status = exc.response.status_code
            body = exc.response.text[:300]
            raise RuntimeError(f"LLM API returned HTTP {status}: {body}") from exc

    raise last_error or RuntimeError(
        f"LLM API failed after {_MAX_RETRIES} attempts."
    )
