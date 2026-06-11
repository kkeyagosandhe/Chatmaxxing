"""
Retry wrappers for Gemini calls.

Gemini occasionally returns 503 UNAVAILABLE or 429 RESOURCE_EXHAUSTED.
These wrap the calls so transient errors retry with appropriate backoff.
"""

import re
import time
from google.genai import types, errors
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type


def _before_sleep(retry_state):
    """If the last exception was a 429 with a retryDelay, honour it."""
    exc = retry_state.outcome.exception()
    if exc:
        match = re.search(r"retryDelay.*?(\d+)s", str(exc))
        if match:
            time.sleep(int(match.group(1)) + 2)


@retry(
    retry=retry_if_exception_type((errors.ServerError, errors.ClientError)),
    wait=wait_exponential(multiplier=2, min=10, max=120),
    stop=stop_after_attempt(8),
    before_sleep=_before_sleep,
)
def _generate(client, model, contents):
    return client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(temperature=0),
    )


def safe_generate(client, model, contents, **_kwargs):
    """Plain text Gemini call with exponential backoff on rate limits and server errors."""
    return _generate(client, model, contents)


@retry(
    retry=retry_if_exception_type((errors.ServerError, errors.ClientError)),
    wait=wait_exponential(multiplier=2, min=10, max=120),
    stop=stop_after_attempt(8),
    before_sleep=_before_sleep,
)
def _generate_structured(client, model, contents, schema):
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0,
        ),
    )
    if response.parsed is not None:
        return response.parsed
    return schema.model_validate_json(response.text)


def safe_generate_structured(client, model, contents, schema, **_kwargs):
    """Structured Gemini call with exponential backoff. Returns a validated Pydantic instance."""
    return _generate_structured(client, model, contents, schema)
