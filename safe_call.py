"""
Retry wrappers for Gemini calls.

503/500 errors use tenacity exponential backoff.
429 RESOURCE_EXHAUSTED errors sleep the retryDelay from the response body,
then retry inline — tenacity would exhaust attempts too fast on long delays.
"""

import re
import time
from google.genai import types, errors
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type


def _extract_retry_delay(exc: Exception) -> int:
    match = re.search(r"retryDelay.*?(\d+)s", str(exc))
    return int(match.group(1)) + 2 if match else 60


def _call_with_rate_limit_handling(fn, *args, max_attempts=6, **kwargs):
    """Call fn, sleeping on 429s and retrying up to max_attempts times."""
    for attempt in range(max_attempts):
        try:
            return fn(*args, **kwargs)
        except errors.ClientError as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                time.sleep(_extract_retry_delay(e))
            else:
                raise
    raise RuntimeError(f"Exceeded {max_attempts} attempts due to rate limiting")


@retry(
    retry=retry_if_exception_type(errors.ServerError),
    wait=wait_exponential(multiplier=2, min=10, max=120),
    stop=stop_after_attempt(5),
)
def _generate(client, model, contents):
    return _call_with_rate_limit_handling(
        client.models.generate_content,
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(temperature=0),
    )


def safe_generate(client, model, contents, **_kwargs):
    """Plain text Gemini call with backoff on server errors and inline retry on 429s."""
    return _generate(client, model, contents)


@retry(
    retry=retry_if_exception_type(errors.ServerError),
    wait=wait_exponential(multiplier=2, min=10, max=120),
    stop=stop_after_attempt(5),
)
def _generate_structured(client, model, contents, schema):
    response = _call_with_rate_limit_handling(
        client.models.generate_content,
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
    """Structured Gemini call with backoff on server errors and inline retry on 429s."""
    return _generate_structured(client, model, contents, schema)
