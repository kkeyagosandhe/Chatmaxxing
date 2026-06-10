"""
Retry wrappers for Gemini calls.

Gemini occasionally returns 503 UNAVAILABLE or times out on longer runs.
These wrap the calls so a transient server error retries instead of crashing.
"""

from google.genai import types, errors
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type


def _is_retryable(e: Exception) -> bool:
    return isinstance(e, (errors.ServerError, errors.ClientError))


@retry(
    retry=retry_if_exception_type((errors.ServerError, errors.ClientError)),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5),
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
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5),
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