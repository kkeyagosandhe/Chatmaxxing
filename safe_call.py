"""
Retry wrappers for Gemini calls.

Gemini occasionally returns 503 UNAVAILABLE or times out on longer runs.
These wrap the calls so a transient server error retries instead of crashing.
"""

import time
from google.genai import types, errors


def safe_generate(client, model, contents, max_retries=4, base_delay=3):
    """Plain text Gemini call with automatic retry on transient server errors."""
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(model=model, contents=contents, config=types.GenerateContentConfig(temperature=0),)
        except errors.ServerError as e:
            if attempt < max_retries - 1:
                wait = base_delay * (attempt + 1)
                print(f"  [retry] Gemini server error ({e.code}). "
                      f"Waiting {wait}s, attempt {attempt + 2}/{max_retries}...")
                time.sleep(wait)
            else:
                print(f"  [give up] Gemini still failing after {max_retries} attempts.")
                raise
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  [retry] Unexpected error: {e}. Retrying...")
                time.sleep(base_delay)
            else:
                raise


def safe_generate_structured(client, model, contents, schema, max_retries=4, base_delay=3):
    """Structured Gemini call: forces output into the given Pydantic schema.

    Returns a validated instance of `schema` (not raw text).
    """
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=0,

                ),
            )
            # response.parsed is the validated Pydantic object
            if response.parsed is not None:
                return response.parsed
            # fallback: validate from text if .parsed is empty
            return schema.model_validate_json(response.text)
        except errors.ServerError as e:
            if attempt < max_retries - 1:
                wait = base_delay * (attempt + 1)
                print(f"  [retry] Gemini server error ({e.code}). "
                      f"Waiting {wait}s, attempt {attempt + 2}/{max_retries}...")
                time.sleep(wait)
            else:
                print(f"  [give up] Gemini still failing after {max_retries} attempts.")
                raise
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  [retry] Unexpected error: {e}. Retrying...")
                time.sleep(base_delay)
            else:
                raise