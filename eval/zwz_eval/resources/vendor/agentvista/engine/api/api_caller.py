"""
API caller for vision chat-completions. Uses REASONING_API_KEY and REASONING_END_POINT from env.
Retries on 429/5xx; optional fallback via REASONING_API_KEY_2 and REASONING_END_POINT_2.
"""

import os
import time
import random
import json
import requests
import re
import hashlib
import tempfile
import fcntl

os.environ["VERBOSE_PARAM_VALIDATION"] = "false"
os.environ["DEBUG_FULL_RESPONSE"] = "false"

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
NON_RETRYABLE_STATUS_CODES = {400, 401, 403, 404}
API_TIMEOUT = 240
BASE_WAIT_TIME = 1.0
MAX_WAIT_TIME = 15.0
ROUND_ROBIN_429_WAIT_MIN = 5.0
ROUND_ROBIN_429_WAIT_MAX = 10.0
HIGH_TOKEN_THRESHOLD = 25000
DEFAULT_MAX_RETRIES = 5


def _wait_for_per_key_rate_limit(api_key: str, api_name: str) -> None:
    """Apply a process-shared request rate limit independently per API key.

    AgentVista workers are separate processes, so a per-process sleep would
    exceed provider RPM limits as soon as several shards run concurrently.
    This small lock-protected timestamp file serializes dispatches for one
    credential while leaving different credentials independent.
    """
    raw_rpm = os.environ.get("AGENTVISTA_RPM_PER_KEY", "").strip()
    if not raw_rpm:
        return
    try:
        rpm = float(raw_rpm)
        if rpm <= 0:
            return
    except ValueError:
        print(f"[{api_name}] Ignoring invalid AGENTVISTA_RPM_PER_KEY={raw_rpm!r}")
        return

    state_dir = os.environ.get("AGENTVISTA_RATE_LIMIT_DIR", tempfile.gettempdir())
    os.makedirs(state_dir, exist_ok=True)
    fingerprint = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]
    state_path = os.path.join(state_dir, f"agentvista-rpm-{fingerprint}.lock")
    interval = 60.0 / rpm
    with open(state_path, "a+", encoding="utf-8") as state:
        fcntl.flock(state.fileno(), fcntl.LOCK_EX)
        state.seek(0)
        try:
            next_allowed = float(state.read().strip() or "0")
        except ValueError:
            next_allowed = 0.0
        now = time.monotonic()
        if next_allowed > now:
            time.sleep(next_allowed - now)
            now = time.monotonic()
        state.seek(0)
        state.truncate()
        state.write(str(now + interval))
        state.flush()
        fcntl.flock(state.fileno(), fcntl.LOCK_UN)


def _alternate_api_order(api_entries: list) -> list:
    """Rotate a credential pool across all worker processes."""
    state_dir = os.environ.get("AGENTVISTA_RATE_LIMIT_DIR", tempfile.gettempdir())
    os.makedirs(state_dir, exist_ok=True)
    fingerprint = hashlib.sha256("\n".join(entry[0] for entry in api_entries).encode("utf-8")).hexdigest()[:16]
    state_path = os.path.join(state_dir, f"agentvista-api-rotation-{fingerprint}.lock")
    with open(state_path, "a+", encoding="utf-8") as state:
        fcntl.flock(state.fileno(), fcntl.LOCK_EX)
        state.seek(0)
        try:
            index = int(state.read().strip() or "0")
        except ValueError:
            index = 0
        state.seek(0)
        state.truncate()
        state.write(str(index + 1))
        state.flush()
        fcntl.flock(state.fileno(), fcntl.LOCK_UN)
    offset = index % len(api_entries)
    return api_entries[offset:] + api_entries[:offset]

# ====================== API config functions ======================================

def _build_payload(model_name: str, messages: list, sampling_params: dict, tools: list = None):
    """Build the API request payload."""
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": sampling_params['temperature'],
        "top_p": sampling_params['top_p'],
        "max_tokens": sampling_params['max_tokens'],
    }

    disable_thinking = os.environ.get("AGENTVISTA_DISABLE_THINKING", "").lower() in (
        "1",
        "true",
        "yes",
    )
    # The AIStudio GLM service consumes enable_thinking as a top-level field.
    # Local vLLM/Qwen services consume it through chat_template_kwargs.
    def set_thinking_disabled():
        if model_name.lower().startswith("glm-"):
            payload["enable_thinking"] = False
        else:
            payload["chat_template_kwargs"] = {"enable_thinking": False}

    if disable_thinking:
        set_thinking_disabled()

    if tools:
        payload["tools"] = tools
        payload["parallel_tool_calls"] = False
    elif not disable_thinking and os.environ.get(
        "AGENTVISTA_DISABLE_THINKING_WHEN_NO_TOOLS", ""
    ).lower() in ("1", "true", "yes"):
        # Qwen models may spend the entire completion budget in
        # reasoning_content on the forced final turn.  Keep thinking enabled
        # while tools are available, then request a normal answer when the
        # hard tool budget removes the schema.
        set_thinking_disabled()

    return payload


def _add_reasoning_param(payload: dict, model_name: str, api_name: str = "API", end_point: str = None):
    """
    Add reasoning parameters based on model family and API provider.

    Supported model families:
    - GPT series (gpt-5, o1, o3, o4-mini, etc.): uses reasoning_effort (OpenAI format)
    - Gemini series: uses reasoning: { effort } or reasoning: { max_tokens } (OpenRouter unified)
    - Claude series: uses reasoning: { effort } or reasoning: { max_tokens } (OpenRouter unified)
    - Grok series: uses reasoning: { effort } or reasoning: { max_tokens } (OpenRouter unified)
    """
    model_lower = model_name.lower()
    is_gemini_model = "gemini" in model_lower
    # AIStudio's OpenAI-compatible GLM proxy accepts the OpenAI top-level
    # reasoning_effort knob.  It is intentionally distinct from the GLM
    # top-level enable_thinking switch handled in _build_payload above.
    is_glm_model = model_lower.startswith("glm-")
    is_gpt_model = "gpt-5" in model_lower or "o1" in model_lower or "o3" in model_lower or "o4" in model_lower
    is_claude_model = "claude" in model_lower
    is_grok_model = "grok" in model_lower

    is_openrouter = False
    if end_point:
        is_openrouter = "openrouter" in end_point.lower()

    reasoning_max_tokens = os.environ.get("REASONING_MAX_TOKENS")
    reasoning_effort = os.environ.get("REASONING_EFFORT")

    # GPT series: use OpenAI-format reasoning_effort (all providers support this)
    if is_gpt_model or is_glm_model:
        if reasoning_effort and str(reasoning_effort).strip().lower() not in ["none", "false", ""]:
            payload["reasoning_effort"] = str(reasoning_effort).strip().lower()

    # Other models on OpenRouter (Gemini, Claude, Grok): use unified reasoning interface
    elif is_openrouter and (is_gemini_model or is_claude_model or is_grok_model):
        if reasoning_max_tokens and str(reasoning_max_tokens).strip().lower() not in ["none", "false", ""]:
            payload["reasoning"] = {
                "max_tokens": int(reasoning_max_tokens),
                "exclude": False
            }
        elif reasoning_effort and str(reasoning_effort).strip().lower() not in ["none", "false", ""]:
            payload["reasoning"] = {
                "effort": str(reasoning_effort).strip().lower(),
                "exclude": False
            }

    # Non-OpenRouter Gemini models: use effort parameter
    elif is_gemini_model:
        if reasoning_effort and str(reasoning_effort).strip().lower() not in ["none", "false", ""]:
            payload["reasoning"] = {
                "effort": str(reasoning_effort).strip().lower(),
                "exclude": False
            }


def _make_api_request(end_point: str, headers: dict, payload: dict, api_name: str = "API"):
    """
    Send an API request and return the response.

    Returns:
        tuple: (response, error_type)
               response: requests.Response or None
               error_type: "429", "timeout", "network", "other", or None (success)
    """
    try:
        api_key = headers.get("Authorization", "").removeprefix("Bearer ")
        if api_key:
            _wait_for_per_key_rate_limit(api_key, api_name)
        response = requests.post(end_point, headers=headers, json=payload, timeout=API_TIMEOUT)
        if response.status_code == 429:
            return response, "429"
        return response, None
    except requests.exceptions.Timeout:
        print(f"[{api_name}] API timeout")
        return None, "timeout"
    except requests.exceptions.RequestException as e:
        print(f"[{api_name}] API call failed: {e}")
        return None, "network"
    except Exception as e:
        print(f"[{api_name}] Unexpected error: {e}")
        return None, "other"


def _parse_api_response(response, payload: dict, api_name: str, api_key: str, end_point: str,
                       attempt: int = 0, max_attempts: int = 1):
    """
    Parse the API response.

    Returns:
        tuple: (result, is_429, error_type)
               result: parsed result (dict/str/None)
               is_429: whether a 429 error occurred
               error_type: one of "http_error", "empty_choices", "empty_content", "invalid_format", or None
    """
    if response.status_code != 200:
        error_info = None
        error_text = None
        try:
            error_info = response.json()
            error_text = str(error_info)
            attempt_str = f" on attempt {attempt + 1}/{max_attempts}" if max_attempts > 1 else ""
            print(f"[{api_name}] API Error {response.status_code}{attempt_str}: {error_info}")
        except:
            error_text = response.text
            attempt_str = f" on attempt {attempt + 1}/{max_attempts}" if max_attempts > 1 else ""
            print(f"[{api_name}] API Error {response.status_code}{attempt_str}: {response.text}")

        if attempt == 0:
            _validate_payload_params(payload, response.status_code, error_info, error_text, api_name,
                                    response_result=None, api_key=api_key, end_point=end_point)

        is_429 = (response.status_code == 429)
        return None, is_429, "http_error"

    try:
        result = response.json()
    except json.JSONDecodeError as e:
        print(f"[{api_name}] Failed to parse JSON response: {e}")
        return None, False, "invalid_format"

    if attempt == 0 and os.environ.get("DEBUG_FULL_RESPONSE", "").lower() in ["true", "1", "yes"]:
        try:
            print(f"[{api_name}] FULL RAW RESPONSE:\n{json.dumps(result, ensure_ascii=False, indent=2)}")
        except Exception:
            print(f"[{api_name}] FULL RAW RESPONSE (non-serializable): {result}")

    if attempt == 0:
        _validate_payload_params(payload, response.status_code, None, None, api_name,
                                response_result=result, api_key=api_key, end_point=end_point)

    if "choices" in result and result["choices"]:
        message = result["choices"][0]["message"]

        if "tool_calls" in message and message["tool_calls"]:
            return message, False, None

        content = message.get("content")
        if content and content.strip():
            if "reasoning_details" in message or "reasoning" in message:
                return message, False, None
            return content, False, None
        else:
            attempt_str = f" on attempt {attempt + 1}/{max_attempts}" if max_attempts > 1 else ""
            print(f"[{api_name}] API returned empty content{attempt_str}")
            if max_attempts > 1:
                try:
                    print(f"[{api_name}] Empty message detail: {json.dumps(message, ensure_ascii=False)}")
                except Exception:
                    print(f"[{api_name}] Empty message detail: {message}")
            return None, False, "empty_content"
    else:
        if "choices" in result and len(result["choices"]) == 0:
            attempt_str = f" on attempt {attempt + 1}/{max_attempts}" if max_attempts > 1 else ""
            print(f"[{api_name}] Model refused to generate (empty choices){attempt_str}")
            usage = result.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            print(f"[{api_name}] Usage: prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}")

            if "error" in result:
                print(f"[{api_name}] Error details: {result['error']}")

            if prompt_tokens > HIGH_TOKEN_THRESHOLD:
                print(f"[{api_name}] Warning: Very high prompt_tokens ({prompt_tokens}), may have hit token limit")

            print(f"[{api_name}] Possible causes: content filtering, safety policy, token limit, or model overload")
            return None, False, "empty_choices"
        else:
            attempt_str = f" on attempt {attempt + 1}/{max_attempts}" if max_attempts > 1 else ""
            print(f"[{api_name}] Invalid API response format{attempt_str}: {result}")
            return None, False, "invalid_format"


# ====================== Debug information ======================================

def _extract_invalid_params_from_error(error_info, error_text):
    """Extract names of invalid parameters from the error message."""
    invalid_params = []

    if isinstance(error_info, dict):
        error_msg = str(error_info.get("error", {}))
        if isinstance(error_info.get("error"), dict):
            error_msg = str(error_info["error"].get("message", ""))
        elif isinstance(error_info.get("error"), str):
            error_msg = error_info["error"]

        patterns = [
            r"unknown parameter[:\s]+['\"]?(\w+)['\"]?",
            r"invalid parameter[:\s]+['\"]?(\w+)['\"]?",
            r"parameter ['\"]?(\w+)['\"]? (?:is not|not) (?:supported|allowed|valid)",
            r"['\"]?(\w+)['\"]? (?:is not|not) a valid parameter",
            r"unexpected parameter[:\s]+['\"]?(\w+)['\"]?",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, error_msg, re.IGNORECASE)
            invalid_params.extend(matches)

    if error_text:
        patterns = [
            r"unknown parameter[:\s]+['\"]?(\w+)['\"]?",
            r"invalid parameter[:\s]+['\"]?(\w+)['\"]?",
            r"parameter ['\"]?(\w+)['\"]? (?:is not|not) (?:supported|allowed|valid)",
            r"['\"]?(\w+)['\"]? (?:is not|not) a valid parameter",
            r"unexpected parameter[:\s]+['\"]?(\w+)['\"]?",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, error_text, re.IGNORECASE)
            invalid_params.extend(matches)

    return list(set(invalid_params))


def _get_model_info_from_api(api_key, end_point, model_name, api_name="API"):
    """
    Attempt to retrieve model information from the models list API.

    Returns:
        dict: Model information, or None if unavailable.
    """
    try:
        if "/chat/completions" in end_point:
            models_endpoint = end_point.replace("/chat/completions", "/models")
        elif "/v1/" in end_point:
            base_url = end_point.rsplit("/v1/", 1)[0]
            models_endpoint = f"{base_url}/v1/models"
        else:
            return None

        headers = {
            "Authorization": f"Bearer {api_key}",
        }

        response = requests.get(models_endpoint, headers=headers, timeout=10)
        if response.status_code == 200:
            models_data = response.json()
            if "data" in models_data:
                for model in models_data["data"]:
                    if model.get("id") == model_name:
                        return model
    except Exception:
        pass

    return None


def _format_param_value(value, max_length=50):
    """Format a parameter value for display."""
    if isinstance(value, dict):
        if "effort" in value:
            return f"{{effort: {value.get('effort')}, exclude: {value.get('exclude', False)}}}"
        return f"{{...}} ({len(value)} keys)"
    elif isinstance(value, list):
        return f"[{len(value)} items]"
    elif isinstance(value, str) and len(value) > max_length:
        return value[:max_length] + "..."
    else:
        return str(value)


def _validate_payload_params(payload, response_status, error_info=None, error_text=None,
                            api_name="API", response_result=None, api_key=None, end_point=None):
    """Validate payload parameters and print a diagnostic report."""
    payload_params = list(payload.keys())

    verbose = os.environ.get("VERBOSE_PARAM_VALIDATION", "").lower() in ["true", "1", "yes"]

    if response_status == 200:
        if verbose:
            print(f"[{api_name}] Payload param validation: request succeeded (HTTP 200)")
            print(f"[{api_name}]   Note: HTTP 200 only means the request was accepted; not all params may have taken effect")
            print(f"[{api_name}]   Parameters used ({len(payload_params)}):")

            for param in payload_params:
                value = payload[param]
                formatted_value = _format_param_value(value)
                print(f"[{api_name}]     - {param}: {formatted_value}")

            if response_result:
                if "reasoning" in payload or "reasoning_effort" in payload:
                    has_reasoning = False
                    reasoning_locations = []

                    if isinstance(response_result, dict):
                        if any(key in response_result for key in ["reasoning", "reasoning_details", "reasoning_content"]):
                            has_reasoning = True
                            reasoning_locations.append("top-level")

                        if "choices" in response_result and response_result["choices"]:
                            message = response_result["choices"][0].get("message", {})
                            if any(key in message for key in ["reasoning", "reasoning_details", "reasoning_content"]):
                                has_reasoning = True
                                reasoning_locations.append("choices[0].message")

                        if "choices" in response_result and response_result["choices"]:
                            choice = response_result["choices"][0]
                            if any(key in choice for key in ["reasoning", "reasoning_details", "reasoning_content"]):
                                has_reasoning = True
                                reasoning_locations.append("choices[0]")

                    if has_reasoning:
                        print(f"[{api_name}]   reasoning param active: found reasoning content in {', '.join(reasoning_locations)}")
                    else:
                        print(f"[{api_name}]   reasoning param may not be active: no reasoning field found (checked top-level, choices[0], choices[0].message)")

                if "tools" in payload:
                    has_tool_calls = False
                    if isinstance(response_result, dict):
                        has_tool_calls = "tool_calls" in response_result
                        if has_tool_calls:
                            print(f"[{api_name}]   tools param active: response contains tool_calls")
                        else:
                            print(f"[{api_name}]   tools param set but no tool_calls in this response")

            if api_key and end_point:
                model_name = payload.get("model")
                if model_name:
                    model_info = _get_model_info_from_api(api_key, end_point, model_name, api_name)
                    if model_info:
                        print(f"[{api_name}]   Model info:")
                        if "context_length" in model_info:
                            print(f"[{api_name}]     - context length: {model_info['context_length']}")
                        if "pricing" in model_info:
                            pricing = model_info["pricing"]
                            print(f"[{api_name}]     - pricing: prompt=${pricing.get('prompt', 'N/A')}/1M, completion=${pricing.get('completion', 'N/A')}/1M")
                        if "top_provider" in model_info:
                            print(f"[{api_name}]     - top provider: {model_info['top_provider'].get('name', 'N/A')}")
                    else:
                        if verbose:
                            print(f"[{api_name}]   Model info query failed: could not fetch info for model '{model_name}'")
        return

    invalid_params = _extract_invalid_params_from_error(error_info, error_text)

    if invalid_params:
        valid_params = [p for p in payload_params if p not in invalid_params]
        print(f"[{api_name}] Payload param validation report:")
        print(f"[{api_name}]   Valid params: {', '.join(valid_params) if valid_params else 'none'}")
        print(f"[{api_name}]   Invalid params: {', '.join(invalid_params)}")

        if verbose:
            print(f"[{api_name}]   Invalid param details:")
            for param in invalid_params:
                if param in payload:
                    value = payload[param]
                    formatted_value = _format_param_value(value)
                    print(f"[{api_name}]     - {param}: {formatted_value}")
    else:
        print(f"[{api_name}] Payload param validation: unable to determine validity (error message has no param info)")
        print(f"[{api_name}]   Parameters used: {', '.join(payload_params)}")
        if error_info:
            print(f"[{api_name}]   Error info: {error_info}")
        elif error_text:
            print(f"[{api_name}]   Error info: {error_text[:200]}...")


# ====================== Call vision API ==============================================

def _try_single_attempt(api_key: str, end_point: str, model_name: str, messages: list,
                        sampling_params: dict, api_name: str = "API", tools: list = None):
    """
    Try a single API attempt (no retries). Used for round-robin polling.

    Returns:
        Tuple of (result, is_429, error_info).
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    payload = _build_payload(model_name, messages, sampling_params, tools)
    _add_reasoning_param(payload, model_name, api_name, end_point)

    response, request_error = _make_api_request(end_point, headers, payload, api_name)

    # Handle 429 rate-limit error first
    if request_error == "429" or (response and response.status_code == 429):
        wait_time = ROUND_ROBIN_429_WAIT_MIN + random.uniform(0, ROUND_ROBIN_429_WAIT_MAX - ROUND_ROBIN_429_WAIT_MIN)
        print(f"[{api_name}] API rate limit hit (429), waiting {wait_time:.2f} seconds before trying other API")
        time.sleep(wait_time)
        return (None, True, f"{api_name}: HTTP 429 (rate limit)")

    if request_error:
        error_msg = f"{api_name}: {request_error}"
        print(f"[{api_name}] Request failed: {request_error}")
        return (None, False, error_msg)

    result, is_429, error_type = _parse_api_response(response, payload, api_name, api_key, end_point, attempt=0, max_attempts=1)

    error_info = None
    if result is None:
        if error_type == "http_error":
            try:
                error_detail = response.json() if response else {}
                error_info = f"{api_name}: HTTP {response.status_code} - {error_detail.get('error', {}).get('message', str(error_detail))}"
            except:
                error_info = f"{api_name}: HTTP {response.status_code}"
        elif error_type:
            error_info = f"{api_name}: {error_type}"

    return (result, is_429, error_info)


def _try_single_api(api_key: str, end_point: str, model_name: str, messages: list,
                    sampling_params: dict, max_retries: int, api_name: str = "API", tools: list = None):
    """
    Internal function to try a single API with full retry logic.

    Returns:
        Model response (string or dict with tool_calls), or None on failure.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    payload = _build_payload(model_name, messages, sampling_params, tools)
    _add_reasoning_param(payload, model_name, api_name, end_point)

    print(f"[{api_name}] API Request attempt 1:")
    print(f"Model: {payload['model']}")
    print(f"Messages count: {len(payload['messages'])}")
    print(f"Temperature: {payload.get('temperature', 'not set')}")
    for i, msg in enumerate(payload['messages'][:3]):
        content_preview = str(msg.get('content', ''))[:100] + '...' if len(str(msg.get('content', ''))) > 100 else str(msg.get('content', ''))
        print(f"  Message {i}: role={msg.get('role')}, content_preview={content_preview}")
    if len(payload['messages']) > 3:
        print(f"  ... and {len(payload['messages']) - 3} more messages")

    for attempt in range(max_retries):
        response, request_error = _make_api_request(end_point, headers, payload, api_name)

        if request_error:
            if attempt < max_retries - 1:
                time.sleep(BASE_WAIT_TIME)
                continue
            return None

        # Handle 429: exponential backoff then retry
        if response.status_code == 429:
            wait_time = BASE_WAIT_TIME * (2 ** attempt) + random.uniform(0, 1)
            wait_time = min(wait_time, MAX_WAIT_TIME)
            print(f"[{api_name}] API rate limit hit on attempt {attempt + 1}/{max_retries}. Waiting {wait_time:.2f} seconds before retrying.")
            time.sleep(wait_time)
            continue

        result, is_429, error_type = _parse_api_response(response, payload, api_name, api_key, end_point,
                                                         attempt=attempt, max_attempts=max_retries)

        if result is not None:
            return result

        # empty_choices means the model refused to generate; retrying won't help
        if error_type == "empty_choices":
            return None

        # Non-retryable HTTP status codes
        if error_type == "http_error" and response is not None and response.status_code in NON_RETRYABLE_STATUS_CODES:
            return None

        if attempt < max_retries - 1:
            time.sleep(BASE_WAIT_TIME)
            continue
        else:
            return None

    return None


def call_vision_api(model_name: str, messages: list, sampling_params: dict, max_retries: int = None, tools: list = None):
    """
    Call the vision API with robust retry logic, including exponential backoff for rate limiting.
    Supports round-robin polling between primary and fallback APIs for faster response.

    Args:
        model_name: Name of the model to use.
        messages: List of message dictionaries.
        sampling_params: Dictionary containing temperature, top_p, max_tokens.
        max_retries: Maximum number of retry attempts (shared between both APIs in round-robin).
        tools: Optional list of tool declarations for function calling.

    Returns:
        Model response (string or dict with tool_calls), or None on failure.
    """
    if max_retries is None:
        try:
            max_retries = max(1, int(os.environ.get("AGENTVISTA_API_MAX_RETRIES", DEFAULT_MAX_RETRIES)))
        except ValueError:
            max_retries = DEFAULT_MAX_RETRIES

    api_key_1 = os.environ.get("REASONING_API_KEY")
    end_point_1 = os.environ.get("REASONING_END_POINT")

    if not all([api_key_1, end_point_1]):
        raise ValueError("REASONING_API_KEY and REASONING_END_POINT must be set.")

    pool_keys = [key.strip() for key in os.environ.get("REASONING_API_KEYS", "").split(",") if key.strip()]
    if not pool_keys:
        pool_keys = [api_key_1]
        api_key_2 = os.environ.get("REASONING_API_KEY_2")
        if api_key_2:
            pool_keys.append(api_key_2)
    # Preserve order while removing accidental duplicate credentials.
    pool_keys = list(dict.fromkeys(pool_keys))
    api_entries = [(key, end_point_1, f"API key {idx + 1}") for idx, key in enumerate(pool_keys)]

    if len(api_entries) > 1:
        print(f"[API Round-Robin] Rotating {len(api_entries)} credentials; {max_retries} retry round(s)")
        consecutive_429_count = 0
        all_errors = []
        api_order = _alternate_api_order(api_entries)

        for attempt in range(max_retries):
            outcomes = []
            for api_key, end_point, api_name in api_order:
                result, is_429, error = _try_single_attempt(api_key, end_point, model_name, messages,
                                                             sampling_params, api_name=api_name, tools=tools)
                outcomes.append(is_429)
                if result is not None:
                    return result
                if error:
                    all_errors.append(f"Round {attempt + 1}: {error}")
            if outcomes and all(outcomes):
                consecutive_429_count += 1
                extra_wait = min(consecutive_429_count * 5, 15)
                print(f"[API Round-Robin] Both APIs rate-limited (429), waiting {extra_wait} seconds before next round")
                if attempt < max_retries - 1:
                    time.sleep(extra_wait)
            else:
                consecutive_429_count = 0
                if attempt < max_retries - 1:
                    time.sleep(1)

        error_summary = "All API attempts failed across the credential pool"
        if all_errors:
            error_summary += f". Errors: {'; '.join(all_errors)}"
        print(f"[API Round-Robin] {error_summary}")
        return f"Error: {error_summary}"
    else:
        print(f"[API] No fallback API configured, using primary API with {max_retries} retries")
        result = _try_single_api(api_key_1, end_point_1, model_name, messages, sampling_params,
                                max_retries, api_name="Primary API", tools=tools)
        if result is not None:
            return result
        # A small number of local vLLM Qwen requests fail before generation
        # because the server-side auto-tool parser cannot parse the model's
        # emitted tool JSON (HTTP 400: unterminated string).  Repeating the
        # identical schema request cannot recover.  One final schema-free
        # request preserves the same prompt/history and lets the model emit a
        # normal final-text answer; callers opt in to accepting it separately.
        if tools:
            print("[API] Tool-schema fallback: retrying once without tool declarations")
            result = _try_single_api(api_key_1, end_point_1, model_name, messages, sampling_params,
                                     1, api_name="Primary API text fallback", tools=None)
            if result is not None:
                return result
        else:
            print(f"[API] Primary API failed after {max_retries} attempts")
        print(f"[API] Primary API failed after {max_retries} attempts")
        return "Error: All API attempts failed (primary API only)"
