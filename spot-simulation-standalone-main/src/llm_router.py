"""Routing logic for Spot NL control language models."""

from __future__ import annotations

import os
import re
import time
import json
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from message_bus import rospy

# Optional dependencies -----------------------------------------------------
try:
    from ollama_client import OllamaClient
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    OllamaClient = None  # type: ignore

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    genai = None  # type: ignore
    types = None  # type: ignore


class LLMRouter:
    """Router to abstract different LLM backends (120B, Gemini Robotics, Gemini Pro)."""

    def __init__(self) -> None:
        self.current_model = "base-120b-low"
        self.default_model = "base-120b-low"
        self._log_requests = os.getenv("LLM_LOG_REQUESTS", "0").strip().lower() in ("1", "true", "yes", "on")
        self._log_lock = threading.Lock()
        self._log_path = Path(__file__).resolve().parents[1] / "logs" / "llm_requests.jsonl"

        self.ollama_120b: Optional[OllamaClient] = None
        self.ollama_20b: Optional[OllamaClient] = None
        self.gemini_client = None
        self.gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.gemini_thinking_budget = int(os.getenv("GEMINI_THINKING_BUDGET", "3000"))

        if OLLAMA_AVAILABLE:
            self._setup_ollama()
        if GENAI_AVAILABLE and self.gemini_api_key:
            self._setup_gemini()
        elif not self.gemini_api_key:
            rospy.logwarn("LLMRouter: GEMINI_API_KEY not set - Gemini models disabled")
        elif not GENAI_AVAILABLE:
            rospy.logwarn("LLMRouter: google-genai not available - Gemini models disabled")

    # Ollama setup ---------------------------------------------------------
    def _setup_ollama(self) -> None:
        try:
            self.ollama_120b = OllamaClient()
            try:
                models = self.ollama_120b.list_models()
                if not models:
                    rospy.logwarn("LLMRouter: Ollama server reachable but returned no models")
                    self.ollama_120b = None
                rospy.loginfo(f"LLMRouter: Ollama 120B backend ready (models: {len(models)})")
            except Exception as probe_exc:  # pragma: no cover - debug logging
                rospy.logerr(f"LLMRouter: Ollama server probe failed: {probe_exc}")
                self.ollama_120b = None

            if self.ollama_120b:
                self.ollama_20b = OllamaClient()
                self.ollama_20b.model = "gpt-oss:20b"
                rospy.loginfo("LLMRouter: Ollama 20B backend ready")
        except Exception as exc:  # pragma: no cover - defensive logging
            rospy.logerr(f"LLMRouter: Failed to setup Ollama: {exc}")

    # Gemini setup ---------------------------------------------------------
    def _setup_gemini(self) -> None:
        try:
            self.gemini_client = genai.Client(api_key=self.gemini_api_key)
            rospy.loginfo("LLMRouter: Gemini backend ready")
        except Exception as exc:  # pragma: no cover - defensive logging
            rospy.logerr(f"LLMRouter: Failed to setup Gemini: {exc}")

    # Routing --------------------------------------------------------------
    def set_model(self, model_id: str) -> None:
        self.current_model = model_id
        rospy.loginfo(f"LLMRouter: Switched to model {model_id}")

    def reset_to_default(self) -> None:
        self.current_model = self.default_model
        rospy.loginfo(f"LLMRouter: Reset to default model {self.default_model}")

    def generate(self, prompt: str, options: Optional[Dict[str, Any]] = None, *, use_20b: bool = False) -> Optional[Dict[str, Any]]:
        options = dict(options or {})
        uses_thinking = self._model_uses_thinking_budget(self.current_model)
        if uses_thinking:
            options.setdefault("thinking_budget", self.gemini_thinking_budget)
        else:
            options.pop("thinking_budget", None)
        
        rospy.loginfo(f"DEBUG: LLMRouter.generate called with current_model={self.current_model}")
        self._log_request_payload(backend="router", model=self.current_model, prompt=prompt, options=options)

        if use_20b and self.ollama_20b:
            try:
                result = self.ollama_20b.generate(prompt=prompt, options=options)
                self._log_response_payload(backend="ollama_20b", model=self.current_model, response=result)
                return result
            except Exception as exc:  # pragma: no cover - defensive logging
                rospy.logerr(f"LLMRouter: 20B generation failed: {exc}")
                return None

        if self.current_model.startswith("base-120b"):
            rospy.logwarn(f"DEBUG: Using Ollama path for model {self.current_model}")
            result = self._generate_ollama_120b(prompt, options)
            self._log_response_payload(backend="ollama_120b", model=self.current_model, response=result)
            return result

        # Gemini variants -------------------------------------------------
        if self.current_model == "gemini-robotics-er-1.5-preview":
            result = self._generate_gemini(prompt, options, fallback_model=None, use_thinking_budget=False)
            self._log_response_payload(backend="gemini", model=self.current_model, response=result)
            return result

        if self.current_model == "gemini-2.5-pro":
            result = self._generate_gemini(prompt, options, fallback_model=None, use_thinking_budget=True)
            self._log_response_payload(backend="gemini", model=self.current_model, response=result)
            return result

        if self.current_model in ("gemini-2.5-flash", "gemini-2.5-flash-lite"):
            result = self._generate_gemini(prompt, options, fallback_model=None, use_thinking_budget=True)
            self._log_response_payload(backend="gemini", model=self.current_model, response=result)
            return result

        if self.current_model == "gemini-3-flash-preview":
            result = self._generate_gemini(prompt, options, fallback_model=None, use_thinking_budget=True)
            self._log_response_payload(backend="gemini", model=self.current_model, response=result)
            return result

        if self.current_model in ("gemini-2.0-flash", "gemini-2.0-flash-lite"):
            result = self._generate_gemini(prompt, options, fallback_model=None, use_thinking_budget=False)
            self._log_response_payload(backend="gemini", model=self.current_model, response=result)
            return result

        # if self.current_model in ("gemini-1.5-pro", "gemini-1.5-flash"):
        #     return self._generate_gemini(prompt, options, fallback_model="base-120b-low", use_thinking_budget=False)

        rospy.logerr(f"LLMRouter: Unknown model {self.current_model}")
        return None

    def _model_uses_thinking_budget(self, model_id: str) -> bool:
        return model_id in {
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-3-flash-preview",
        }

    def _generate_ollama_120b(self, prompt: str, options: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.ollama_120b:
            rospy.logwarn("LLMRouter: Ollama 120B not available")
            return None

        try:
            reasoning_level = "low"
            if self.current_model == "base-120b-medium":
                reasoning_level = "medium"
            elif self.current_model == "base-120b-high":
                reasoning_level = "high"

            enhanced_prompt = f"Reasoning: {reasoning_level}\n\n{prompt}"
            result = self.ollama_120b.generate(prompt=enhanced_prompt, options=options)
            if result is None:
                rospy.logwarn("LLMRouter: Ollama 120B returned no result")
            return result
        except Exception as exc:  # pragma: no cover - defensive logging
            rospy.logerr(f"LLMRouter: 120B generation failed: {exc}")
            return None

    def _generate_gemini(self, prompt: str, options: Dict[str, Any], *, fallback_model: Optional[str], use_thinking_budget: bool) -> Optional[Dict[str, Any]]:
        if not self.gemini_client:
            rospy.logerr("LLMRouter: Gemini client not available - cannot generate plan")
            rospy.logerr("LLMRouter: Please ensure GEMINI_API_KEY or GOOGLE_API_KEY is set and google-genai is installed")
            return None

        max_attempts = 2
        for attempt in range(max_attempts):
            try:
                config_kwargs: Dict[str, Any] = {"temperature": options.get("temperature", 0.0)}
                thinking_budget = options.get("thinking_budget")
                try:
                    thinking_budget_int = int(thinking_budget) if thinking_budget is not None else self.gemini_thinking_budget
                except (TypeError, ValueError):
                    thinking_budget_int = self.gemini_thinking_budget

                if use_thinking_budget and thinking_budget_int:
                    config_kwargs["thinking_config"] = types.ThinkingConfig(
                        thinking_budget=thinking_budget_int,
                        include_thoughts=True,
                    )

                response = self.gemini_client.models.generate_content(
                    model=self.current_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(**config_kwargs),
                )
                usage = self._usage_metadata_to_dict(getattr(response, "usage_metadata", None))
                thinking = self._extract_thinking_summary(response)
                return {
                    "response": response.text,
                    "usage_metadata": usage,
                    "thinking": thinking,
                }
            except Exception as exc:  # pragma: no cover - defensive logging
                is_quota = "RESOURCE_EXHAUSTED" in str(exc).upper()
                delay = _extract_retry_delay_seconds(exc)
                if is_quota and delay and attempt < max_attempts - 1:
                    rospy.logwarn(f"LLMRouter: Gemini quota hit; retrying in {delay:.1f}s")
                    time.sleep(delay)
                    continue

                rospy.logerr(f"LLMRouter: Gemini generation failed: {exc}")
                rospy.logerr("LLMRouter: Gemini is the selected model - not falling back to Ollama")
                import traceback
                rospy.logerr(f"LLMRouter: Error traceback: {traceback.format_exc()}")
                return None

        return None

    def is_gemini_available(self) -> bool:
        return GENAI_AVAILABLE and self.gemini_client is not None

    @staticmethod
    def _usage_metadata_to_dict(usage: Any) -> Optional[Dict[str, Any]]:
        """Convert Gemini usage metadata to a JSON-serializable dict."""
        if usage is None:
            return None

        def _pull(key: str) -> Any:
            if isinstance(usage, dict):
                return usage.get(key)
            return getattr(usage, key, None)

        prompt_tokens = _pull("prompt_token_count")
        response_tokens = _pull("candidates_token_count")
        total_tokens = _pull("total_token_count")
        thinking_tokens = _pull("thinking_token_count") or _pull("thoughts_token_count")

        usage_dict: Dict[str, Any] = {}
        if prompt_tokens is not None:
            usage_dict["prompt_token_count"] = prompt_tokens
        if response_tokens is not None:
            usage_dict["candidates_token_count"] = response_tokens
        if total_tokens is not None:
            usage_dict["total_token_count"] = total_tokens
        if thinking_tokens is not None:
            usage_dict["thinking_token_count"] = thinking_tokens

        return usage_dict or None

    @staticmethod
    def _extract_thinking_summary(response: Any) -> Optional[list[str]]:
        """Pull Gemini thinking summaries (if returned) into a simple list of strings."""
        try:
            candidates = getattr(response, "candidates", None)
            if not candidates:
                return None

            content = getattr(candidates[0], "content", None)
            parts = getattr(content, "parts", None) if content else None
            if not parts:
                return None

            thoughts: list[str] = []
            for part in parts:
                if getattr(part, "thought", False) and getattr(part, "text", None):
                    thoughts.append(part.text)

            return thoughts or None
        except Exception:
            return None

    def _log_request_payload(self, *, backend: str, model: str, prompt: str, options: Dict[str, Any]) -> None:
        if not self._log_requests:
            return
        try:
            payload = {
                "timestamp": time.time(),
                "type": "request",
                "backend": backend,
                "model": model,
                "options": options,
                "prompt": prompt,
            }
            log_path = self._log_path
            log_path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(payload, ensure_ascii=False)
            with self._log_lock:
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
        except Exception as exc:
            rospy.logwarn(f"LLMRouter: Failed to log LLM request: {exc}")

    def _log_response_payload(self, *, backend: str, model: str, response: Any) -> None:
        if not self._log_requests:
            return
        try:
            payload = {
                "timestamp": time.time(),
                "type": "response",
                "backend": backend,
                "model": model,
                "response": response,
            }
            log_path = self._log_path
            log_path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(payload, ensure_ascii=False, default=str)
            with self._log_lock:
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
        except Exception as exc:
            rospy.logwarn(f"LLMRouter: Failed to log LLM response: {exc}")


def _extract_retry_delay_seconds(exc: Exception) -> Optional[float]:
    """Best-effort extraction of retry delay seconds from quota errors."""
    text = str(exc)
    match = re.search(r"retry(?: in)?\s*([0-9]+(?:\.[0-9]+)?)s", text, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None

    match = re.search(r"retryDelay[\"']?\s*:\s*'?(\d+)s", text, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None

    return None
