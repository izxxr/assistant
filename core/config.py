# Copyright (C) Izhar Ahmad 2025-2026

from __future__ import annotations

from typing import Any

import kevin
import os
import dotenv

__all__ = (
    "Config",
)


_default: Any = object()


class Config:
    """Configuration for assistant from environment variables."""

    def __init__(self) -> None:
        self.load()

    def _get(
        self,
        key: str,
        default: Any = _default,
        *,
        boolean: bool = False,
        integer: bool = False,
        number: bool = False,
    ) -> Any:
        try:
            value = os.environ[key]
        except KeyError:
            if default is not _default:
                return default
            raise
        else:
            if boolean:
                return value.lower() in ("true", "1")
            if integer:
                return int(value.lower())
            if number:
                return float(value.lower())

            return value

    def load(self) -> None:
        """Loads the config from .env file."""
        dotenv.load_dotenv()

        # Assistant config
        self.username = self._get("KEVIN_USERNAME", "Izhar")
        self.system_prompt_path = self._get("KEVIN_SYSTEM_PROMPT_PATH", None)
        self.include_default_system_prompt = self._get("KEVIN_INCLUDE_DEFAULT_SYSTEM_PROMPT",
                                                       self.system_prompt_path is None)

        # Inference
        self.huggingface_token = self._get("KEVIN_HUGGINGFACE_TOKEN")
        self.inference_model = self._get("KEVIN_INFERENCE_MODEL", "Qwen3-4B-Instruct-2507")
        self.inference_provider = self._get("KEVIN_INFERENCE_PROVIDER", "nscale")

        # Waker
        self.waker = self._get("KEVIN_WAKER", "hotkey").lower()
        self.wake_hotkey = self._get("KEVIN_WAKE_HOTKEY", "alt+shift+/").lower()
        self.porcupine_access_key = self._get("KEVIN_PORCUPINE_ACCESS_KEY", None)
        self.porcupine_keyword_path = self._get("KEVIN_PORCUPINE_KEYWORD_PATH", None)

        # STT
        self.text_input = self._get("KEVIN_TEXT_INPUT", False, boolean=True)
        self.faster_whisper_model = self._get("KEVIN_FASTER_WHISPER_MODEL", "tiny.en").lower()

        # TTS
        self.text_output = self._get("KEVIN_TEXT_OUTPUT", False, boolean=True)
        self.piper_voice_path = self._get("KEVIN_PIPER_VOICE_PATH", None)

        self.validate()

    def validate(self) -> None:
        """Validates the loaded configuration."""

        assert self.waker.lower() in ("hotkey", "porcupine"), "KEVIN_WAKER must be either 'hotkey' or 'porcupine'"

        if self.waker == "porcupine":
            assert self.porcupine_access_key is not None, "KEVIN_PORCUPINE_ACCESS_KEY must be provided when KEVIN_WAKER=porcupine"
            assert self.porcupine_keyword_path is not None, "KEVIN_PORCUPINE_KEYWORD_PATH must be provided when KEVIN_WAKER=porcupine"

        if not self.text_output:
            assert self.piper_voice_path is not None, "KEVIN_PIPER_VOICE_PATH must be provided when KEVIN_TEXT_OUTPUT=true"

    def get_stt(self) -> kevin.stt.STTProvider | None:
        """Get STT provider instance based on given configuration."""
        if self.text_input:
            return
        
        return kevin.stt.FasterWhisperSTT(
            model_size_or_path=self.faster_whisper_model,
            whisper_model_options={"compute_type": "int8"}
        )

    def get_tts(self) -> kevin.tts.TTSProvider | None:
        """Get TTS provider instance based on given configuration."""
        if self.text_output:
            return

        return kevin.tts.PiperTTS(voice_path=self.piper_voice_path)

    def get_waker(self) -> kevin.waker.Waker | None:
        """Get Waker instance based on given configuration."""
        if self.waker == "hotkey":
            return kevin.waker.HotkeyWaker(self.wake_hotkey)
        if self.waker == "porcupine":
            return kevin.waker.PorcupineWaker(
                access_key=self.porcupine_access_key,
                keyword_paths=[self.porcupine_keyword_path]
            )
    
    def get_system_prompts(self) -> list[str] | None:
        """Get system prompts based on given configuration."""
        if not self.system_prompt_path:
            return

        with open(self.system_prompt_path, "r") as f:
            return [f.read()]
