# Copyright (C) Izhar Ahmad 2025-2026

from __future__ import annotations

from core.config import Config
from kevin.inference import HuggingFaceInferenceBackend
from kevin import Kevin

config = Config()
assistant = Kevin(
    inference=HuggingFaceInferenceBackend(
        token=config.huggingface_token,
        model=config.inference_model,
        provider=config.inference_provider,
    ),
    stt=config.get_stt(),
    tts=config.get_tts(),
    waker=config.get_waker(),
    system_prompts=config.get_system_prompts(),
    include_default_prompt=config.include_default_system_prompt,
    username=config.username,
)

if __name__ == "__main__":
    assistant.start()
