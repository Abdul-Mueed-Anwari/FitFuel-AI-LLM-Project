from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_LOADED = False


def load_env() -> None:
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except Exception:
        pass


def setting(name: str) -> str | None:
    load_env()
    value = os.getenv(name)
    if value:
        return value
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        if get_script_run_ctx() is None:
            return None
        import streamlit as st

        found = st.secrets.get(name)
        return str(found) if found else None
    except Exception:
        return None


def provider_name() -> str:
    if setting("GROQ_API_KEY"):
        return "Groq"
    if setting("GEMINI_API_KEY"):
        return "Gemini"
    if setting("OPENAI_API_KEY"):
        return "OpenAI"
    return "Local tools"


def llm_configured() -> bool:
    return provider_name() != "Local tools"


def get_chat_model():
    load_env()
    if setting("GROQ_API_KEY"):
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=setting("GROQ_MODEL") or "openai/gpt-oss-120b",
            temperature=0.2,
            api_key=setting("GROQ_API_KEY"),
            timeout=40,
            max_tokens=700,
        )
    if setting("GEMINI_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=setting("GEMINI_MODEL") or "gemini-2.0-flash",
            temperature=0.2,
            api_key=setting("GEMINI_API_KEY"),
            request_timeout=25,
            max_tokens=400,
        )
    if setting("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=setting("OPENAI_MODEL") or "gpt-4o-mini",
            temperature=0.2,
            api_key=setting("OPENAI_API_KEY"),
            timeout=25,
            max_completion_tokens=400,
        )
    return None


last_error = ""


def complete(prompt: str) -> str | None:
    global last_error
    last_error = ""
    model = get_chat_model()
    if model is None:
        last_error = "no-model"
        return None
    try:
        result = model.invoke(prompt)
    except Exception as exc:
        last_error = type(exc).__name__
        return None
    content = getattr(result, "content", "")
    if isinstance(content, list):
        content = " ".join(str(part) for part in content)
    text = str(content).strip()
    return text or None
