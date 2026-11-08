from __future__ import annotations

import html
from datetime import datetime

import streamlit as st

from fitfuel.intent import medical_reason
from fitfuel.llm import llm_configured, load_env
from fitfuel.pipeline import run_turn
from fitfuel.profiles import EQUIPMENT_LABELS, GOAL_LABELS, clean_profile
from fitfuel import store

load_env()
store.init_db()

st.set_page_config(page_title="FitFuel", page_icon="🏋️", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,560;9..144,650&family=Outfit:wght@380;460;560;640&display=swap');
    html, body, [class*="css"] { font-family: Outfit, sans-serif; }
    h1, h2, h3, .ff-brand, .ff-side-brand { font-family: Fraunces, Georgia, serif !important; letter-spacing: -0.03em; }
    .block-container { padding-top: 0.7rem; max-width: 1360px; padding-bottom: 370px; }
    section[data-testid="stSidebar"] { padding-bottom: 0 !important; box-sizing: border-box; }
    [data-testid="stSidebarUserContent"] { padding-top: 0.7rem !important; padding-bottom: 0.55rem !important; }
    #MainMenu, [data-testid="stDecoration"], [data-testid="stMainMenu"], [data-testid="stStatusWidget"],
    [data-testid="stToolbarActions"], footer:not(.ff-footer) { display: none !important; }
    header[data-testid="stHeader"] {
      background: transparent !important; height: 0 !important; min-height: 0 !important;
      overflow: visible !important; pointer-events: none !important; z-index: 1000008 !important;
    }
    header[data-testid="stHeader"] [data-testid="stToolbar"] {
      background: transparent !important; height: 0 !important; min-height: 0 !important;
      overflow: visible !important; pointer-events: none !important;
    }
    header[data-testid="stHeader"] button:not([data-testid="stExpandSidebarButton"]) { display: none !important; }
    [data-testid="stExpandSidebarButton"] {
      pointer-events: auto !important; display: flex !important; position: fixed !important;
      top: 0.85rem !important; left: 0.7rem !important; z-index: 1000009 !important;
      background: #121a15 !important; border: 1px solid #d6ff6b !important; border-radius: 999px !important;
      width: 2.35rem !important; height: 2.15rem !important; min-height: 2.15rem !important;
    }
    [data-testid="stExpandSidebarButton"],
    [data-testid="stExpandSidebarButton"] span { color: #d6ff6b !important; }
    @media (min-width: 901px) {
      section[data-testid="stSidebar"] {
        top: 0 !important; height: calc(100vh - 320px) !important; max-height: calc(100vh - 320px) !important;
        overflow: hidden !important;
      }
      .stApp:not(:has(.st-key-coach_dock)) section[data-testid="stSidebar"] {
        height: 100vh !important; max-height: 100vh !important;
      }
      section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.5rem !important; }
      [data-testid="stSidebarContent"] {
        height: 100% !important; max-height: 100% !important; overflow-x: hidden !important; overflow-y: auto !important;
      }
    }
    [data-testid="stSidebar"], [data-testid="stSidebarContent"] { background: #121a15 !important; }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] small { color: #e7f0e6 !important; }
    [data-testid="stSidebar"] hr { border-color: #2a3a30; }
    [data-testid="stSidebarCollapseButton"] svg { fill: #e7f0e6; }
    [data-testid="stSidebar"] .ff-side-brand { color: #d6ff6b !important; font-size: 1.7rem; margin: 0.2rem 0 0; }
    [data-testid="stElementContainer"]:has(.ff-banner) {
      position: sticky; top: 0; z-index: 40; background: #f3efe6; padding: 0.15rem 0 0.55rem;
    }
    .ff-banner { display: flex; justify-content: space-between; align-items: center; gap: 1rem;
      background: #121a15; color: #f6f3ec; border-radius: 18px; padding: 0.85rem 1.2rem;
      box-shadow: 0 12px 30px rgba(18, 26, 21, 0.18); }
    .ff-banner .ff-brand { margin: 0; }
    .ff-banner strong { display: block; font-family: Fraunces, Georgia, serif; font-size: 1.55rem; line-height: 1.05; color: #f6f3ec; }
    .ff-banner p { margin: 0.15rem 0 0; color: #d6ff6b; max-width: 22rem; text-align: right; }
    .ff-brand { color: #d6ff6b; font-size: 0.95rem; margin: 0 0 0.35rem; }
    .ff-stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.7rem; margin: 0 0 1.1rem; }
    .ff-stats div { background: #fffdf8; border: 1px solid #e5dfd0; border-radius: 16px; padding: 0.75rem 0.9rem; }
    .ff-stats span { display: block; color: #5c6a5b; font-size: 0.82rem; }
    .ff-stats strong { font-family: Fraunces, Georgia, serif; font-size: 1.7rem; color: #142117; letter-spacing: -0.03em; }
    .ff-stats em { color: #5c6a5b; font-style: normal; margin-left: 0.25rem; font-size: 0.85rem; }
    .ff-note { color: #5d6b5c; font-size: 0.86rem; margin: 0.15rem 0 1rem; }
    .ff-points { display: grid; gap: 0.7rem; margin-top: 0.4rem; }
    .ff-point { background: #fffdf8; border: 1px solid #e4dece; border-radius: 16px; padding: 0.9rem 1rem; }
    .ff-point strong { display: block; color: #142117; margin-bottom: 0.15rem; }
    .ff-point span { color: #526052; font-size: 0.92rem; }
    .ff-section { display: flex; align-items: baseline; gap: 0.7rem; margin: 0.35rem 0 0.7rem; }
    .ff-section span { color: #1b5c3a; font-weight: 640; font-size: 0.78rem; letter-spacing: 0.14em; }
    .ff-section h2 { margin: 0; font-size: 1.7rem; color: #142117; }
    .ff-day, .ff-meal { background: #fffdf8; border: 1px solid #e5dfd0; border-radius: 18px; padding: 0.95rem 1rem 0.8rem; height: 100%; }
    .ff-day-top { display: flex; justify-content: space-between; align-items: baseline; gap: 0.5rem; margin-bottom: 0.7rem; }
    .ff-day-top em { color: #6a7568; font-style: normal; font-size: 0.82rem; }
    .ff-kicker { color: #1b5c3a; font-size: 0.75rem; font-weight: 640; letter-spacing: 0.12em; text-transform: uppercase; }
    .ff-day h3, .ff-meal h3 { margin: 0.1rem 0 0; font-size: 1.15rem; color: #142117; line-height: 1.25; overflow-wrap: normal; word-break: normal; }
    .ff-day ul { list-style: none; margin: 0; padding: 0; }
    .ff-day li { padding: 0.55rem 0; border-top: 1px solid #efeadd; }
    .ff-day li b { display: block; font-weight: 560; overflow-wrap: normal; word-break: normal; }
    .ff-day li span, .ff-meal .ff-sub { color: #5c6a5b; font-size: 0.88rem; }
    .ff-day li small { display: block; color: #7b8578; margin-top: 0.12rem; }
    .ff-kcal { margin: 0.55rem 0 0.15rem; }
    .ff-kcal strong { font-family: Fraunces, Georgia, serif; font-size: 1.85rem; letter-spacing: -0.03em; color: #142117; }
    .ff-kcal span { color: #5c6a5b; margin-left: 0.2rem; }
    .ff-ings { color: #6d786c; font-size: 0.84rem; margin: 0.45rem 0 0; }
    .st-key-coach_dock {
      position: fixed !important; left: 0 !important; right: 0 !important; bottom: 0 !important;
      width: 100vw !important; max-width: 100vw !important; z-index: 1000002 !important;
      background: #121a15 !important; border: none !important; border-top: 2px solid #d6ff6b !important;
      border-radius: 0 !important; box-shadow: 0 -16px 40px rgba(18, 26, 21, 0.35);
      height: 320px; max-height: 320px; overflow: hidden; margin: 0 !important;
      padding: 0.7rem 1.2rem 0.75rem !important;
    }
    .st-key-coach_dock [data-testid="stHorizontalBlock"] { align-items: stretch !important; height: 100%; }
    .st-key-coach_dock [data-testid="stColumn"]:has(.ff-coach-mark) {
      height: 276px; max-height: 276px; overflow-x: hidden; overflow-y: auto;
    }
    .st-key-coach_dock [data-testid="stCaptionContainer"] p { color: #c5d4c4 !important; font-size: 0.88rem !important; margin: 0.15rem 0 0.35rem !important; }
    .ff-coach-title { color: #f6f3ec !important; font-family: Fraunces, Georgia, serif !important; font-size: 1.25rem !important;
      line-height: 1.15; margin: 0.15rem 0 0 !important; letter-spacing: -0.03em; }
    .ff-thread { height: 276px; max-height: 276px; overflow-x: hidden; overflow-y: scroll; scrollbar-gutter: stable; }
    * { scrollbar-width: thin; scrollbar-color: #1b5c3a #efeadd; }
    *::-webkit-scrollbar { width: 10px; height: 10px; }
    *::-webkit-scrollbar-thumb { background: #1b5c3a; border-radius: 8px; }
    *::-webkit-scrollbar-track { background: #efeadd; }
    .ff-thread, .st-key-coach_dock, [data-testid="stSidebar"], [data-testid="stSidebarContent"],
    .st-key-coach_dock [data-testid="stColumn"]:has(.ff-coach-mark) {
      scrollbar-color: #d6ff6b #243328;
    }
    .ff-thread::-webkit-scrollbar-thumb,
    .st-key-coach_dock::-webkit-scrollbar-thumb,
    [data-testid="stSidebar"]::-webkit-scrollbar-thumb,
    [data-testid="stSidebarContent"]::-webkit-scrollbar-thumb,
    .st-key-coach_dock [data-testid="stColumn"]:has(.ff-coach-mark)::-webkit-scrollbar-thumb { background: #d6ff6b; }
    .ff-thread::-webkit-scrollbar-track,
    .st-key-coach_dock::-webkit-scrollbar-track,
    [data-testid="stSidebar"]::-webkit-scrollbar-track,
    [data-testid="stSidebarContent"]::-webkit-scrollbar-track,
    .st-key-coach_dock [data-testid="stColumn"]:has(.ff-coach-mark)::-webkit-scrollbar-track { background: #243328; }
    .ff-thread-inner { display: flex; flex-direction: column; gap: 0.45rem; }
    .st-key-coach_dock [data-testid="InputInstructions"] {
      color: transparent !important; font-size: 0 !important; line-height: 1.2rem !important;
      height: 1.2rem; position: relative; left: 0 !important; overflow: visible;
    }
    .st-key-coach_dock [data-testid="InputInstructions"]::after {
      content: "Press Ctrl+Enter to send query";
      color: #c5d4c4; font-size: 0.78rem; line-height: 1.2rem;
      position: absolute; left: 0; top: 0; white-space: nowrap;
    }
    .ff-check { background: #1c2820; border: 1px solid #d6ff6b; border-radius: 12px;
      padding: 0.45rem 0.7rem 0.55rem; margin: 0.45rem 0 0; }
    .ff-check summary { color: #d6ff6b; font-weight: 640; cursor: pointer; }
    .ff-check p { color: #f6f3ec; margin: 0.4rem 0 0; font-size: 0.88rem; line-height: 1.4; }
    .ff-check b { color: #d6ff6b; font-weight: 640; }
    .ff-thread-empty { margin: 0; color: #c5d4c4; }
    .ff-msg { margin: 0; background: #1c2820; border-radius: 14px; padding: 0.65rem 0.8rem; color: #f6f3ec; max-width: 88%; }
    .ff-you { align-self: flex-end; background: #243328; }
    .ff-coach { align-self: flex-start; }
    .ff-msg strong { display: block; color: #d6ff6b; font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 0.2rem; }
    [data-testid="stElementContainer"]:has(.ff-nav-anchor),
    [data-testid="stElementContainer"]:has(.ff-nav-anchor) + [data-testid="stElementContainer"] { display: none; }
    .ff-footer { margin-top: 2.2rem; background: #121a15; color: #e7f0e6; border-radius: 22px; padding: 1.35rem 1.4rem 1.2rem; }
    .ff-footer .ff-kicker { color: #d6ff6b; }
    .ff-footer h2 { color: #f6f3ec; font-size: 1.45rem; margin: 0.15rem 0 0.35rem; }
    .ff-footer p { color: #c9d5c8; margin: 0.35rem 0 0; max-width: 52rem; }
    .ff-footer ul { margin: 0.7rem 0 0; padding: 0; list-style: none; display: grid; gap: 0.35rem; }
    .ff-footer li { color: #e7f0e6; }
    .ff-footer li span { color: #d6ff6b; font-weight: 560; }
    [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {
      background: #243328 !important; color: #f4f7f2 !important; border: 1px solid #3d5646 !important;
    }
    @media (max-width: 900px) {
      [data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; }
      [data-testid="stColumn"] { width: 100% !important; flex: 1 1 100% !important; min-width: 100% !important; }
      .ff-stats { grid-template-columns: 1fr 1fr; }
      .ff-banner { display: block; }
      .ff-banner p { text-align: left; margin-top: 0.35rem; }
      .st-key-coach_dock {
        height: 30vh; max-height: 30vh; overflow: hidden;
        padding: 0.4rem 0.55rem 0.35rem !important;
      }
      .st-key-coach_dock [data-testid="stHorizontalBlock"] { height: auto !important; }
      .st-key-coach_dock [data-testid="stHorizontalBlock"]:has(.ff-thread) {
        flex-direction: column !important; flex-wrap: nowrap !important;
        height: auto !important; align-items: stretch !important; gap: 0.25rem !important;
      }
      .st-key-coach_dock [data-testid="stHorizontalBlock"]:has(.ff-thread) > [data-testid="stColumn"] {
        width: 100% !important; min-width: 0 !important; flex: 0 0 auto !important;
      }
      .st-key-coach_dock [data-testid="stHorizontalBlock"]:has(.ff-thread) > [data-testid="stColumn"]:first-child {
        overflow: hidden !important;
      }
      .st-key-coach_dock [data-testid="stHorizontalBlock"]:has(.ff-thread) > [data-testid="stColumn"]:last-child {
        background: #121a15 !important; position: relative !important; z-index: 2 !important;
        padding-top: 0.85rem !important;
      }
      .ff-thread { height: calc(30vh - 6.8rem) !important; max-height: calc(30vh - 6.8rem) !important; min-height: 0 !important; }
      .ff-coach-title { font-size: 0.95rem !important; white-space: nowrap; margin: 0 !important; }
      .st-key-coach_dock .ff-coach-label { display: block !important; margin: 0 !important; line-height: 1.1 !important; }
      .st-key-coach_dock [data-testid="stElementContainer"]:has(.ff-coach-mark) {
        min-height: 2.2rem !important; margin: 0 0 0.2rem !important;
      }
      .st-key-coach_dock [data-testid="stColumn"]:has(.ff-coach-mark) [data-testid="stVerticalBlock"] { gap: 0.2rem !important; }
      .st-key-coach_dock [data-testid="stCaptionContainer"] { display: none !important; }
      .st-key-coach_dock [data-testid="stElementContainer"]:has([data-testid="stCaptionContainer"]) { display: none !important; }
      .st-key-coach_dock [data-testid="stForm"] { padding: 0 !important; border: none !important; margin: 0 !important; }
      .st-key-coach_dock .ff-check,
      .st-key-coach_dock [data-testid="stElementContainer"]:has(.ff-check) { display: none !important; }
      .st-key-coach_dock [data-testid="stColumn"]:has(.ff-coach-mark) {
        height: auto !important; max-height: none !important; overflow: visible !important;
      }
      .st-key-coach_dock [data-testid="stHorizontalBlock"]:not(:has(.ff-thread)):not(:has(textarea)) {
        display: none !important;
      }
      .st-key-coach_dock textarea { height: 40px !important; min-height: 40px !important; }
      .st-key-coach_dock [data-testid="InputInstructions"] { display: none !important; }
      .st-key-coach_dock details { margin: 0 !important; }
      .st-key-coach_dock [data-testid="stForm"] [data-testid="stHorizontalBlock"] { flex-wrap: nowrap !important; }
      .st-key-coach_dock [data-testid="stForm"] [data-testid="stColumn"]:first-child {
        width: auto !important; min-width: 0 !important; flex: 1 1 auto !important;
      }
      .st-key-coach_dock [data-testid="stForm"] [data-testid="stColumn"]:last-child {
        width: 4.6rem !important; min-width: 4.6rem !important; max-width: 4.6rem !important; flex: 0 0 4.6rem !important;
      }
      .block-container { padding-bottom: 1.5rem; }
      .stApp:has(.st-key-coach_dock) .block-container { padding-bottom: 32vh; }
      section[data-testid="stSidebar"],
      [data-testid="stSidebarContent"],
      [data-testid="stSidebarUserContent"] {
        overflow-x: hidden !important;
      }
      section[data-testid="stSidebar"]::-webkit-scrollbar,
      [data-testid="stSidebarContent"]::-webkit-scrollbar,
      [data-testid="stSidebarUserContent"]::-webkit-scrollbar { height: 0 !important; }
      section[data-testid="stSidebar"] { padding-bottom: 1.25rem; }
      [data-testid="stElementContainer"]:has(.ff-nav-anchor) + [data-testid="stElementContainer"] { display: block; width: 2.5rem; margin: 0.2rem 0 0.15rem; }
      [data-testid="stElementContainer"]:has(.ff-nav-anchor) + [data-testid="stElementContainer"] button {
        width: 2.35rem; min-height: 2.15rem; padding: 0; border-radius: 999px;
        background: #121a15; color: #d6ff6b; border: 1px solid #d6ff6b;
      }
      [data-testid="stSidebarCollapseButton"],
      [data-testid="stExpandSidebarButton"] { display: none !important; pointer-events: none !important; }
    }
    .ff-coach-label { color: #d6ff6b; font-size: 0.78rem; font-weight: 640; letter-spacing: 0.14em; text-transform: uppercase; margin: 0; }
    div[data-testid="stMetric"] { background: #fffdf8; border: 1px solid #e5dfd0; border-radius: 16px; padding: 0.7rem 0.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

GOALS = {
    "Lose fat": "fat_loss",
    "Build muscle": "muscle_gain",
    "Body recomposition": "maintain",
}
CUISINES = {
    "No specific cuisine": "none",
    "Pakistani": "pakistani",
    "Chinese": "chinese",
    "Indian": "indian",
    "Mediterranean": "mediterranean",
    "Mexican": "mexican",
    "Italian": "italian",
}
EQUIPMENT = {
    "Bodyweight only": "bodyweight",
    "Dumbbells": "dumbbells",
    "Resistance bands": "bands",
    "Home gym": "home",
    "Full gym": "gym",
}
DIETS = {
    "No specific diet": "none",
    "Vegetarian": "vegetarian",
    "Vegan": "vegan",
    "Pescatarian": "pescatarian",
}
SEXES = {"Female": "female", "Male": "male", "Prefer not to say": "other"}
SUGGESTIONS = [
    ("Lighter dinners", "Make dinners under 500 calories"),
    ("30 minutes", "What if I only have 30 minutes?"),
    ("Vegetarian", "Make my meals vegetarian."),
    ("Swap squats", "Can I swap squats?"),
]
SETUP_MESSAGES = {"Create my weekly plan", "Build another week for me"}


def _label(mapping: dict, value: str) -> str:
    for label, internal in mapping.items():
        if internal == value:
            return label
    return next(iter(mapping))


def _init_state() -> None:
    st.session_state.setdefault("user_id", None)
    st.session_state.setdefault("form_nonce", 0)
    st.session_state.setdefault("prefill", {})
    st.session_state.setdefault("use_llm", llm_configured())
    st.session_state.setdefault("show_side", False)
    st.session_state.setdefault("pending_user", None)
    st.session_state.setdefault("answering", False)


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def _profile_line(profile: dict) -> str:
    equipment = EQUIPMENT_LABELS.get(profile["equipment"], profile["equipment"]).removeprefix("a ").title()
    bits = [
        GOAL_LABELS.get(profile["goal"], profile["goal"]).title(),
        f"{profile['days_per_week']} days",
        f"{profile['session_minutes']} min",
        equipment,
    ]
    if profile.get("diet") not in (None, "", "none"):
        bits.append(str(profile["diet"]).title())
    if profile.get("cuisine") not in (None, "", "none"):
        bits.append(str(profile["cuisine"]).title())
    avoids = [str(item) for item in profile.get("avoid") or []]
    if avoids:
        bits.append("no " + ", ".join(avoids))
    return " · ".join(bits)


def _banner(title: str, meta: str = "") -> None:
    meta_html = f"<p>{_esc(meta)}</p>" if meta else ""
    st.markdown(
        f'<div class="ff-banner"><div><p class="ff-brand">FitFuel</p><strong>{_esc(title)}</strong></div>{meta_html}</div>',
        unsafe_allow_html=True,
    )
    _nav_arrow()
    st.markdown(
        '<p class="ff-note">General training and meal ideas for healthy adults. Not medical advice. '
        "Pain, pregnancy, and medical conditions belong with a qualified professional.</p>",
        unsafe_allow_html=True,
    )


def _nav_arrow() -> None:
    st.markdown('<span class="ff-nav-anchor"></span>', unsafe_allow_html=True)
    arrow = "‹" if st.session_state.get("show_side") else "›"
    if st.button(arrow, key="nav-arrow", help="Show or hide the side panel"):
        st.session_state.show_side = not st.session_state.get("show_side")
        st.rerun()


def _side_style() -> None:
    if st.session_state.get("show_side"):
        css = """
        @media (max-width: 900px) {
          section[data-testid="stSidebar"] {
            display: block !important; visibility: visible !important;
            position: fixed !important; z-index: 1000006 !important;
            top: 0 !important; left: 0 !important; transform: none !important;
            height: 100vh !important; max-height: 100vh !important;
            height: 100dvh !important; max-height: 100dvh !important;
            overflow-x: hidden !important; overflow-y: auto !important;
            width: min(18rem, 86vw) !important;
            min-width: min(18rem, 86vw) !important;
            max-width: min(18rem, 86vw) !important;
            padding-bottom: 1.25rem !important;
            box-shadow: 12px 0 30px rgba(0,0,0,.28);
          }
          [data-testid="stElementContainer"]:has(.ff-nav-anchor) + [data-testid="stElementContainer"] {
            position: fixed !important; z-index: 1000007 !important;
            top: 0.7rem !important; left: 0.7rem !important; right: auto !important;
            width: 2.5rem !important; margin: 0 !important;
          }
          [data-testid="stSidebarUserContent"] { padding-top: 3.1rem !important; }
        }
        """
    else:
        css = """
        @media (max-width: 900px) {
          [data-testid="stSidebar"] { display: none !important; }
        }
        """
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def _when(stamp: str) -> str:
    if not stamp:
        return "Saved"
    try:
        parsed = datetime.fromisoformat(stamp)
    except ValueError:
        return stamp[:10]
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone()
    hour = parsed.strftime("%I").lstrip("0") or "12"
    return f"{parsed.strftime('%b')} {parsed.day}, {hour}:{parsed.strftime('%M %p').lower()}"


def _plan_labels(plans: list[dict]) -> dict[str, str]:
    total = len(plans)
    labels = {}
    for index, item in enumerate(plans):
        week = total - index
        labels[item["id"]] = f"Week {week} · {_when(item.get('created_at') or '')}"
    return labels


def _visible_messages(messages: list[dict]) -> list[dict]:
    visible = []
    skip_reply = False
    for message in messages:
        if message["role"] == "user" and message["content"] in SETUP_MESSAGES:
            skip_reply = True
            continue
        if skip_reply and message["role"] == "assistant":
            skip_reply = False
            continue
        visible.append(message)
    return visible


def _section(number: str, title: str) -> None:
    st.markdown(
        f'<div class="ff-section"><span>{_esc(number)}</span><h2>{_esc(title)}</h2></div>',
        unsafe_allow_html=True,
    )


def _trace(trace: list[dict]) -> None:
    if not trace:
        return
    rows = []
    for step in trace:
        rows.append(
            "<p><b>"
            + _esc(step.get("agent", "Coach"))
            + "</b> · "
            + _esc(step.get("tool", ""))
            + " · "
            + _esc(step.get("detail", ""))
            + "</p>"
        )
    st.markdown(
        '<details class="ff-check"><summary>How this was checked</summary>'
        + "".join(rows)
        + "</details>",
        unsafe_allow_html=True,
    )


def _stats(plan: dict) -> None:
    macros = plan.get("macros") or {}
    totals = plan.get("meal_totals") or {}
    st.markdown(
        '<div class="ff-stats">'
        f'<div><span>Daily target</span><strong>{_esc(macros.get("calories", "—"))}</strong><em>kcal</em></div>'
        f'<div><span>Protein target</span><strong>{_esc(macros.get("protein_g", "—"))}</strong><em>g</em></div>'
        f'<div><span>Meals</span><strong>{_esc(totals.get("calories", "—"))}</strong><em>kcal</em></div>'
        f'<div><span>Meal protein</span><strong>{_esc(totals.get("protein_g", "—"))}</strong><em>g</em></div>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.caption(macros.get("method") or "Rough estimates from a formula, not a lab test.")


def _render_plan(plan: dict) -> None:
    if not plan.get("validation", {}).get("ok", True):
        st.warning("This plan still needs a check: " + "; ".join(plan["validation"].get("issues") or []))

    _section("01", "Training")
    days = plan.get("workouts") or []
    width = 2
    for start in range(0, len(days), width):
        columns = st.columns(width)
        for column, day in zip(columns, days[start : start + width]):
            title = str(day.get("title") or "Day")
            day_name, focus = (title.split("·", 1) + [""])[:2] if "·" in title else (title, "")
            items = []
            for exercise in day.get("exercises") or []:
                items.append(
                    "<li><b>"
                    + _esc(exercise["name"])
                    + "</b><span>"
                    + _esc(f"{exercise['sets']} × {exercise['reps']} · rest {exercise['rest_sec']}s")
                    + "</span><small>"
                    + _esc(exercise.get("cue") or "")
                    + "</small></li>"
                )
            with column:
                st.markdown(
                    '<div class="ff-day"><div class="ff-day-top"><div><p class="ff-kicker">'
                    + _esc(day_name.strip())
                    + "</p><h3>"
                    + _esc(focus.strip() or "Session")
                    + '</h3></div><em>'
                    + _esc(f"{day.get('minutes', '')} min")
                    + "</em></div><ul>"
                    + "".join(items)
                    + "</ul></div>",
                    unsafe_allow_html=True,
                )

    _section("02", "Meals")
    meals = plan.get("meals") or []
    for start in range(0, len(meals), 2):
        meal_columns = st.columns(2)
        for column, meal in zip(meal_columns, meals[start : start + 2]):
            column.markdown(
                '<div class="ff-meal"><p class="ff-kicker">'
                + _esc(str(meal["slot"]).title())
                + "</p><h3>"
                + _esc(meal["name"])
                + '</h3><p class="ff-kcal"><strong>'
                + _esc(meal["calories"])
                + "</strong><span>kcal</span></p><p class=\"ff-sub\">"
                + _esc(f"{meal['protein_g']} g protein · {meal['portion']} serving")
                + '</p><p class="ff-ings">'
                + _esc(", ".join(meal.get("ingredients") or []))
                + "</p></div>",
                unsafe_allow_html=True,
            )


def _sidebar(profile: dict | None) -> None:
    st.sidebar.markdown('<p class="ff-side-brand">FitFuel</p>', unsafe_allow_html=True)
    st.sidebar.caption("Your week of training and meals.")
    if profile:
        st.sidebar.markdown(f"**{profile['name']}**")
        st.sidebar.caption(_profile_line(profile))
        if st.sidebar.button("New person", use_container_width=True):
            st.session_state.user_id = None
            st.session_state.pending_user = None
            st.session_state.answering = False
            st.session_state.form_nonce += 1
            st.rerun()
        plans = store.list_plans(st.session_state.user_id)
        if plans:
            labels = _plan_labels(plans)
            current = store.get_current_plan(st.session_state.user_id)
            ids = list(labels)
            picked = ids.index(current["id"]) if current and current["id"] in ids else 0
            choice = st.sidebar.selectbox(
                "Your plans",
                ids,
                index=picked,
                format_func=lambda item: labels[item],
                key=f"plans-{st.session_state.user_id}-{ids[picked]}",
            )
            if st.sidebar.button("Open this plan", use_container_width=True):
                store.set_current(st.session_state.user_id, choice)
                st.rerun()
        if st.sidebar.button("Start a new week", use_container_width=True):
            with st.spinner("Building another week..."):
                run_turn(
                    st.session_state.user_id,
                    "Build another week for me",
                    force_intent="generate",
                    use_llm=st.session_state.use_llm,
                    allow_web=False,
                )
            st.rerun()
    users = store.list_users()
    if users:
        st.sidebar.divider()
        saved = {}
        for item in users:
            if item["name"] not in saved:
                saved[item["name"]] = item["id"]
        saved = {user_id: name for name, user_id in saved.items()}
        picked = st.sidebar.selectbox(
            "Saved people",
            ["__new__"] + list(saved),
            format_func=lambda item: "Choose someone" if item == "__new__" else saved[item],
        )
        if picked != "__new__" and st.sidebar.button("Open", use_container_width=True):
            st.session_state.user_id = picked
            st.rerun()


def _onboarding() -> None:
    _banner("Your week of training and meals.", "Set your goal, then change the week in chat.")
    a, b, c = st.columns(3, gap="medium")
    cards = [
        ("Training that fits the week", "Sets, reps, and rest for the days and equipment you actually have."),
        ("Meals that follow your rules", "Breakfast through dinner, with allergies and diet choices left out."),
        ("Change the same week", "Ask for shorter sessions, lighter dinners, or a different way of eating."),
    ]
    for column, (title, copy) in zip((a, b, c), cards):
        column.markdown(
            f'<div class="ff-point"><strong>{title}</strong><span>{copy}</span></div>',
            unsafe_allow_html=True,
        )
    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
    with st.form(f"onboard-{st.session_state.form_nonce}"):
            st.subheader("Start your week")
            left, right = st.columns(2)
            with left:
                name = st.text_input("Name", value="")
                age = st.number_input("Age", min_value=16, max_value=80, value=25)
                sex = st.selectbox("Sex, for the calorie estimate", list(SEXES))
                weight = st.number_input("Weight (kg)", min_value=40.0, max_value=200.0, value=70.0, step=0.5)
                height = st.number_input("Height (cm)", min_value=140, max_value=220, value=170)
            with right:
                goal = st.selectbox("Goal", list(GOALS))
                days = st.slider("Training days each week", 2, 6, 4)
                minutes = st.selectbox("Minutes per session", [30, 45, 60], index=1)
                equipment = st.selectbox("Equipment", list(EQUIPMENT), index=3)
                diet = st.selectbox("Diet", list(DIETS))
                cuisine = st.selectbox("Cuisine", list(CUISINES))
                avoid = st.multiselect(
                    "Foods to leave out",
                    ["dairy", "eggs", "gluten", "nuts", "peanuts", "soy", "shellfish", "fish", "pork"],
                )
            allergies = st.text_input("Other allergies, separated by commas")
            notes = st.text_area("Anything else?", placeholder="Home gym, evenings only, no dairy")
            submitted = st.form_submit_button("Build my week", type="primary", use_container_width=True)
    if not submitted:
        return
    if medical_reason(notes) or medical_reason(allergies):
        st.error("Take injuries and medical conditions to a professional. Remove that note and FitFuel can build a general week.")
        return
    if not name.strip():
        st.error("Add your name to save this week.")
        return
    extra_allergies = [part.strip().lower() for part in allergies.split(",") if part.strip()]
    profile = clean_profile(
        {
            "name": name,
            "age": age,
            "sex": SEXES[sex],
            "weight_kg": weight,
            "height_cm": height,
            "goal": GOALS[goal],
            "days_per_week": days,
            "session_minutes": minutes,
            "equipment": EQUIPMENT[equipment],
            "diet": DIETS[diet],
            "cuisine": CUISINES[cuisine],
            "avoid": avoid,
            "allergies": extra_allergies,
            "notes": notes,
        }
    )
    user_id = store.create_user(profile)
    with st.spinner("Building your week..."):
        result = run_turn(
            user_id,
            "Create my weekly plan",
            force_intent="generate",
            use_llm=st.session_state.use_llm,
            allow_web=False,
        )
    st.session_state.user_id = user_id
    st.rerun()


def _footer() -> None:
    st.markdown(
        """
        <footer class="ff-footer">
          <p class="ff-kicker">Project</p>
          <h2>FitFuel Coach</h2>
          <p>Abdul Mueed Anwari · NLP and LLM bootcamp. A multi-agent gym and meal planner for beginners, built with Python, Streamlit, and LangGraph.</p>
          <ul>
            <li><span>Coach</span> reads the profile, routes the message, and saves the plan. Tools: get_user_profile, save_plan, update_plan.</li>
            <li><span>Workout</span> builds the week from the approved exercise list. Tool: knowledge retriever.</li>
            <li><span>Nutrition</span> builds meals and the calorie estimate. Tools: knowledge retriever, calculator, Wikipedia, DuckDuckGo.</li>
            <li><span>Validator</span> checks diet, allergies, equipment, and session length before the plan is shown. Tool: validate_plan.</li>
          </ul>
          <p>The week starts from the local list. Chat goes to the language model with your profile and this week. A change is checked first, then added, or you are told why it was not. This is not medical advice and it does not replace a doctor or dietitian.</p>
        </footer>
        """,
        unsafe_allow_html=True,
    )


def _queue(user_id: str, message: str) -> None:
    store.add_message(user_id, "user", message)
    st.session_state.pending_user = message
    st.session_state.answering = False
    st.rerun()


def _finish_reply(user_id: str) -> None:
    message = st.session_state.get("pending_user")
    if not message:
        return
    if not st.session_state.get("answering"):
        st.session_state.answering = True
        st.rerun()
    waiting = "Waiting for Groq..." if st.session_state.use_llm else "Coach is writing..."
    with st.spinner(waiting):
        result = run_turn(user_id, message, use_llm=st.session_state.use_llm, allow_web=True)
    store.add_message(user_id, "assistant", result["answer"], result["trace"])
    st.session_state.pending_user = None
    st.session_state.answering = False
    st.rerun()


def _scroll_thread(token: int) -> None:
    import streamlit.components.v1 as components

    components.html(
        f"""
        <div style="display:none">{token}</div>
        <script>
        const started = performance.now();
        const pin = () => {{
          const box = window.parent.document.querySelector(".ff-thread");
          if (box) box.scrollTop = box.scrollHeight;
          if (performance.now() - started < 2200) requestAnimationFrame(pin);
        }};
        pin();
        </script>
        """,
        height=0,
    )


def _chat(user_id: str) -> None:
    messages = _visible_messages(store.list_messages(user_id))
    thread_col, compose_col = st.columns([1.15, 1], gap="medium")
    with thread_col:
        live = " ff-thread-live" if messages or st.session_state.get("answering") else ""
        parts = [f'<div class="ff-thread{live}"><div class="ff-thread-inner">']
        if not messages and not st.session_state.get("answering"):
            parts.append('<p class="ff-thread-empty">No messages yet.</p>')
        for message in messages:
            mine = message["role"] == "user"
            who = "You" if mine else "Coach"
            kind = "ff-you" if mine else "ff-coach"
            body = _esc(message["content"]).replace("\n", "<br>")
            parts.append(f'<p class="ff-msg {kind}"><strong>{who}</strong>{body}</p>')
        if st.session_state.get("answering"):
            wait = "Waiting for Groq..." if st.session_state.use_llm else "Writing..."
            parts.append(f'<p class="ff-msg ff-coach"><strong>Coach</strong>{wait}</p>')
        parts.append('<div id="ff-latest"></div></div></div>')
        st.markdown("".join(parts), unsafe_allow_html=True)
    clicked = None
    with compose_col:
        st.markdown(
            '<p class="ff-coach-mark ff-coach-label">Coach</p>'
            '<p class="ff-coach-title">Ask our AI bot to assist you</p>',
            unsafe_allow_html=True,
        )
        st.caption("Talk normally about this week. If you ask for a change, it is checked first, then I tell you whether it was added.")
        picks = st.columns(4)
        for index, (label, text) in enumerate(SUGGESTIONS):
            if picks[index].button(label, key=f"suggest-{label}", use_container_width=True):
                clicked = text
        asked = any(
            message["role"] == "user" and message["content"] not in {"Create my weekly plan", "Build another week for me"}
            for message in messages
        )
        with st.form("coach-send", clear_on_submit=True):
            typed_col, send_col = st.columns([3, 1], vertical_alignment="bottom")
            with typed_col:
                typed = st.text_area(
                    "Message",
                    height=68,
                    placeholder="Make dinners under 500 calories",
                    label_visibility="collapsed",
                )
            with send_col:
                sent = st.form_submit_button(
                    "Send" if asked else "Ask",
                    use_container_width=True,
                    type="primary",
                )
        latest = next((message for message in reversed(messages) if message["role"] == "assistant"), None)
        if latest and not st.session_state.get("answering"):
            _trace(latest["trace"])
    _scroll_thread(len(messages) + (1 if st.session_state.get("answering") else 0))
    if clicked:
        _queue(user_id, clicked)
    elif sent and typed and typed.strip():
        _queue(user_id, typed.strip())
    elif st.session_state.get("pending_user"):
        _finish_reply(user_id)


def main() -> None:
    _init_state()
    _side_style()
    profile = None
    if st.session_state.user_id:
        try:
            profile = store.get_profile(st.session_state.user_id)
        except KeyError:
            st.session_state.user_id = None
    _sidebar(profile)
    if profile is None:
        _onboarding()
        _footer()
        return
    _banner(f"{profile['name']}'s week", _profile_line(profile))
    plan = store.get_current_plan(st.session_state.user_id)
    if plan:
        _stats(plan)
    if plan:
        _render_plan(plan)
    else:
        st.info("This week has not been built yet.")
        if st.button("Build my week"):
            run_turn(
                st.session_state.user_id,
                "Create my weekly plan",
                force_intent="generate",
                use_llm=st.session_state.use_llm,
            )
            st.rerun()
    with st.container(key="coach_dock"):
        _chat(st.session_state.user_id)
    _footer()


main()
