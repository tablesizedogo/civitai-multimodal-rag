"""Prompt normalisation shared by BM25 and the text-embedding index.

Civitai prompts are noisy: attention weights "(word:1.2)", LoRA tags
"<lora:name:0.8>", quality boilerplate ("masterpiece, best quality, score_9"),
underscores in booru tags, and sometimes a whole generator dump
("Positive Prompt: ... Seed: ..."). We strip all of that so that both
retrievers see roughly the *content* of the prompt.
"""
from __future__ import annotations

import re

_LORA = re.compile(r"<\s*(?:lora|lyco|hypernet|embedding)\s*:[^>]*>", re.I)
_WEIGHT = re.compile(r":\s*-?\d+(?:\.\d+)?\s*(?=[)\]>])")  # (word:1.2) -> (word)
_BRACKETS = re.compile(r"[()\[\]{}<>]")
_GEN_DUMP_KEYS = re.compile(
    r"^\s*(generation mode|model|width|height|seed|steps|cfg scale|sampler|scheduler|"
    r"seed variance[^:]*|negative prompt|clip skip|size|denoising[^:]*)\s*:.*$",
    re.I | re.M,
)
_POS_PREFIX = re.compile(r"^\s*positive prompt\s*:", re.I | re.M)
_EMBED_NAMES = re.compile(r"\bembedding:\S+", re.I)

BOILERPLATE = [
    r"score[ _]\d+(?:[ _]up)?", r"masterpiece", r"best[ _]quality", r"high(?:est)?[ _]quality",
    r"amazing[ _]quality", r"very[ _]aesthetic", r"absurdres", r"highres", r"high[ _]res",
    r"ultra[ _-]?detailed", r"extremely[ _]detailed", r"highly[ _]detailed", r"hyper[ _-]?detailed",
    r"8k", r"4k", r"uhd", r"hdr", r"high[ _]def(?:inition)?", r"newest", r"safe", r"sfw", r"pg",
    r"good[ _]quality", r"normal[ _]quality", r"intricate[ _]details", r"high[ _]resolution", r"break",
    r"rating[ _:]?(?:safe|general)", r"source_\w+", r"year[ _]20\d\d", r"official[ _]art",
]
_BOILER = re.compile(r"(?<![\w-])(?:" + "|".join(BOILERPLATE) + r")(?![\w-])", re.I)
_SPACES = re.compile(r"\s+")
_SEP = re.compile(r"\s*([,.;|])\s*(?:[,.;|]\s*)+")
_TOKEN = re.compile(r"[^\W_]+", re.U)

# Small stop-word list (EN + RU) for BM25; prompts are short, so keep it minimal.
STOP = set(
    "a an the of and or in on at to with for from by is are be as it its this that "
    "very into over under while their his her them they he she we you your our "
    "и в во на с со к по из за от до для о об а но не что это как".split()
)


def clean_prompt(text: str) -> str:
    """Human-readable cleaned prompt (used for embeddings and display)."""
    t = text or ""
    t = _POS_PREFIX.sub("", t)
    t = _GEN_DUMP_KEYS.sub("", t)
    t = _LORA.sub(" ", t)
    t = _EMBED_NAMES.sub(" ", t)
    t = _WEIGHT.sub("", t)
    t = _BRACKETS.sub(" ", t)
    t = _BOILER.sub(" ", t)
    t = t.replace("_", " ").replace("\\", " ")
    t = _BOILER.sub(" ", t)
    t = _SPACES.sub(" ", t)
    t = re.sub(r"\s+([,.;|])", r"\1", t)
    t = _SEP.sub(r"\1 ", t)
    return t.strip(" ,.;|")


def tokenize(text: str) -> list[str]:
    """Lower-cased word tokens without stop words (for BM25 docs and queries)."""
    return [w for w in _TOKEN.findall(text.lower()) if w not in STOP and len(w) > 1]
