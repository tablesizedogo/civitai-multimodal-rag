"""Strict-SFW guard shared by the search demo and the prompt assistant.

`BLOCK_TERMS` is a verbatim copy of the blocklist used by the collector
(`collect.py`) to drop images whose positive prompt mentions NSFW, suggestive,
gory or minor-related content. The collector matches terms as *prefixes*
(conservative: "bra" also drops "bracelet"), which is fine for building a
corpus but would refuse harmless user queries ("brave knight"). For queries and
generated text we therefore match whole words (with an optional plural "s"),
except for a few stems that are intentionally open-ended.

Russian terms are added because the demo accepts Russian queries.
"""
from __future__ import annotations

import re

BLOCK_TERMS = [
    "nsfw", "nude", "nudity", "naked", "nipple", "nipples", "areola", "topless", "bottomless",
    "breast", "breasts", "boob", "boobs", "cleavage", "underboob", "sideboob", "busty", "huge tits", "tits",
    "lingerie", "bikini", "underwear", "panties", "panty", "pantsu", "bra", "thong", "g-string", "micro bikini",
    "swimsuit", "bodysuit", "latex", "fishnet", "garter", "see-through", "see through", "transparent clothes",
    "wet clothes", "pussy", "vagina", "vulva", "penis", "cock", "dick", "testicles", "anus", "anal",
    "sex", "sexy", "sexual", "erotic", "erotica", "porn", "porno", "pornographic", "hentai", "ecchi", "lewd",
    "fetish", "bdsm", "bondage", "shibari", "ahegao", "cum", "semen", "orgasm", "masturbat", "fellatio",
    "blowjob", "paizuri", "futanari", "yaoi", "yuri", "seductive", "sensual", "provocative", "explicit",
    "uncensored", "ass", "butt", "buttocks", "crotch", "groin", "spread legs", "cameltoe", "upskirt",
    "nsfw:", "r18", "r-18", "18+", "xxx", "onlyfans", "stripper", "strip", "pole dance",
    "loli", "lolicon", "shota", "shotacon", "underage", "child nude",
    "gore", "guro", "gory", "dismember", "decapitat", "mutilat", "disembowel", "corpse", "guts", "entrails",
    "torture", "suicide", "self-harm",
    # added after visual review of sample_grid (borderline-suggestive outfits)
    "sheer", "plunging", "deep neckline", "low-cut", "revealing", "skimpy", "scantily", "barely clothed",
    "skin-tight", "tight dress", "midriff", "bare shoulders", "bare back", "backless", "thighs", "thigh",
    # anime / Illustrious-specific risky tags
    "oppai", "sukumizu", "school swimsuit", "leotard", "bunny girl", "bunnysuit", "bunny suit", "playboy",
    "thighhigh", "thigh-high", "thigh high", "zettai ryouiki", "stocking", "pantyhose", "skindentation",
    "covered nipples", "no bra", "nopan", "miniskirt", "micro skirt", "microskirt", "skirt lift", "lifted by self",
    "short shorts", "wardrobe malfunction", "off shoulder", "off-shoulder", "on bed", "lying on bed", "bedroom",
    "bath", "onsen", "shower", "towel", "wet", "sweat", "tentacle", "leash", "slave", "pregnant", "milf",
    "succubus", "curvy", "voluptuous", "hourglass", "wide hips", "hips", "tight", "yoga pants", "leggings",
    "navel", "belly", "cropped", "crop top", "tank top", "shirtless", "bare chest", "pecs", "muscular male",
    "from below", "pov", "spread", "kneeling", "all fours", "straddl", "licking", "tongue out", "saliva",
    "drool", "heavy breathing", "aroused", "blush lines", "flat chest", "petite", "little girl", "aged down",
    "child", "kid", "toddler", "infant", "young girl", "schoolgirl", "school girl", "gyaru", "harem", "kiss",
    "nsfw_", "rating:", "questionable", "sensitive", "bdsm", "latex suit", "gag", "collar",
    "blood", "bloody", "bloodied", "wound", "wounds", "weapon to head", "severed",
    # added after visual review of v3 grids (2026-10-19)
    "spaghetti strap", "spaghetti-strap", "chest is visible", "chest visible", "exposed chest", "bare shoulder",
    "oiran", "camisole", "strapless", "tube top", "low neckline", "big chest", "large chest", "huge chest",
    "tight top", "tight shirt", "form-fitting", "form fitting", "figure-hugging",
    "pg-13", "pg13", "squatting", "squat", "denim shorts", "short pants", "hot pants", "dark romance",
    "zero suit", "plugsuit", "skintight", "skin tight", "samus aran",
    "perfect legs", "perfect body", "beautiful legs", "long legs",
    # added after visual review of 100k grids (2026-10-20): foot-fetish / giantess cluster
    "giantess", "foot focus", "feet focus", "soles", "toes", "foot worship", "macrophilia", "vore", "ourap",
]

# Terms that are stems on purpose (match any continuation).
STEMS = {"masturbat", "decapitat", "mutilat", "disembowel", "straddl", "nsfw", "porn", "hentai", "erotic"}

# Russian query terms (stems; Cyrillic words inflect heavily).
RU_STEMS = [
    "голая", "голый", "голые", "обнаж", "нагота", "наг", "эрот", "порн", "секс", "хентай", "интим",
    "сиськ", "грудь", "груди", "попк", "задниц", "ягодиц", "бель", "трус", "лифчик", "бюстгальтер",
    "купальник", "бикини", "стринг", "чулк", "колготк", "декольте", "откровенн", "соблазн", "развратн",
    "пошл", "фетиш", "бдсм", "связан", "кровь", "кровав", "кровищ", "расчлен", "труп", "пытк",
    "самоубий", "суицид", "убийств", "лоли", "несовершеннолет", "ребен", "ребён", "дет", "малыш",
    "школьниц", "шлюх", "проститут", "стриптиз", "минет", "член", "вагин", "пенис", "оргазм",
]

# Extra patterns for model / LoRA names (collector rule for resource names).
RESOURCE_NAME_BLOCK = re.compile(r"uncensor|nsfw|porn|nude|naked|hentai|lewd|erotic|sexy|dirty|bypass|slider|ass\b|thigh|breast|boob|body|girls?\b", re.I)

_EN_WORDS = sorted({t.lower() for t in BLOCK_TERMS if t.lower() not in STEMS}, key=len, reverse=True)
_EN = re.compile(
    r"(?<![a-z0-9])(?:"
    + "|".join(re.escape(t) for t in sorted(STEMS, key=len, reverse=True))
    + r")|(?<![a-z0-9])(?:"
    + "|".join(re.escape(t) for t in _EN_WORDS)
    + r")(?:s|es)?(?![a-z0-9])",
    re.I,
)
_RU = re.compile(r"(?<![а-яё])(?:" + "|".join(re.escape(t) for t in RU_STEMS) + r")", re.I)
# Words that start like a Russian stem but are harmless.
_RU_ALLOW = re.compile(r"(?<![а-яё])(?:деталь|детал|детализ|наган|наград|наглядн|членств|бельгия|бельв|трусц|связь)", re.I)


# Harmless English phrases that contain a blocked word.
_EN_ALLOW = re.compile(r"comic strips?|strip mall|film strip|led strip|bath(?:ed)? in (?:light|sunlight|moonlight)", re.I)


def blocked_term(text: str) -> str | None:
    """Return the first blocked term found in `text` (EN or RU), else None."""
    t = _EN_ALLOW.sub(" ", text or "")
    m = _EN.search(t)
    if m:
        return m.group(0).lower()
    for m in _RU.finditer(t):
        tail = t[m.start():]
        if not _RU_ALLOW.match(tail):
            return m.group(0).lower()
    return None


def is_safe(text: str) -> bool:
    return blocked_term(text) is None


def safe_resource_name(name: str) -> bool:
    """LoRA / checkpoint names: drop 'uncensored', 'nsfw', 'filter bypass' etc."""
    return not (RESOURCE_NAME_BLOCK.search(name or "") or blocked_term(name or ""))


def sanitize(text: str) -> tuple[str, list[str]]:
    """Remove comma-separated chunks that contain a blocked term.

    Returns (clean_text, removed_terms). Used on LLM output as a last line of
    defence; the caller refuses if the result is empty.
    """
    removed, keep = [], []
    for chunk in re.split(r"(?<=[,.;\n])", text or ""):
        term = blocked_term(chunk)
        if term:
            removed.append(term)
        else:
            keep.append(chunk)
    out = re.sub(r"\s*,\s*,+", ", ", "".join(keep)).strip(" ,;")
    return out, removed
