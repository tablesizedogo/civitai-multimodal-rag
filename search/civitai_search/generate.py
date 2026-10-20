"""Generation step of the RAG pipeline: a grounded prompt assistant.

    python -m civitai_search.generate "foggy forest at dawn, watercolor" [--method E] [--k 6]

Pipeline
  1. SFW guard on the user idea (safety.blocked_term) -> refuse if blocked.
  2. Retrieve top-k reference images (default E = multilingual CLIP, works for
     English and Russian ideas) together with their real prompts and settings.
  3. A small local instruct LLM (Qwen2.5-1.5B-Instruct, GGUF Q4_K_M, llama.cpp
     on CPU) writes JSON: prompt, negative prompt, base model, checkpoint,
     sampler, steps, CFG, LoRAs and the reference ids it used.
     In "constrained" mode (default) the JSON schema turns every setting into an
     enum of values that actually occur in the references, so the decoder cannot
     emit an invented sampler or LoRA. "free" mode keeps plain types and is used
     in the evaluation to measure how often the raw LLM stays grounded.
  4. Post-check: every setting is looked up in the references. A value is
     *grounded* if some reference has it, *coherent* if a reference with the
     recommended base model has it. Anything else is replaced by the
     rank-weighted majority value of the coherent references and marked as
     corrected. Each kept value is returned with the ids that support it.
  5. SFW guard on the generated prompt (sanitize, refuse if nothing is left);
     LoRA / checkpoint names like "uncensored" or "filter bypass" are dropped.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from functools import lru_cache

from .config import LLM_FILE, LLM_GGUF, LLM_REPO, LLM_THREADS
from .data import Item, load_items, model_family
from .safety import blocked_term, safe_resource_name, sanitize

SETTING_FIELDS = ("base_model", "checkpoint", "sampler", "steps", "cfg")

SYSTEM_PROMPT = """You are a prompt assistant for text-to-image models (Flux, Krea, Illustrious, SDXL, Pony).
You receive a user idea and several REFERENCE images found on Civitai, each with its real prompt and generation settings.
Write a new prompt for the user's idea that borrows the wording, style tags and structure of the references that fit the idea best.
Rules:
- The prompt must be in English, 25-70 words, and describe the user's idea (not a reference's subject). Do not copy tags that contradict the idea.
- Use the prompt style of the chosen base model: comma-separated tags for Illustrious/Pony/NoobAI/SDXL anime models, natural sentences for Flux/Krea/Z Image.
- Settings (base_model, checkpoint, sampler, steps, cfg, loras) must be copied from the references, preferably all from references with the same base model. Never invent a value. If no reference lists a checkpoint, use "none". If no fitting reference uses LoRAs, return [].
- negative_prompt: adapt the references' negative prompt, or "" if they have none.
- The content must be safe for work: no nudity, no suggestive content, no gore.
- sources: ids of the references you relied on.
Reply with JSON only."""


@dataclass
class Setting:
    value: object
    support: list[int] = field(default_factory=list)  # reference ids with this value
    grounded: bool = False  # value occurs in some reference (raw LLM output)
    coherent: bool = False  # ...in a reference with the recommended base model
    corrected: bool = False  # replaced by the post-check
    raw: object = None  # what the LLM produced


@dataclass
class Answer:
    query: str
    refused: str | None = None
    prompt: str = ""
    negative_prompt: str = ""
    settings: dict = field(default_factory=dict)  # name -> Setting
    loras: list = field(default_factory=list)  # list of dict(name, weight, support, grounded, coherent)
    dropped_loras: list = field(default_factory=list)
    sources: list[int] = field(default_factory=list)
    invalid_sources: list = field(default_factory=list)
    removed_terms: list[str] = field(default_factory=list)
    hits: list = field(default_factory=list)  # (row, score)
    ref_ids: list[int] = field(default_factory=list)
    raw_json: str = ""
    json_ok: bool = True
    timings: dict = field(default_factory=dict)
    mode: str = "constrained"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["hits"] = [[int(r), float(s)] for r, s in self.hits]
        return d


# ---------------------------------------------------------------- helpers

_LORA_TAG = re.compile(r"<\s*(?:lora|lyco)\s*:[^>]*>", re.I)


def _norm(v) -> str:
    """Normalise a setting value for matching ('Euler a' == 'euler_a' == 'EULER A')."""
    if v is None:
        return ""
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    s = str(v).strip().lower().replace("\\", "/").split("/")[-1]
    s = re.sub(r"\.(safetensors|ckpt|pt|gguf)$", "", s)
    return re.sub(r"[\s_\-]+", "", s)


def _sampler(it: Item) -> str | None:
    s = it.sampler
    if not s or s.lower() in ("undefined", "none"):
        return None
    return s


def _checkpoint(it: Item) -> str | None:
    m = it.model
    if not m or not safe_resource_name(m):
        return None
    urn = re.match(r"urn:air:\w+:checkpoint:civitai:(\d+)@(\d+)", m)
    if urn:  # Civitai AIR id -> readable form
        return f"civitai model {urn.group(1)} (version {urn.group(2)})"
    return m.replace("\\", "/").split("/")[-1].removesuffix(".safetensors")


def _num(v):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return int(f) if f.is_integer() else round(f, 2)


def item_settings(it: Item) -> dict:
    return {
        "base_model": it.base_model or None,
        "checkpoint": _checkpoint(it),
        "sampler": _sampler(it),
        "steps": _num(it.steps),
        "cfg": _num(it.cfg_scale),
    }


def item_loras(it: Item) -> list[dict]:
    out = []
    for l in it.loras or []:
        if isinstance(l, dict) and l.get("name"):
            out.append({"name": str(l["name"]), "weight": _num(l.get("weight"))})
    return out


def _prompt_for_llm(text: str, limit: int) -> str:
    """Strip LoRA tags, collapse whitespace, cut at the last comma/space before `limit`."""
    t = _LORA_TAG.sub(" ", text or "")
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) <= limit:
        return t
    cut = t[:limit]
    pos = max(cut.rfind(","), cut.rfind(". "))
    if pos < limit // 2:
        pos = cut.rfind(" ")
    return cut[:pos].rstrip(" ,.") + " ..."


def format_references(items: list[Item], prompt_chars: int = 360) -> str:
    blocks = []
    for it in items:
        s = item_settings(it)
        loras = ", ".join(f"{l['name']}:{l['weight']}" for l in item_loras(it) if safe_resource_name(l["name"])) or "none"
        neg = _prompt_for_llm(it.negative_prompt, 160) or "none"
        blocks.append(
            f"[#{it.id}] base_model: {s['base_model']} | checkpoint: {s['checkpoint'] or 'none'} | "
            f"sampler: {s['sampler'] or 'n/a'} | steps: {s['steps'] if s['steps'] is not None else 'n/a'} | "
            f"cfg: {s['cfg'] if s['cfg'] is not None else 'n/a'} | loras: {loras}\n"
            f"prompt: {_prompt_for_llm(it.prompt, prompt_chars)}\nnegative: {neg}"
        )
    return "\n\n".join(blocks)


def json_schema(refs: list[Item] | None, constrained: bool) -> dict:
    """JSON schema for llama.cpp grammar-constrained decoding."""
    def enum_or(typ, values):
        vals = sorted({v for v in values if v is not None}, key=str)
        if constrained and vals:
            return {"enum": vals}
        return {"type": typ}

    refs = refs or []
    sets = [item_settings(it) for it in refs]
    lora_names = sorted({l["name"] for it in refs for l in item_loras(it) if safe_resource_name(l["name"])})
    lora_item = {"type": "object", "properties": {
        "name": {"enum": lora_names} if (constrained and lora_names) else {"type": "string"},
        "weight": {"type": "number"}}, "required": ["name", "weight"]}
    props = {
        "prompt": {"type": "string", "maxLength": 520},
        "negative_prompt": {"type": "string", "maxLength": 300},
        "base_model": enum_or("string", [s["base_model"] for s in sets]),
        "checkpoint": ({"enum": sorted({s["checkpoint"] for s in sets if s["checkpoint"]}) + ["none"]}
                       if constrained else {"type": "string"}),
        "sampler": enum_or("string", [s["sampler"] for s in sets]),
        "steps": enum_or("integer", [s["steps"] for s in sets]),
        "cfg": enum_or("number", [s["cfg"] for s in sets]),
        "loras": {"type": "array", "items": lora_item, "maxItems": 4} if (lora_names or not constrained)
        else {"type": "array", "maxItems": 0},
        "sources": {"type": "array", "items": {"enum": [it.id for it in refs]} if constrained else {"type": "integer"},
                    "maxItems": 6},
    }
    return {"type": "object", "properties": props, "required": list(props)}


# ---------------------------------------------------------------- LLM

class LocalLLM:
    def __init__(self, gguf: str | None = None, n_ctx: int = 4096, threads: int = LLM_THREADS):
        from llama_cpp import Llama

        path = gguf or LLM_GGUF
        if not path:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(LLM_REPO, LLM_FILE)
        self.path = path
        self.name = path.split("/")[-1]
        self.llm = Llama(model_path=path, n_ctx=n_ctx, n_threads=threads, verbose=False, seed=0)

    def chat_json(self, system: str, user: str, schema: dict, max_tokens: int = 420, temperature: float = 0.2):
        out = self.llm.create_chat_completion(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_tokens, temperature=temperature, top_p=0.9, seed=0, repeat_penalty=1.15,
            response_format={"type": "json_object", "schema": schema},
        )
        return out["choices"][0]["message"]["content"], out.get("usage", {})


@lru_cache(maxsize=2)
def get_llm(gguf: str | None = None) -> LocalLLM:
    return LocalLLM(gguf)


# ---------------------------------------------------------------- post-check

def _weighted_majority(values: list[tuple[object, float]]):
    acc: dict[str, list] = {}
    for v, w in values:
        if v is None:
            continue
        k = _norm(v)
        if k not in acc:
            acc[k] = [v, 0.0]
        acc[k][1] += w
    if not acc:
        return None
    return max(acc.values(), key=lambda x: x[1])[0]


def ground(parsed: dict, refs: list[Item]) -> tuple[dict, list, list]:
    """Check every recommended setting against the references; fix what is not grounded."""
    weights = {it.id: 1.0 / (r + 1) for r, it in enumerate(refs)}
    sets = {it.id: item_settings(it) for it in refs}
    fam = {it.id: model_family(it.base_model) for it in refs}

    out: dict[str, Setting] = {}
    # 1) base model first: it defines which references are "coherent"
    raw_bm = parsed.get("base_model")
    sup = [i for i, s in sets.items() if s["base_model"] and _norm(s["base_model"]) == _norm(raw_bm)]
    bm = Setting(value=raw_bm, raw=raw_bm, support=sup, grounded=bool(sup), coherent=bool(sup))
    if not sup:
        bm.value = _weighted_majority([(s["base_model"], weights[i]) for i, s in sets.items()])
        bm.support = [i for i, s in sets.items() if _norm(s["base_model"]) == _norm(bm.value)]
        bm.corrected = True
    out["base_model"] = bm
    coh_ids = set(bm.support) or {i for i in sets if fam[i] == model_family(str(bm.value or ""))}

    for f in ("checkpoint", "sampler", "steps", "cfg"):
        raw = parsed.get(f)
        if f in ("steps", "cfg"):
            raw = _num(raw)
        if f == "checkpoint" and (raw is None or _norm(raw) in ("", "none", "na")):
            # "none" is grounded if a coherent reference has no checkpoint listed
            sup = [i for i in coh_ids if not sets[i]["checkpoint"]]
            out[f] = Setting(value=None, raw=raw, support=sup, grounded=True, coherent=bool(sup))
            if not sup:  # every coherent ref names a checkpoint -> recommend the majority one
                st = out[f]
                st.value = _weighted_majority([(sets[i][f], weights[i]) for i in coh_ids])
                st.support = [i for i in coh_ids if _norm(sets[i][f]) == _norm(st.value)]
                st.corrected = True
            continue
        any_sup = [i for i, s in sets.items() if s[f] is not None and _norm(s[f]) == _norm(raw)]
        coh_sup = [i for i in any_sup if i in coh_ids]
        st = Setting(value=raw, raw=raw, support=coh_sup, grounded=bool(any_sup), coherent=bool(coh_sup))
        if not coh_sup:
            st.value = _weighted_majority([(sets[i][f], weights[i]) for i in coh_ids])
            st.support = [i for i in coh_ids if st.value is not None and _norm(sets[i][f]) == _norm(st.value)]
            st.corrected = True
        out[f] = st

    # LoRAs: keep only names that a coherent reference actually used (and that are SFW)
    lora_src = defaultdict(list)
    for it in refs:
        for l in item_loras(it):
            lora_src[_norm(l["name"])].append((it.id, l))
    kept, dropped = [], []
    for l in parsed.get("loras") or []:
        name = l.get("name") if isinstance(l, dict) else str(l)
        if not name:
            continue
        src = lora_src.get(_norm(name), [])
        coh = [(i, ll) for i, ll in src if i in coh_ids]
        if not safe_resource_name(name):
            dropped.append({"name": name, "reason": "unsafe name"})
        elif not coh:
            dropped.append({"name": name, "reason": "not in references" if not src else "other base model",
                            "grounded": bool(src)})
        elif any(_norm(k["name"]) == _norm(name) for k in kept):
            continue
        else:
            w_ref = coh[0][1]["weight"]
            kept.append({"name": coh[0][1]["name"], "weight": w_ref, "llm_weight": _num(l.get("weight")) if isinstance(l, dict) else None,
                         "support": [i for i, _ in coh], "grounded": True, "coherent": True})
    return out, kept, dropped


def tidy_prompt(text: str, max_words: int = 90) -> str:
    """Drop repeated comma-separated tags and runaway repetitions of a small LLM."""
    seen, out, words = set(), [], 0
    for tag in re.split(r",\s*", text.strip()):
        key = re.sub(r"\W+", " ", tag.lower()).strip()
        if not key or key in seen:
            continue
        # a tag whose words all appeared already in two or more earlier tags is a loop
        toks = key.split()
        if len(toks) >= 3 and sum(any(t in s for s in seen) for t in toks) == len(toks) and len(out) > 8:
            continue
        seen.add(key)
        out.append(tag.strip())
        words += len(toks)
        if words >= max_words:
            break
    return ", ".join(out)


# ---------------------------------------------------------------- assistant

class PromptAssistant:
    def __init__(self, rets: dict, llm: LocalLLM | None = None, default_method: str = "E"):
        self.rets = rets
        self.items = load_items()
        self.llm = llm or get_llm()
        self.default_method = default_method if default_method in rets else "C"

    def retrieve(self, query: str, method: str | None, k: int):
        r = self.rets[method or self.default_method]
        hits = r.search(query, k * 2)
        # defence in depth: the corpus is already SFW-filtered
        hits = [(row, s) for row, s in hits if blocked_term(self.items[row].prompt) is None]
        return hits[:k]

    def answer(self, query: str, method: str | None = None, k: int = 6, mode: str = "constrained",
               hits: list | None = None) -> Answer:
        ans = Answer(query=query, mode=mode)
        term = blocked_term(query)
        if term:
            ans.refused = f"The request contains a blocked term ('{term}'). This demo is strictly SFW."
            return ans
        t0 = time.perf_counter()
        ans.hits = hits if hits is not None else self.retrieve(query, method, k)
        refs = [self.items[r] for r, _ in ans.hits[:k]]
        ans.ref_ids = [it.id for it in refs]
        t1 = time.perf_counter()
        if not refs:
            ans.refused = "No reference images found."
            return ans
        user = f"REFERENCES:\n{format_references(refs)}\n\nUSER IDEA: {query}\n\nWrite the JSON."
        raw, usage = self.llm.chat_json(SYSTEM_PROMPT, user, json_schema(refs, mode == "constrained"))
        t2 = time.perf_counter()
        ans.raw_json = raw
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            ans.json_ok, parsed = False, {}
        ans.settings, ans.loras, ans.dropped_loras = ground(parsed, refs)
        ref_set = set(ans.ref_ids)
        for s in parsed.get("sources") or []:
            try:
                (ans.sources if int(s) in ref_set else ans.invalid_sources).append(int(s))
            except (TypeError, ValueError):
                ans.invalid_sources.append(s)
        ans.prompt, rem1 = sanitize(tidy_prompt(str(parsed.get("prompt") or "")))
        ans.negative_prompt = str(parsed.get("negative_prompt") or "").strip()
        ans.removed_terms = rem1
        if not ans.prompt:
            ans.refused = "The generated prompt did not pass the SFW filter." if rem1 else "The model returned no prompt."
        ans.timings = {"retrieval_s": round(t1 - t0, 3), "llm_s": round(t2 - t1, 2),
                       "total_s": round(time.perf_counter() - t0, 2),
                       "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens")}
        return ans


# ---------------------------------------------------------------- rendering

def _cite(ids: list[int], limit: int = 4) -> str:
    links = [f"[#{i}](https://civitai.com/images/{i})" for i in ids[:limit]]
    if len(ids) > limit:
        links.append(f"+{len(ids) - limit}")
    return ", ".join(links) or "—"


def render_markdown(ans: Answer) -> str:
    if ans.refused:
        return f"**Refused.** {ans.refused}"
    lines = ["### Prompt", f"```\n{ans.prompt}\n```"]
    if ans.negative_prompt:
        lines += ["**Negative prompt**", f"```\n{ans.negative_prompt}\n```"]
    lines += ["### Settings", "| Setting | Value | Supported by | Check |", "|---|---|---|---|"]
    for f in SETTING_FIELDS:
        s = ans.settings.get(f)
        if s is None:
            continue
        val = "—" if s.value is None else s.value
        if f == "checkpoint" and s.value is None:
            val = "(base model default, none listed)"
        check = "✓ grounded" if not s.corrected else f"↺ corrected (LLM said `{s.raw}`)"
        lines.append(f"| {f} | `{val}` | {_cite(s.support)} | {check} |")
    for l in ans.loras:
        lines.append(f"| LoRA | `{l['name']}` @ {l['weight']} | {_cite(l['support'])} | ✓ grounded |")
    if not ans.loras:
        lines.append("| LoRA | none | — | |")
    for d in ans.dropped_loras:
        lines.append(f"| LoRA | ~~{d['name']}~~ | — | dropped: {d['reason']} |")
    if ans.sources:
        lines.append(f"\n**Prompt inspired by:** {_cite(ans.sources, 6)}")
    if ans.removed_terms:
        lines.append(f"\n_SFW filter removed: {', '.join(sorted(set(ans.removed_terms)))}_")
    t = ans.timings
    lines.append(f"\n<sub>retrieval {t.get('retrieval_s')} s · LLM {t.get('llm_s')} s "
                 f"({t.get('prompt_tokens')} prompt + {t.get('completion_tokens')} generated tokens)</sub>")
    return "\n".join(lines)


def main() -> None:
    from .retrieval import build_all

    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--method", default="E")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--mode", choices=["constrained", "free"], default="constrained")
    ap.add_argument("--gguf", default=None)
    args = ap.parse_args()
    rets = build_all(only=args.method)
    pa = PromptAssistant(rets, get_llm(args.gguf), default_method=args.method)
    ans = pa.answer(args.query, args.method, args.k, args.mode)
    print(render_markdown(ans))
    print(json.dumps(ans.timings))


if __name__ == "__main__":
    main()
