#!/usr/bin/env python3
"""Collect SFW-only AI-generated images + generation metadata from Civitai public API.

Quota mode (v2): per-base-model-family quotas (FAMILIES). Rerunnable: state (per-stream cursors, seen ids, per-user counts) is kept in state.json,
so each run continues where the previous one stopped. Usage:
    python3 collect.py                        # fill quotas (resume via state.json)
    python3 collect.py --families flux        # only one family
"""
import argparse, json, os, re, sys, time, random, threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import requests

ROOT = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(ROOT, "images")
META = os.path.join(ROOT, "meta.jsonl")
STATE = os.path.join(ROOT, "state.json")
API = "https://civitai.com/api/v1/images"

FAMILIES = {
    "krea2": {"quota": 50000, "members": ["Krea 2", "Krea 2 Turbo"]},
    "flux": {"quota": 17000, "members": ["Flux.1 S", "Flux.1 D", "Flux.1 Krea", "Flux.1 Kontext", "Flux.2 D",
                                       "Flux.2 Klein 9B", "Flux.2 Klein 9B-base", "Flux.2 Klein 4B",
                                       "Flux.2 Klein 4B-base", "Flux.3", "Flux.3 Klein", "Flux.3 Klein-base"]},
    "illustrious": {"quota": 33000, "members": ["Illustrious"]},
    # v11: extra base models that fill the shortfall to ~100k (quota set via --extra-quota, default 0 = inactive)
    "zimage": {"quota": 0, "members": ["ZImageTurbo", "ZImageBase"], "extra": True},
    "anima": {"quota": 0, "members": ["Anima"], "extra": True},
    "sdxl": {"quota": 0, "members": ["SDXL 1.0", "SDXL Lightning", "SDXL Hyper", "SDXL Turbo"], "extra": True},
    "pony": {"quota": 0, "members": ["Pony", "Pony V7"], "extra": True},
    "moss": {"quota": 0, "members": ["MOSS Image"], "extra": True},
}
EXTRA = {f for f, d in FAMILIES.items() if d.get("extra")}
BASE2FAM = {m: f for f, d in FAMILIES.items() for m in d["members"]}
SORTS = [("Most Reactions", p) for p in ("Day", "Week", "Month", "Year", "AllTime")] + \
        [("Most Comments", p) for p in ("Day", "Week", "Month", "Year", "AllTime")] + \
        [("Most Collected", p) for p in ("Week", "Month", "Year", "AllTime")] + \
        [("Newest", p) for p in ("Week", "Month", "AllTime")]
# the API ignores sort/period when modelId is given (identical cursors), so one stream per model is enough
MODEL_SORTS = [("Most Reactions", "AllTime")]
MODELS_API = "https://civitai.com/api/v1/models"
MODEL_NAME_EXTRA_BLOCK = re.compile(r"uncensor|nsfw|porn|nude|naked|hentai|lewd|erotic|sexy|dirty|bypass|slider|ass\b|thigh|breast|boob|body|girls?\b", re.I)


def family_streams():
    out = {}
    for fam, d in FAMILIES.items():
        lst = []
        for m in d["members"]:
            for so, pe in SORTS:
                lst.append({"name": f"{fam}|{m}|{so}|{pe}", "sort": so, "period": pe, "baseModels": [m]})
        out[fam] = lst
    return out


def top_models(sess, base_model, mtype, n):
    """Popular SFW checkpoints/LoRAs of one base model (for per-model image streams)."""
    out, cursor = [], None
    while len(out) < n:
        params = [("limit", 50), ("types", mtype), ("baseModels", base_model), ("sort", "Most Downloaded"),
                  ("nsfw", "false")] + ([("cursor", cursor)] if cursor else [])
        try:
            r = sess.get(MODELS_API, params=params, timeout=60)
            d = r.json() if r.status_code == 200 else {}
        except Exception:
            d = {}
        items = d.get("items") or []
        for m in items:
            name = m.get("name") or ""
            if m.get("nsfw") or prompt_blocked(name) or MODEL_NAME_EXTRA_BLOCK.search(name):
                continue
            if not any((v.get("baseModel") == base_model) for v in m.get("modelVersions") or []):
                continue
            out.append({"id": m["id"], "name": name, "type": mtype})
        cursor = (d.get("metadata") or {}).get("nextCursor")
        if not items or not cursor:
            break
        time.sleep(1)
    return out[:n]


def model_streams(st, sess, fams, n_ckpt=40, n_lora=30):
    """Streams filtered by modelId of popular checkpoints/LoRAs inside each quota base model (cached in state)."""
    cache = st.setdefault("model_lists", {})
    out = {}
    for fam in fams:
        lst = []
        for m in FAMILIES[fam]["members"]:
            if m not in cache:
                cache[m] = top_models(sess, m, "Checkpoint", n_ckpt) + top_models(sess, m, "LORA", n_lora)
                print(f"model list {m}: {len(cache[m])}", flush=True)
            for md in cache[m]:
                for so, pe in MODEL_SORTS:
                    lst.append({"name": f"{fam}|model:{md['id']}|{so}|{pe}", "sort": so, "period": pe,
                                "baseModels": [], "modelId": md["id"]})
        out[fam] = lst
    return out


# Legacy (v1) sorting streams for diversity
STREAMS = [
    {"name": "reactions_month", "sort": "Most Reactions", "period": "Month"},
    {"name": "reactions_week", "sort": "Most Reactions", "period": "Week"},
    {"name": "comments_month", "sort": "Most Comments", "period": "Month"},
    {"name": "reactions_year", "sort": "Most Reactions", "period": "Year"},
    {"name": "newest", "sort": "Newest", "period": "AllTime"},
]

# Conservative NSFW blocklist applied to the POSITIVE prompt only
# (negative prompts usually contain "nsfw, nude" precisely to avoid them).
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
_PAT = re.compile(r"(?<![a-z0-9])(" + "|".join(re.escape(t) for t in BLOCK_TERMS) + r")", re.I)


def prompt_blocked(prompt):
    m = _PAT.search(prompt or "")
    return m.group(1).lower() if m else None


def purge_blocked():
    """Re-apply the current blocklist to already-kept records (after blocklist updates)."""
    if not os.path.exists(META):
        return 0
    keep, removed = [], 0
    manual = set()
    if os.path.exists(os.path.join(ROOT, "manual_remove.txt")):
        manual = {int(x) for x in open(os.path.join(ROOT, "manual_remove.txt")).read().split() if x.strip()}
    for line in open(META):
        r = json.loads(line)
        if r["id"] in manual or prompt_blocked(r.get("prompt")):
            p = os.path.join(ROOT, r["local_path"])
            if os.path.exists(p):
                os.remove(p)
            removed += 1
        else:
            keep.append(line)
    with open(META + ".tmp", "w") as f:
        f.writelines(keep)
    os.replace(META + ".tmp", META)
    return removed


def load_state():
    if os.path.exists(STATE):
        return json.load(open(STATE))
    return {"cursors": {}, "exhausted": [], "seen": [], "user_counts": {}, "stats": {}}


def save_state(st):
    tmp = STATE + ".tmp"
    json.dump(st, open(tmp, "w"))
    os.replace(tmp, STATE)


def get_json(sess, params, tries=6):
    delay = 2
    for _ in range(tries):
        try:
            r = sess.get(API, params=params, timeout=60)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                ra = r.headers.get("Retry-After")
                time.sleep(float(ra) if ra and ra.isdigit() else delay)
                delay = min(delay * 2, 120)
                continue
            print("API error", r.status_code, r.text[:200], file=sys.stderr)
            return None
        except requests.RequestException as e:
            print("API exc", e, file=sys.stderr)
            time.sleep(delay); delay = min(delay * 2, 120)
    return None


def resized_url(url):
    # CDN: replace "original=true" segment with width=512,optimized=true -> ~512px webp
    return re.sub(r"/(original=true|width=\d+[^/]*)/", "/width=512,optimized=true/", url, count=1)


def download(sess, item):
    url = resized_url(item["url"])
    delay = 2
    for _ in range(4):
        try:
            r = sess.get(url, timeout=60)
            if r.status_code == 200 and r.headers.get("content-type", "").startswith("image/") and len(r.content) > 2000:
                ext = {"image/webp": "webp", "image/jpeg": "jpg", "image/png": "png"}.get(
                    r.headers["content-type"].split(";")[0], "img")
                path = os.path.join(IMG_DIR, f"{item['id']}.{ext}")
                with open(path, "wb") as f:
                    f.write(r.content)
                try:
                    from PIL import Image
                    with Image.open(path) as im:
                        im.verify()
                except Exception:
                    os.remove(path); return None, url
                return path, url
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(delay); delay *= 2; continue
            return None, url
        except requests.RequestException:
            time.sleep(delay); delay *= 2
    return None, url


def model_name(meta):
    if meta.get("Model"):
        return meta["Model"]
    for r in meta.get("resources") or []:
        if isinstance(r, dict) and r.get("type") in ("model", "checkpoint") and r.get("name"):
            return r["name"]
    return None


def build_record(item, path, url):
    meta = item.get("meta") or {}
    loras = []
    for r in (meta.get("resources") or []):
        if isinstance(r, dict) and r.get("type") in ("lora", "lycoris", "locon", "embedding", "textualinversion"):
            loras.append({"name": r.get("name"), "type": r.get("type"), "weight": r.get("weight")})
    civ_res = [r for r in (meta.get("civitaiResources") or []) if isinstance(r, dict)]
    return {
        "id": item["id"],
        "image_url": item["url"],
        "download_url": url,
        "page_url": f"https://civitai.com/images/{item['id']}",
        "local_path": os.path.relpath(path, ROOT),
        "width": item.get("width"), "height": item.get("height"),
        "nsfwLevel": item.get("nsfwLevel"), "browsingLevel": item.get("browsingLevel"), "nsfw": item.get("nsfw"),
        "createdAt": item.get("createdAt"), "username": item.get("username"),
        "baseModel": item.get("baseModel"), "modelVersionIds": item.get("modelVersionIds"),
        "stats": item.get("stats"),
        "prompt": meta.get("prompt"), "negativePrompt": meta.get("negativePrompt"),
        "Model": model_name(meta), "sampler": meta.get("sampler"), "scheduler": meta.get("scheduler") or meta.get("Schedule type"),
        "steps": meta.get("steps"), "cfgScale": meta.get("cfgScale"), "seed": meta.get("seed"),
        "clipSkip": meta.get("clipSkip"), "size": meta.get("Size"),
        "resources": meta.get("resources"), "civitaiResources": civ_res, "loras": loras,
        "has_comfy_workflow": bool(meta.get("comfy")),
        "civitai_generator": bool(meta.get("ecosystem") or meta.get("civitaiResources")),
    }


def fam_of(rec_or_item):
    return BASE2FAM.get(rec_or_item.get("baseModel"))


def reorganize_existing():
    """Keep only records of quota families in meta.jsonl; move the rest to meta_other.jsonl + images_other/."""
    if not os.path.exists(META):
        return 0
    os.makedirs(os.path.join(ROOT, "images_other"), exist_ok=True)
    keep, other = [], []
    for line in open(META):
        r = json.loads(line)
        (keep if fam_of(r) else other).append((line, r))
    if not other:
        return 0
    with open(os.path.join(ROOT, "meta_other.jsonl"), "a") as f:
        for line, r in other:
            src = os.path.join(ROOT, r["local_path"])
            if os.path.exists(src):
                dst = os.path.join(ROOT, "images_other", os.path.basename(src))
                os.replace(src, dst)
                r["local_path"] = os.path.relpath(dst, ROOT)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(META + ".tmp", "w") as f:
        f.writelines(l for l, _ in keep)
    os.replace(META + ".tmp", META)
    return len(other)


def safe_record(r):
    return (r.get("nsfwLevel") == "None" and r.get("browsingLevel") == 1 and r.get("nsfw") is False
            and (r.get("prompt") or "").strip() and not prompt_blocked(r.get("prompt")))


def restore_from_other(manual):
    """v11: move earlier-collected records of now-active extra families back from meta_other.jsonl (re-checked)."""
    po = os.path.join(ROOT, "meta_other.jsonl")
    if not os.path.exists(po):
        return 0
    active = {f for f in EXTRA if FAMILIES[f]["quota"] > 0}
    keep_other, back, dropped = [], [], 0
    for line in open(po):
        r = json.loads(line)
        f = fam_of(r)
        if f in active:
            src = os.path.join(ROOT, r["local_path"])
            if r["id"] in manual or not safe_record(r) or not os.path.exists(src):
                if os.path.exists(src):
                    os.remove(src)
                dropped += 1; continue
            dst = os.path.join(IMG_DIR, os.path.basename(src))
            os.replace(src, dst); r["local_path"] = os.path.relpath(dst, ROOT)
            back.append(json.dumps(r, ensure_ascii=False) + "\n")
        else:
            keep_other.append(line)
    if back or dropped:
        with open(META, "a") as f:
            f.writelines(back)
        with open(po + ".tmp", "w") as f:
            f.writelines(keep_other)
        os.replace(po + ".tmp", po)
    return len(back), dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", type=int, default=200)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--cap-frac", type=float, default=0.01, help="per-author cap as fraction of family quota")
    ap.add_argument("--families", default=",".join(FAMILIES), help="comma list of families to work on")
    ap.add_argument("--no-model-streams", action="store_true")
    ap.add_argument("--threads", type=int, default=3, help="page-fetch threads per family")
    ap.add_argument("--extra-quota", default="", help="v11: e.g. zimage=8000,anima=8000")
    ap.add_argument("--max-empty-pages", type=int, default=25,
                    help="stop a stream after this many consecutive pages without new candidates")
    args = ap.parse_args()
    for kv in filter(None, args.extra_quota.split(",")):
        k, v = kv.split("="); assert k in EXTRA, k
        FAMILIES[k]["quota"] = int(v)
    print("quotas", {f: d["quota"] for f, d in FAMILIES.items()}, flush=True)
    os.makedirs(IMG_DIR, exist_ok=True)
    t0 = time.time()
    st = load_state()
    st.setdefault("q", {"cursors": {}, "exhausted": [], "empty_run": {}})
    q = st["q"]
    if not st.get("v9_reset"):
        # quotas were raised (and per-author caps with them): give previously stalled streams another chance
        st["v9_stats_before"] = dict(st["stats"])
        q["exhausted"] = []; q["empty_run"] = {}
        st["v9_reset"] = True
    stats = Counter(st["stats"])
    purged = purge_blocked()
    if purged:
        stats["drop_post_review_purge"] += purged
        print("purged", purged, "records by blocklist/manual_remove", flush=True)
    moved = reorganize_existing()
    if moved:
        stats["moved_non_quota_existing"] += moved
        print("moved", moved, "non-quota records to meta_other.jsonl", flush=True)
    manual = set()
    if os.path.exists(os.path.join(ROOT, "manual_remove.txt")):
        manual = {int(x) for x in open(os.path.join(ROOT, "manual_remove.txt")).read().split() if x.strip()}
    rb = restore_from_other(manual)
    if rb:
        stats["restored_from_other"] += rb[0]; stats["drop_restore_recheck"] += rb[1]
        print("restored from meta_other", rb, flush=True)
    seen = set(st["seen"])
    # v11: ids dropped earlier only as "base model not in quota" sit in `seen`; for extra-family streams dedupe
    # against kept ids + manual removals + a separate seen set instead (rating/prompt/blocklist checks are deterministic)
    seen_x = set(st.get("seen_x", []))
    capped = set(st.get("capped", []))
    recs = [json.loads(l) for l in open(META)] if os.path.exists(META) else []
    kept_ids = {r["id"] for r in recs} | {json.loads(l)["id"] for l in open(os.path.join(ROOT, "meta_other.jsonl"))} \
        if os.path.exists(os.path.join(ROOT, "meta_other.jsonl")) else {r["id"] for r in recs}
    fam_count = Counter(fam_of(r) for r in recs)
    ucount = Counter((fam_of(r), r.get("username") or "?") for r in recs)
    sess = requests.Session(); sess.headers["User-Agent"] = "rag-civitai-research/0.2"
    pool = ThreadPoolExecutor(args.workers)
    metaf = open(META, "a")
    allstreams = family_streams()
    fams = [f for f in args.families.split(",") if f in FAMILIES and FAMILIES[f]["quota"] > 0]
    if not args.no_model_streams:
        ms = model_streams(st, sess, fams)
        save_state(st)
        for f in fams:
            # interleave: base-model streams first, per-model streams after
            allstreams[f] = allstreams[f] + ms.get(f, [])
    active = {f: [s for s in allstreams[f] if s["name"] not in q["exhausted"]] for f in fams}
    idx = Counter()

    def save():
        stats["kept"] = sum(fam_count.values())
        st.update(seen=list(seen), stats=dict(stats), capped=list(capped), seen_x=list(seen_x))
        st["fam_count"] = dict(fam_count)
        save_state(st)

    def iid_ok(it):
        return it["id"] not in kept_ids
    lock = threading.Lock()
    inflight = Counter(); busy = set(); last_save = [0.0]
    tl = threading.local()

    def api_sess():
        if not hasattr(tl, "s"):
            tl.s = requests.Session(); tl.s.headers["User-Agent"] = "rag-civitai-research/0.3"
        return tl.s

    def pick(f):
        quota = FAMILIES[f]["quota"]
        if fam_count[f] + inflight[f] >= quota or not active[f]:
            return "STOP" if (fam_count[f] >= quota or not active[f]) else None
        for _ in range(len(active[f])):
            s = active[f][idx[f] % len(active[f])]; idx[f] += 1
            if s["name"] not in busy:
                busy.add(s["name"]); return s
        return None

    def worker(f):
        while True:
            with lock:
                s = pick(f)
            if s == "STOP":
                return
            if s is None:
                time.sleep(2); continue
            params = [("limit", args.page), ("nsfw", "None"), ("browsingLevel", 1), ("withMeta", "true"),
                      ("sort", s["sort"]), ("period", s["period"])] + [("baseModels", b) for b in s["baseModels"]]
            if s.get("modelId"):
                params.append(("modelId", s["modelId"]))
            cur = q["cursors"].get(s["name"])
            if cur:
                params.append(("cursor", cur))
            data = get_json(api_sess(), params)
            time.sleep(args.sleep)
            with lock:
                if data is None:
                    stats["api_failed_pages"] += 1
                    q["empty_run"][s["name"]] = q["empty_run"].get(s["name"], 0) + 3
                    if q["empty_run"][s["name"]] >= args.max_empty_pages and s in active[f]:
                        active[f].remove(s); q["exhausted"].append(s["name"])
                    busy.discard(s["name"]); continue
                items = data.get("items") or []
                stats["requested"] += len(items)
                cands = []
                for it in items:
                    iid = it.get("id")
                    if f in EXTRA:
                        if iid in kept_ids or iid in seen_x or iid in manual:
                            stats["drop_duplicate"] += 1; stats["x:drop_duplicate"] += 1; continue
                        seen_x.add(iid)
                    elif iid in seen:
                        stats["drop_duplicate"] += 1; continue
                    seen.add(iid)
                    if it.get("type") != "image":
                        stats["drop_not_image"] += 1; continue
                    if it.get("nsfwLevel") is None or it.get("browsingLevel") is None or it.get("nsfw") is None:
                        stats["drop_missing_rating"] += 1; continue
                    if it.get("nsfwLevel") != "None" or it.get("browsingLevel") != 1 or it.get("nsfw") is not False:
                        stats["drop_not_safest_rating"] += 1
                        if f in EXTRA: stats[f"x:{f}:drop_not_safest_rating"] += 1
                        continue
                    fi = fam_of(it)
                    if fi is None:
                        stats["drop_base_model_not_in_quota"] += 1; continue
                    meta = it.get("meta") or {}
                    if not (meta.get("prompt") or "").strip():
                        stats["drop_no_prompt"] += 1
                        if fi in EXTRA: stats[f"x:{fi}:drop_no_prompt"] += 1
                        continue
                    term = prompt_blocked(meta["prompt"])
                    if term:
                        stats["drop_prompt_blocklist"] += 1
                        stats[f"drop_prompt_blocklist:{fi}"] += 1
                        stats["blockterm:" + term] += 1; continue
                    u = it.get("username") or "?"
                    capf = max(5, int(FAMILIES[fi]["quota"] * args.cap_frac))
                    if ucount[(fi, u)] >= capf:
                        seen.discard(iid)
                        if iid not in capped:
                            capped.add(iid); stats["drop_user_cap"] += 1
                        continue
                    ucount[(fi, u)] += 1
                    cands.append(it)
                batch, defer = [], []
                room = {g: FAMILIES[g]["quota"] - fam_count[g] - inflight[g] for g in FAMILIES}
                for it in cands:
                    g = fam_of(it)
                    if room[g] > 0 and iid_ok(it):
                        batch.append(it); room[g] -= 1; inflight[g] += 1
                    else:
                        defer.append(it)
                for it in defer:
                    seen.discard(it["id"]); ucount[(fam_of(it), it.get("username") or "?")] -= 1
                    stats["deferred_quota_full"] += 1
            results = list(pool.map(lambda x: download(sess, x), batch))
            with lock:
                ok = 0
                for it, (path, url) in zip(batch, results):
                    inflight[fam_of(it)] -= 1
                    if not path:
                        stats["drop_download_failed"] += 1
                        ucount[(fam_of(it), it.get("username") or "?")] -= 1
                        continue
                    metaf.write(json.dumps(build_record(it, path, url), ensure_ascii=False) + "\n")
                    kept_ids.add(it["id"])
                    fam_count[fam_of(it)] += 1; ok += 1
                metaf.flush()
                q["empty_run"][s["name"]] = 0 if ok else q["empty_run"].get(s["name"], 0) + 1
                nxt = (data.get("metadata") or {}).get("nextCursor")
                if nxt and items:
                    q["cursors"][s["name"]] = nxt
                if not items:
                    # the API sometimes returns an empty page on heavy queries (timeouts): retry later, give up after ~5
                    stats["api_empty_pages"] += 1
                    q["empty_run"][s["name"]] = q["empty_run"].get(s["name"], 0) + 4
                if (items and not nxt) or q["empty_run"][s["name"]] >= args.max_empty_pages:
                    if s in active[f]:
                        active[f].remove(s)
                    q["exhausted"].append(s["name"])
                    print(f"stream done: {s['name']}", flush=True)
                busy.discard(s["name"])
                if time.time() - last_save[0] > 15:
                    save(); last_save[0] = time.time()
                el = int(time.time() - t0)
                print(f"[{el}s] {s['name']} page={len(items)} new={ok} counts={dict(fam_count)}", flush=True)

    def safe_worker(f):
        while True:
            try:
                return worker(f)
            except Exception:
                import traceback; traceback.print_exc()
                time.sleep(5)

    threads = [threading.Thread(target=safe_worker, args=(f,), daemon=True) for f in fams for _ in range(args.threads)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    pool.shutdown(); metaf.close(); save()
    for f in fams:
        print(f, fam_count[f], "/", FAMILIES[f]["quota"], "REACHED" if fam_count[f] >= FAMILIES[f]["quota"] else "API EXHAUSTED")
    print(json.dumps({k: v for k, v in stats.items() if not k.startswith("blockterm:")}, indent=1))
    print("elapsed_s", int(time.time() - t0))


if __name__ == "__main__":
    main()
