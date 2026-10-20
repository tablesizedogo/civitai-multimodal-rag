"""Demo screenshots for the README (needs the app running on :7861 and playwright).

    python -m civitai_search.app --port 7861 &
    python docs/make_screenshots.py [--url http://127.0.0.1:7861] [--image path/to/query.webp]
"""
import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent / "screenshots"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:7861")
    ap.add_argument("--image", default=None)
    ap.add_argument("--chrome", default="/usr/bin/google-chrome")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=a.chrome, headless=True)
        pg = b.new_page(viewport={"width": 1500, "height": 1000})
        pg.goto(a.url, wait_until="networkidle")

        # 1. text search (Russian query, method E)
        pg.get_by_label("Text query").fill("лиса в осеннем лесу")
        pg.get_by_role("button", name="Search", exact=True).click()
        pg.wait_for_selector("details", timeout=60000)
        pg.wait_for_timeout(2500)
        pg.screenshot(path=str(OUT / "search_text_ru.png"), full_page=False)

        # 2. image search
        if a.image:
            pg.get_by_label("Text query").fill("")
            pg.locator("input[type=file]").first.set_input_files(a.image)
            pg.wait_for_timeout(1500)
            pg.get_by_role("button", name="Search", exact=True).click()
            pg.wait_for_timeout(5000)
            pg.screenshot(path=str(OUT / "search_image.png"), full_page=False)

        # 3. prompt assistant
        pg.get_by_role("tab", name="Prompt assistant (RAG)").click()
        pg.get_by_label("Your idea").fill("a cozy cabin in a snowy forest at night, warm light in the windows")
        pg.get_by_role("button", name="Write prompt").click()
        pg.get_by_text("Supported by").wait_for(timeout=180000)
        pg.wait_for_timeout(2500)
        pg.screenshot(path=str(OUT / "assistant.png"), full_page=True)
        b.close()
    print("saved to", OUT)


if __name__ == "__main__":
    main()
