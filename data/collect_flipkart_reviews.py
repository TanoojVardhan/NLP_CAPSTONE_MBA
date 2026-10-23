"""
collect_flipkart_reviews.py
---------------------------------------------------------------------------------------------
Collects public Flipkart product reviews in the same three views we used for the project
(most recent, negative first, most helpful) and writes them in the same raw format as our
original dump (flipkart_raw.psv).

NOTE: this is a REBUILT version of the collection step, written on 23 Oct 2026.
The original one-off browser script from 14 Oct 2026 was not saved. This script follows
the same method, but it is not the exact code that produced data/flipkart_raw.csv.

How it works
1. Takes the product list (web-address name + Flipkart item ID) from data/flipkart_raw.csv.
2. Opens a normal Chrome/Chromium window (you can watch it) and, for each product, opens
   the review pages sorted three ways, page by page.
3. Reads every review card on the page: rating, title, text, date, helpful votes,
   certified-buyer mark. Reviewer names are NOT saved.
4. Writes one line per review, separated by "|", with a "#P" line before each product.
5. Waits 2-4 seconds between pages. It does not log in and does not try to get past any
   CAPTCHA or block: if Flipkart shows one, the script stops.

Install once:
    pip install playwright pandas
    python -m playwright install chromium

Try one product first:
    python data/collect_flipkart_reviews.py --test

Full run (resumable: rerun to continue where it stopped):
    python data/collect_flipkart_reviews.py --out data/flipkart_raw_recollected.psv
"""
import argparse, hashlib, os, random, re, sys, time
import pandas as pd
from playwright.sync_api import sync_playwright

BASE = os.path.dirname(os.path.abspath(__file__))
VIEWS = [("R", "MOST_RECENT", 8), ("N", "NEGATIVE_FIRST", 5), ("H", "MOST_HELPFUL", 8)]


def load_products(raw_csv):
    d = pd.read_csv(raw_csv, usecols=["product_index", "product_slug", "item_id"])
    p = d.drop_duplicates("product_index").sort_values("product_index")
    return [(int(r.product_index), r.product_slug, r.item_id) for r in p.itertuples()]


def review_url(slug, item_id, sort, page):
    return f"https://www.flipkart.com/{slug}/product-reviews/{item_id}?sortOrder={sort}&page={page}"


def clean(s):
    return re.sub(r"\s+", " ", (s or "").replace("|", "/")).strip()


# Runs inside the page: finds each review card by its "Certified Buyer" / rating block and
# returns plain text pieces. Flipkart changes its class names often, so this works from the
# page text instead of fixed class names.
EXTRACT_JS = r"""
() => {
  const out = [];
  const seen = new Set();
  const nodes = [...document.querySelectorAll('div')].filter(d => {
    const t = d.innerText || '';
    return /^[1-5](\.0)?\s*★?\s*\n/.test(t) && t.length < 3000 && t.split('\n').length >= 3;
  });
  for (const card of nodes) {
    // keep the smallest card that still contains the date/buyer line
    if ([...card.querySelectorAll('div')].some(c => nodes.includes(c))) continue;
    const box = card.parentElement && card.parentElement.parentElement ? card.parentElement.parentElement : card;
    const text = box.innerText || '';
    if (seen.has(text)) continue; seen.add(text);
    out.push(text.split('\n').map(s => s.trim()).filter(Boolean));
  }
  return out;
}
"""

DATE_RE = re.compile(r"^(\d+\s+(day|days|month|months|year|years|hour|hours|minute|minutes)\s+ago|[A-Z][a-z]{2},\s*\d{4}|Today|Yesterday)$")


def parse_card(lines):
    """Turn the text lines of one review card into a record, or None if it is not a review."""
    if not lines or not re.match(r"^[1-5](\.0)?\s*★?$", lines[0]):
        return None
    rating = int(lines[0][0])
    rest = [l for l in lines[1:] if l not in ("READ MORE", "Report Abuse", "Permalink", "★")]
    certified = int(any("Certified Buyer" in l for l in rest))
    date = next((l for l in rest if DATE_RE.match(l)), "")
    nums = [int(l.replace(",", "")) for l in rest if re.fullmatch(r"[\d,]+", l)]
    helpful = nums[0] if nums else 0
    body = [l for l in rest if l != date and "Certified Buyer" not in l and not re.fullmatch(r"[\d,]+", l)
            and not l.startswith(",")]            # ", City" location lines are skipped
    if not body:
        return None
    title = body[0]
    text = max(body[1:], key=len) if len(body) > 1 else body[0]
    # body lines just before "Certified Buyer" are usually the reviewer's name: drop the last short one
    rid = hashlib.md5(f"{rating}|{title}|{text}|{date}".encode()).hexdigest()[:8]
    return rid, rating, date, helpful, certified, clean(title), clean(text)


def already_done(out_path):
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path, encoding="utf-8"):
            if line.startswith("#P"):
                done.add(int(line[2:].split("|")[0]))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(BASE, "flipkart_raw.csv"), help="where the product list comes from")
    ap.add_argument("--out", default=os.path.join(BASE, "flipkart_raw_recollected.psv"))
    ap.add_argument("--test", action="store_true", help="one product, one page per view")
    a = ap.parse_args()

    products = load_products(a.raw)
    if a.test:
        products, a.out = products[:1], os.path.join(BASE, "flipkart_test.psv")
    done = set() if a.test else already_done(a.out)
    todo = [p for p in products if p[0] not in done]
    print(f"{len(products)} products, {len(todo)} still to collect -> {a.out}")

    with sync_playwright() as pw, open(a.out, "w" if a.test else "a", encoding="utf-8") as f:
        browser = pw.chromium.launch(headless=False)
        page = browser.new_page()
        for idx, slug, item in todo:
            f.write(f"#P{idx}|{slug}|{item}\n")
            n_prod = 0
            for tag, sort, max_pages in VIEWS:
                for pg in range(1, (1 if a.test else max_pages) + 1):
                    page.goto(review_url(slug, item, sort, pg), wait_until="domcontentloaded", timeout=60000)
                    page.wait_for_timeout(1500)
                    body = page.inner_text("body")[:2000].lower()
                    if "captcha" in body or "are you a human" in body or "access denied" in body:
                        print("Flipkart is showing a check page. Stopping here (not bypassing it). Rerun later.")
                        browser.close(); sys.exit(1)
                    recs = [r for r in (parse_card(c) for c in page.evaluate(EXTRACT_JS)) if r]
                    for r in recs:
                        f.write("|".join([r[0], str(r[1]), r[2], str(r[3]), str(r[4]), tag, r[5], r[6]]) + "\n")
                    f.flush()
                    n_prod += len(recs)
                    print(f"  P{idx} {sort:<14} page {pg}: {len(recs)} reviews")
                    time.sleep(random.uniform(2, 4))
                    if len(recs) < 10:
                        break                       # last page for this view
            print(f"P{idx} {slug[:50]}: {n_prod} reviews")
        browser.close()
    print("Done.")


if __name__ == "__main__":
    main()
