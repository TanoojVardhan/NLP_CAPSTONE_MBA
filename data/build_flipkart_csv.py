"""Clean the raw Flipkart review dump into data/flipkart_reviews.csv (run: python3 build_flipkart_csv.py data/flipkart_raw.csv data/flipkart_reviews.csv)."""
import sys, re, html, unicodedata, datetime as dt
import pandas as pd

RAW, OUT = sys.argv[1], sys.argv[2]
COLLECTED = dt.date(2026, 10, 14)

# index -> (display name, category), same order as the collection list
P = {
0:("boAt Airdopes 131 Gen 2","TWS Earbuds & Neckbands"),1:("Noise Buds VS102","TWS Earbuds & Neckbands"),
2:("realme Buds T300","TWS Earbuds & Neckbands"),3:("OnePlus Nord Buds 2R","TWS Earbuds & Neckbands"),
4:("Redmi Buds 5C","TWS Earbuds & Neckbands"),5:("OPPO Enco Air2i","TWS Earbuds & Neckbands"),
6:("boAt Rockerz 255 Pro Neckband","TWS Earbuds & Neckbands"),7:("realme Buds Wireless 3 Neckband","TWS Earbuds & Neckbands"),
8:("Zebronics Zeb-Thunder Headphones","Audio & Speakers"),9:("JBL Go 3 Speaker","Audio & Speakers"),
10:("boAt Stone 190 Pro Speaker","Audio & Speakers"),11:("boAt Stone 350 Pro Speaker","Audio & Speakers"),
12:("Fire-Boltt Phoenix Ultra Smartwatch","Smartwatches & Wearables"),13:("Noise ColorFit Pulse 3 Smartwatch","Smartwatches & Wearables"),
14:("boAt Wave Call Smartwatch","Smartwatches & Wearables"),15:("Mi 20000mAh 33W Power Bank","Power Banks & Charging"),
16:("Ambrane 10000mAh 22.5W Power Bank","Power Banks & Charging"),17:("Portronics 10000mAh Power Bank","Power Banks & Charging"),
18:("boAt 67W GaN Wall Charger","Power Banks & Charging"),19:("Logitech B170 Wireless Mouse","PC Peripherals & Storage"),
20:("Logitech M240 Wireless Mouse","PC Peripherals & Storage"),21:("Logitech C270 HD Webcam","PC Peripherals & Storage"),
22:("SanDisk Ultra Shift 64GB Pen Drive","PC Peripherals & Storage"),23:("Dell KB500 Wireless Keyboard","PC Peripherals & Storage"),
24:("Dell MS116 Wired Mouse","PC Peripherals & Storage"),25:("SanDisk Ultra Flair 64GB Pen Drive","PC Peripherals & Storage"),
26:("Mi 20000mAh 22.5W Power Bank","Power Banks & Charging"),27:("Portronics 20000mAh Power Bank","Power Banks & Charging"),
28:("realme Buds T310","TWS Earbuds & Neckbands"),29:("OPPO Enco Buds3 Pro","TWS Earbuds & Neckbands"),
30:("boAt Stone Arc Pro Speaker","Audio & Speakers"),31:("Noise Buds VS102 Plus","TWS Earbuds & Neckbands"),
32:("boAt Wave Smart Call Smartwatch","Smartwatches & Wearables"),33:("Logitech M170 Wireless Mouse","PC Peripherals & Storage"),
34:("Xiaomi 67W SonicCharge Charger","Power Banks & Charging"),35:("Ambrane 10000mAh 12W Power Bank","Power Banks & Charging"),
}

CATMAP = {"tws earbuds":"TWS Earbuds & Neckbands","bluetooth neckband":"TWS Earbuds & Neckbands","wireless headphones":"Audio & Speakers",
 "wired earphones":"Audio & Speakers","bluetooth speaker":"Audio & Speakers","soundbar":"Audio & Speakers","smartwatch":"Smartwatches & Wearables",
 "power bank 10000mah":"Power Banks & Charging","power bank 20000mah":"Power Banks & Charging","mobile charger":"Power Banks & Charging",
 "usb cable":"Power Banks & Charging","wireless mouse":"PC Peripherals & Storage","gaming mouse":"PC Peripherals & Storage",
 "keyboard":"PC Peripherals & Storage","webcam":"PC Peripherals & Storage","usb flash drive":"PC Peripherals & Storage"}
def to_date(s):
    s = s.strip()
    m = re.match(r"(\d+) (day|days|month|months|year|years) ago", s)
    if m:
        n, u = int(m.group(1)), m.group(2)
        days = n if u.startswith("day") else n*30 if u.startswith("month") else n*365
        return (COLLECTED - dt.timedelta(days=days)).isoformat(), True
    m = re.match(r"([A-Z][a-z]{2}), (\d{4})", s)
    if m:
        d = dt.datetime.strptime(f"{m.group(1)} {m.group(2)}", "%b %Y").date()
        return d.isoformat(), True          # month only -> 1st of month
    return None, True

raw = pd.read_csv(RAW, dtype=str, keep_default_na=False)
for _, r in raw.drop_duplicates("product_index").iterrows():
    if int(r.product_index) >= 100 and r.search_query:
        nm = " ".join(w.capitalize() if not w.isdigit() else w for w in r.product_slug.split("-")[:6])
        P[int(r.product_index)] = (nm + " [" + r.item_id[3:7] + "]", CATMAP[r.search_query])
rows = []
for r in raw.itertuples(index=False):
    rows.append(dict(raw_id=r.review_id, p=int(r.product_index), itm=r.item_id, rating=int(r.rating), date_raw=r.date_text,
                     helpful=int(r.helpful_votes), verified=bool(int(r.certified)),
                     sample_type={"R":"recent","N":"negative_first","H":"helpful"}[r.view_tag],
                     title=html.unescape(r.title).strip(), text=html.unescape(r.text).strip()))
df = pd.DataFrame(rows)
log = {"raw rows": len(df)}

# 1) exact duplicate records (same product + same id, e.g. a review seen in both sorts)
df = df.sort_values("sample_type", ascending=False, kind="stable")   # keep recent > negative_first > helpful
df = df.drop_duplicates(["p", "raw_id"]); log["after id dedup"] = len(df)

# 2) not verified purchase
df = df[df.verified]; log["verified only"] = len(df)

def letters(s): return re.sub(r"[^A-Za-z]", "", s)
def nonlatin_share(s):
    L = [c for c in s if c.isalpha()]
    return 0 if not L else sum(1 for c in L if ord(c) > 0x024F) / len(L)
def words(s): return re.findall(r"[A-Za-z]+", s)

def garbage_reason(t):
    w = words(t)
    if len(letters(t)) < 8: return "too short / emoji-only"
    if len(w) < 3: return "fewer than 3 words"
    if nonlatin_share(t) > 0.3: return "non-Latin script"
    if len(set(x.lower() for x in w)) == 1: return "one repeated word"
    if re.search(r"(.)\1{5,}", t): return "repeated characters"
    vow = sum(1 for c in letters(t).lower() if c in "aeiou") / max(1, len(letters(t)))
    if vow < 0.15 or vow > 0.75: return "gibberish (vowel ratio)"
    if re.search(r"https?://|www\.|\b\d{10}\b", t): return "link / phone number"
    return None

df["garbage"] = df.text.map(garbage_reason)
log["removed as garbage"] = int(df.garbage.notna().sum())
log["garbage breakdown"] = df.garbage.value_counts().to_dict()
df = df[df.garbage.isna()].drop(columns="garbage")

# 3) duplicate text: same normalised text inside a product, or copy-pasted across products
df["norm"] = df.text.str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
before = len(df)
df = df.drop_duplicates(["p", "norm"]); log["within-product text duplicates removed"] = before - len(df)
before = len(df)
dup_global = df.duplicated("norm", keep=False)
df = df[~dup_global | ~df.duplicated("norm", keep="first")]
log["cross-product copy-paste removed"] = before - len(df)
cnt = df.groupby("p").size()
small = cnt[cnt < 30].index.tolist()
log["products dropped (fewer than 30 clean reviews)"] = [P[i][0] for i in small]
df = df[~df.p.isin(small)]
df = df.sort_values(["p", "date_raw"], kind="stable").reset_index(drop=True)

df["product_name"] = df.p.map(lambda i: P[i][0]); df["category"] = df.p.map(lambda i: P[i][1])
df["review_date"] = df.date_raw.map(lambda s: to_date(s)[0])
df["review_id"] = "FK" + df.p.astype(str).str.zfill(2) + "_" + df.raw_id
out = pd.DataFrame({
    "review_id": df.review_id, "platform": "Flipkart", "asin": df.itm, "product_name": df.product_name,
    "category": df.category, "rating": df.rating, "review_title": df.title, "review_text": df.text,
    "is_verified": True, "review_date": df.review_date, "helpful_votes": df.helpful,
    "sample_type": df.sample_type,
    "source_url": "https://www.flipkart.com/product-reviews/" + df.itm})
out.to_csv(OUT, index=False)
log["final rows"] = len(out); log["products"] = int(out.product_name.nunique()); log["collected_on"] = str(COLLECTED)
import json; json.dump(log, open(OUT.replace("flipkart_reviews.csv", "flipkart_cleaning_log.json"), "w"), indent=1)
for k, v in log.items(): print(f"{k}: {v}")
print(out.rating.value_counts().sort_index().to_dict())
print(out.groupby("category").size().to_dict()); print(out.sample_type.value_counts().to_dict())
print("products:", out.product_name.nunique(), "| min/median/max per product:", out.groupby("product_name").size().agg(["min","median","max"]).to_dict())
