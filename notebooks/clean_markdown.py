"""Notes for ABSA_Capstone_Flipkart.ipynb.

apply(cells) removes every markdown cell and rebuilds the notes around the code cells:
  intro and data notes at the top -> for each code cell: section heading, a note on what it
  does and how, the code, then the interpretation of its result -> key insights, limitations
  and conclusion at the end.
Code cells are matched by the start of their first line, so the notes follow the code even if
the order changes. Used by build_notebook_flipkart.py and for updating the notebook in place.
"""

INTRO = """# E-Commerce Sentiment Analysis & Defect Diagnostics
**Aspect-Based Sentiment Analysis (ABSA) of Flipkart consumer-electronics reviews**
24INFS7101 NLP & Text Analytics in Business, MBA Capstone (A2)

## The business problem
A star rating tells a seller that a product is "3 out of 5", but not **what** is wrong with it. A 3-star average can hide a weak battery, an uncomfortable fit or a refused warranty claim, and each needs a different team to fix it.

## What this notebook does
It reads the text of about 10,000 customer reviews and answers: **which part of the product (aspect) are customers unhappy with, and in which product category?**

For example, *"Battery is great, but customer service was terrible"* is positive about **Battery** and negative about **Customer Service**. A star rating cannot show that split; aspect-based sentiment analysis can.

## The steps, in order
1. Load the data and check it is clean.
2. Clean the text and turn it into numbers (Bag-of-Words, TF-IDF).
3. Tag each clause with one of 10 product aspects and score its sentiment with VADER.
4. Check how far the automatic labels can be trusted (hand check and comparison with star ratings).
5. **Defect diagnostics**: % negative for each aspect, by category and by product. *This is the main result.*
6. Extra text-mining views: named entities, clustering, trend over time, summaries, word embeddings, topic modelling.
7. A machine-learning classifier that predicts positive or negative reviews.
8. Word clouds and an interactive dashboard, then key insights and limitations.
"""

DATA = """## About the data
| Item | Detail |
|---|---|
| Source | Flipkart.com, public product review pages (no login used, no CAPTCHA bypassed, reviewer names not saved) |
| Collected | 14 October 2026, with a script run in a browser session (about 3 page requests per second) |
| Products | 92 consumer-electronics products in 5 categories: TWS earbuds and neckbands, audio and speakers, power banks and chargers, PC peripherals and storage, smartwatches and wearables |
| How reviews were chosen | For each product, three views of the review list: **most recent** (up to 8 pages), **negative first** (up to 5 pages) and **most helpful** (up to 8 pages), 10 reviews per page |
| Size | 23,046 raw reviews, cleaned to **10,004** verified-purchase reviews |
| File | `data/flipkart_reviews.csv` (made from `data/flipkart_raw.csv` by `data/build_flipkart_csv.py`) |

**Columns:** `review_id`, `platform`, `asin` (Flipkart item ID), `product_name`, `category`, `rating` (1-5 stars), `review_title`, `review_text`, `is_verified`, `review_date` (approximate, because Flipkart shows dates like "7 days ago"), `helpful_votes`, `sample_type` (which of the three views the review came from) and `source_url`.

**How the raw data was cleaned** (counts in `data/flipkart_cleaning_log.json`):
| Step | Rows removed | Rows left |
|---|---|---|
| Raw reviews collected | - | 23,046 |
| Same review seen in two views | 4,400 | 18,646 |
| Not a verified purchase | 4 | 18,642 |
| Junk text (under 3 words, emoji-only, repeated characters) | 6,091 | 12,551 |
| Same text twice within one product | 211 | 12,340 |
| Same text copied across different products | 2,168 | 10,172 |
| Products with fewer than 30 reviews (16 products) | 168 | **10,004** |

**Important:** the sample is **not random**. We added the negative-first view on purpose so there would be enough complaints to study, so overall percentages describe this sample, not all Flipkart buyers. Compare aspects with each other rather than quoting an overall rate.

**Ethics:** only public review text, rating, date, verified flag and helpful votes were kept; no reviewer names or profiles. The data is used for coursework only.
"""

# key = start of the code cell's first line -> (section heading or None, note before, interpretation after or None)
NOTES = [
("!pip install", "## 1. Setup",
"""**What this cell does:** installs every library the notebook needs and downloads their language data.

**How:** `!pip install` installs pandas (tables), NLTK (text tools and VADER), spaCy (language model), scikit-learn (machine learning), Gensim (Word2Vec and LDA), Matplotlib/Seaborn/Plotly (charts) and WordCloud. The next two lines download the spaCy English model and NLTK's stop-word list, sentence splitter and VADER dictionary. Run it once; after a first-time install, restart the kernel.""",
"""**Result:** "already satisfied / up-to-date" means everything is installed and the notebook is ready to run."""),

("import os", "## 2. Load and check the data",
"""**What this cell does:** loads the cleaned review file into a table called `df`.

**How:** it sets the folder paths (`data/` for input, `outputs_flipkart/` for results), reads `flipkart_reviews.csv` with pandas, converts `rating` to a number and `review_date` to a date, then prints the table size, number of products and categories, the date range and the first three rows.""",
"""**What the result shows:** 10,004 reviews with 14 columns, 92 products in 5 categories, dated from January 2016 to October 2026. Each row is one review with its product, category, star rating, title, text and which view it came from."""),

("from IPython.display import display", None,
"""**What this cell does:** proves the data is clean and shows how it is spread across views, categories and products.

**How:** `assert` statements stop the notebook if there is any duplicate review ID, any duplicate review text (after lower-casing and removing punctuation) or any review under 3 words. It then prints the cleaning log, the number of reviews from each view and the number of reviews per category and per product.""",
"""**What the result shows:**
- All checks pass: no duplicates, shortest review 3 words, median 9 words, 100% verified buyers.
- Views: 4,011 recent, 3,971 most helpful, 2,022 negative-first.
- Categories: Audio & Speakers 2,762, TWS Earbuds 2,388, Power Banks 2,279, PC Peripherals 2,022, Smartwatches 553.
- Products range from 30 reviews (the minimum we kept) to 210 (OPPO Enco Buds3 Pro), so every product has enough reviews to compare.
- Smartwatches are the smallest group (6 products), so their results are the least certain."""),

("# Rating distribution", None,
"""**What this cell does:** summarises the star ratings before any text analysis.

**How:** `describe()` gives the average, spread and quartiles of `rating`; `value_counts()` counts how many reviews have each star level.""",
"""**What the result shows:** the average is 3.43 stars (median 4). Ratings are **polarised**: 4,435 five-star and 2,908 one-star reviews, but only 582 three-star. Customers tend to love or hate a product. The large number of 1-star reviews comes partly from the negative-first view."""),

("import matplotlib.pyplot as plt", None,
"""**What this cell does:** draws the star-rating distribution as a bar chart and saves it to `outputs_flipkart/figures/01_rating_distribution.png`.

**How:** Seaborn's `countplot` counts reviews per star level.""",
"""**What the chart shows:** a "U" shape: tall bars at 5 and 1 star, short bars at 2, 3 and 4. The star rating alone cannot tell us *why* the 1-star customers are unhappy, which is what the rest of the notebook works out."""),

("import re", "## 3. Text cleaning",
"""**What this cell does:** cleans the review text and makes a word list for each review.

**How:**
- `clean_text()` removes web links (`http...`), replaces any character that is not a letter, number or basic punctuation (`. , ! ? '`) with a space, and collapses repeated spaces. Punctuation is kept because VADER uses it ("good!!!" is stronger than "good"). Numbers are kept because they carry meaning ("stopped after 2 days", "10000mAh").
- `lemmatize_tokens()` lower-cases the text, splits it into words with spaCy, turns each word into its base form (lemmatisation: "charging", "charged" -> "charge"), and keeps only alphabetic words longer than 2 letters that are not stop-words ("the", "is", "and").
- Results are stored in new columns: `clean_text` (used for sentiment and the classifier), `lemmas` (used for Word2Vec and topic modelling) and `token_count`.""",
"""**What the result shows:** the first three reviews before and after cleaning. These reviews were already plain text, so `clean_text` looks the same; `token_count` shows how many useful words were left after removing stop-words (5-6 words each)."""),

("from sklearn.feature_extraction.text import CountVectorizer", "## 4. Word importance: Bag-of-Words vs TF-IDF",
"""**What this cell does:** compares two ways of turning text into numbers and plots the top 20 words for each.

**How:**
- **Bag-of-Words (BoW)** with `CountVectorizer`: counts how often each word appears across all reviews.
- **TF-IDF** (Term Frequency - Inverse Document Frequency) with `TfidfVectorizer`: gives a word a high weight if it is frequent in a review but rare across all reviews, and a low weight if it appears everywhere.
- Both skip English stop-words; the chart is saved as `08_bow_vs_tfidf.png`.""",
"""**What the result shows:** both lists are led by general words (good, product, quality, sound), because short reviews like "good product" are very common. TF-IDF moves words that point to a problem higher up: "working" rises from 7th to 5th and "bad" from 15th to 12th, while very common words lose weight. This is why TF-IDF is used later for the classifier, clustering and summaries. The frequent words (sound, battery, bass, charging, price, backup) also confirm that our 10 aspects cover what customers talk about."""),

("ASPECT_PATTERNS = {", "## 5. Aspect tagging and sentiment scoring",
"""**What this cell does:** the core of the project. It finds which product aspect each part of a review talks about and whether the customer is positive, neutral or negative about it.

**How:**
1. **Aspect keyword lists** (`ASPECT_PATTERNS`): 10 aspects, each with a list of whole words, e.g. Battery & Charging = battery, charge, charging, backup, mAh...; Customer Service = service, support, replacement, warranty, refund, return... Whole-word matching stops "ear" matching inside "earbuds".
2. **Sentence and clause splitting:** each review is split into sentences (NLTK `sent_tokenize`), and each sentence into clauses at contrast words ("but", "however", "although", "though", "while", "yet", "whereas", "except") and semicolons. So "battery is great but service is bad" becomes two clauses with their own scores.
3. **Sentiment with VADER** (Valence Aware Dictionary and sEntiment Reasoner): a ready-made, rule-based scorer for short informal text. It gives each clause a compound score from -1 to +1. Score >= +0.05 is **Positive**, <= -0.05 is **Negative**, otherwise **Neutral**.
4. Every (clause, aspect) pair becomes one row in a new table, `aspect_df`.""",
"""**What the result shows:** **16,566 aspect mentions** across all 10 aspects. The table shows the first rows: e.g. "charging good comfortable for ear i like" gives two mentions (Battery & Charging and Comfort & Fit), both Positive (score +0.83), while "Product quality not good..." is tagged Build & Durability and Negative."""),

("# Spot-check", None,
"""**What this cell does:** prints 40 randomly chosen tagged clauses so a person can judge by eye whether the tags make sense.

**How:** `sample(40, random_state=7)` picks 40 rows from `aspect_df` (the fixed seed gives the same 40 every run) and prints each aspect next to its text.""",
"""**What the result shows:** most tags are right ("The headset get auto disconnected frequently" -> Connectivity; "Not value for money" -> Value for Money). Some are wrong or doubtful, e.g. "Awesome mouse.. No sounds on click" is tagged Sound & Audio Quality, and a clause that lists many features at once ("Bass good... Design good... Value of money 3/5") gives every aspect the same score. This shows the keyword method is useful but not perfect, which is why the next cell measures its accuracy."""),

("audit_path =", "## 6. How far can we trust the labels?",
"""**What this cell does:** measures how often a "Negative" label is really a complaint.

**How:** we took a random sample of 80 mentions that VADER labelled Negative and read each one by hand, marking `genuine_complaint` Yes/No (saved in `data/negative_mentions_manual_audit_flipkart.csv`). The cell counts the Yes answers and calculates a 95% confidence interval (Wilson method), which gives the range the true value is likely to fall in.""",
"""**What the result shows:** **53 of 80 (66%)** Negative labels are genuine complaints about the tagged aspect, with a 95% range of 55%-76%. About one in three is wrong, usually praise misread by VADER ("no problem with battery backup" scored negative because of the word "problem") or a complaint filed under the wrong aspect. By aspect, Build & Durability (17 of 18) and Value for Money (5 of 5) are very reliable; Performance & Speed (2 of 8) and Display & Design (1 of 5) are not. **So the % negative figures later are upper limits: they show where to look, not exact defect rates.**"""),

("# Review-level overall sentiment", None,
"""**What this cell does:** checks VADER against the star ratings for whole reviews.

**How:** VADER scores each full review. Star ratings are converted to the same three labels (1-2 stars = Negative, 3 = Neutral, 4-5 = Positive). The cell prints how often the two agree and a cross-table (rows = star label, columns = VADER label).""",
"""**What the result shows:** VADER agrees with the stars on **77.6%** of reviews. Of the 3,319 one- and two-star reviews, VADER calls 2,098 Negative, 902 Neutral and 319 Positive. Most of the misses are short factual complaints with no emotional words ("stopped working after a month"), which VADER reads as neutral."""),

("fig, axes = plt.subplots(1, 2", None,
"""**What this cell does:** puts VADER's labels and the star-based labels side by side in a chart, and prints the numbers that matter for finding complaints.

**How:** two bar charts (saved as `02_sentiment_vs_rating.png`), then the always-"Positive" baseline, negative recall and negative precision.
- **Baseline:** the score you would get by always saying "Positive".
- **Recall:** of all the truly negative reviews, how many VADER caught.
- **Precision:** of all the reviews VADER flagged as negative, how many really were negative.""",
"""**What the result shows:**
- VADER's 77.6% agreement beats the 61.0% baseline, so it is genuinely useful.
- **Recall 63%:** it catches 2,098 of 3,319 low-rated reviews.
- **Precision 85%:** 2,098 of the 2,482 reviews it flags are really low-rated.
- So VADER is a reliable but incomplete complaint detector: when it says "negative" it is usually right, but it misses about a third of complaints."""),

("agg = (aspect_df", "## 7. Defect diagnostics (main result)",
"""**What this cell does:** calculates, for every product category and aspect, what share of mentions are Positive, Neutral and Negative.

**How:** groups `aspect_df` by category and aspect, turns the label counts into percentages, adds the number of mentions, and lists the 12 combinations with the highest % negative (only those with at least 3 mentions).""",
"""**What the result shows:** the biggest problem areas are **Customer Service in TWS earbuds (46.8% negative)**, **Build & Durability in smartwatches (40.4%)**, **Customer Service in PC peripherals (37.5%)** and **Comfort & Fit in TWS earbuds (37.4%)**. Customer Service appears in the list for every category, which points to a common after-sales problem rather than a single product. Some rows rest on few mentions (Smartwatch Customer Service has only 14), so the next cells add confidence intervals and a minimum size."""),

("os.makedirs(OUT_DIR", None,
"""**What this cell does:** saves all result tables and shows the overall picture for each aspect.

**How:** writes `sentence_aspect_sentiment.csv`, `review_level_sentiment.csv`, `aspect_summary_by_category.csv` and `aspect_summary_by_product.csv` to `outputs_flipkart/`, then calculates for each aspect (all categories together) the counts, % negative, % positive, the number of different reviews, and a 95% Wilson confidence interval for % negative (saved as `aspect_summary_overall.csv`).""",
"""**What the result shows:** across 16,566 mentions, **20.4% are negative**. Ranked by % negative:
| Aspect | % negative | 95% range | Mentions |
|---|---|---|---|
| Customer Service | 37.7 | 33.5-42.1 | 480 |
| Comfort & Fit | 31.3 | 27.9-34.9 | 671 |
| Build & Durability | 29.2 | 27.5-30.9 | 2,806 |
| Battery & Charging | 23.2 | 21.7-24.8 | 2,830 |
| Connectivity | 20.1 | 17.5-23.1 | 790 |
| Performance & Speed | 17.1 | 15.3-19.0 | 1,592 |
| Sound & Audio Quality | 17.0 | 15.8-18.3 | 3,490 |
| Delivery & Packaging | 15.9 | 12.6-19.8 | 396 |
| Value for Money | 13.5 | 12.2-14.8 | 2,533 |
| Display & Design | 8.6 | 7.0-10.5 | 978 |

The top three ranges sit clearly above the 20.4% average, so they are real problem areas, not chance. Build & Durability has the largest number of negative mentions (819). **Price is not the problem**: Value for Money is 76.9% positive."""),

("pivot_neg =", None,
"""**What this cell does:** draws the **defect heatmap**, the key picture of the project.

**How:** rows are aspects, columns are categories; each cell shows % negative and the number of mentions (n). Darker red = more negative. Cells with fewer than 3 mentions are left blank because a percentage on 1-2 mentions means nothing. Saved as `04_defect_heatmap.png`.""",
"""**How to read it:** look for the darkest cells with a large n. The Customer Service row is dark in every category; Build & Durability is dark for PC peripherals (n = 652) and smartwatches; Comfort & Fit is dark for earbuds (n = 310). Light cells such as Display & Design and Value for Money are where customers are mostly satisfied. The PC peripherals Sound cell (29%) mostly reflects click and scroll noise from mice, not audio quality."""),

("strong =", None,
"""**What this cell does:** lists the 10 strongest problem areas that have at least 10 mentions.

**How:** filters the category-by-aspect table to rows with 10 or more mentions and sorts by % negative.""",
"""**What the result shows:** the 10 most reliable hot-spots. In plain words:
1. **Earbud buyers are unhappy with after-sales service** (46.8%) **and with fit** (37.4%, 310 mentions).
2. **Smartwatches break or wear out** (Build 40.4%) and have **battery complaints** (31.2%).
3. **PC peripherals (mice, keyboards) fail early** (Build 34.7%, the largest group with 652 mentions) and their **service** is poor (37.5%).
4. **Power banks** have service (32.5%) and connectivity/cable (33.3%, only 33 mentions) complaints."""),

("plt.figure(figsize=(8, 5))", None,
"""**What this cell does:** shows how often each aspect is talked about.

**How:** counts rows per aspect in `aspect_df` and plots them as horizontal bars (saved as `03_aspect_frequency.png`).""",
"""**What the chart shows:** customers talk most about **Sound & Audio Quality** (3,490 mentions), **Battery & Charging** (2,830), **Build & Durability** (2,806) and **Value for Money** (2,533); they rarely mention Delivery (396) or Customer Service (480). So Customer Service is mentioned rarely, but when it is, it is mostly negative: a small group with a big problem."""),

("prod_summary =", None,
"""**What this cell does:** compares individual products.

**How:** for each product it calculates the average star rating, number of reviews, number of aspect mentions and % negative mentions, and sorts by % negative. A scatter plot (saved as `07_product_rating_volume.png`) shows average rating against number of reviews.""",
"""**What the result shows:**
- **Weakest:** Baot Wireless Earbuds B5 neckband (2.00 stars, 56.8% negative, only 37 mentions), Boat Bassheads 300c (2.27 stars, 46.8%), F Ferons Neo Buds 3 (39.2%) and Zebronics Glassy 10 mouse (39.1%), mostly lesser-known brands.
- **Strongest:** Beetel 20000mAh power bank (4.84 stars, 4.3% negative), OPPO Enco Air2i (9.3%), Logitech M240 mouse (10.2%) and Mivi Fort H350 soundbar (10.9%).
- Stars and text agree: the lowest-rated products have the most negative aspect mentions. Products with under 60 mentions should be read with care."""),

("aspect_df = aspect_df.merge", None,
"""**What this cell does:** checks whether our sampling method changes the answer.

**How:** repeats the % negative calculation separately for each view (recent, negative-first, most helpful), then compares each aspect's % negative using all reviews against using only the recent view (the closest to a normal sample).""",
"""**What the result shows:**
- The view matters a lot for the **overall** level: 8.9% negative in the helpful view, 24.3% in the recent view, 55.1% in the negative-first view. So "20.4% negative" is not a market-wide number.
- But the **order of aspects stays the same** with only recent reviews: Customer Service (42.8%), Comfort & Fit (38.5%) and Build & Durability (31.1%) are still the top three.
- **So the main finding does not depend on how we sampled.** Delivery & Packaging rises to 34.6% in the recent view, a possible new issue worth watching."""),

("nlp_ner =", "## 8. Extra text-mining views\n### 8.1 Named entities",
"""**What this cell does:** finds names of brands, products, places and organisations in the reviews (Named Entity Recognition, NER).

**How:** runs spaCy's English model on every review (first 1,000 characters) and keeps entities labelled ORG (organisation), PRODUCT, GPE (place), PERSON and NORP (nationality/group), then counts them.""",
"""**What the result shows:** 3,134 entities, but many are wrong: "Bass" (252), "ANC" (87), "Superb" (56) and even emojis are labelled as organisations or people. Real names do appear (Flipkart 153, Logitech 36, JBL 23). **Interpretation:** an off-the-shelf NER model is too noisy for short, informal reviews; competitor tracking would need a custom brand list."""),

("if len(ent_df):", None,
"""**What this cell does:** plots the 12 most frequent entities (saved as `09_named_entities.png`).

**How:** a horizontal bar chart of entity counts. (The "Glyph missing" warnings only mean the chart font cannot draw emojis; they do not affect results.)""",
"""**What the chart shows:** the same picture as above: product jargon and emojis dominate, with only a few real brand names. NER adds little to the defect analysis on this data."""),

("import numpy as np", "### 8.2 Clustering",
"""**What this cell does:** groups similar reviews together without using any labels (K-Means clustering), to see whether natural themes appear.

**How:** turns each cleaned review into a TF-IDF vector (300 words), runs K-Means with 5 clusters (`random_state=42` for repeatable results), then shows the cluster sizes, how each cluster splits across categories, and the top words of each cluster.""",
"""**What the result shows:**
| Cluster | Reviews | Top words | Meaning |
|---|---|---|---|
| 0 | 687 | money, value, good, waste, worth | value-for-money remarks (mixed) |
| 1 | 539 | bad, quality, battery, don't, buy | general complaints |
| 2 | 1,748 | good, quality, sound, price, nice | general praise |
| 3 | 829 | **working, month, stopped, days, properly, mouse** | **early failure** |
| 4 | 6,201 | product, quality, sound, best, battery | general praise, all categories |

Clusters do not match product categories, because short reviews like "good product" look alike for any product. The one clearly meaningful cluster is **cluster 3: "stopped working within days or months"**, and PC peripherals are over-represented in it (361 of 829). **This supports the Build & Durability finding.**"""),

("trend =", "### 8.3 Trend over time",
"""**What this cell does:** shows how review volume and average rating change month by month.

**How:** groups reviews by month of `review_date`, then plots the number of reviews (grey bars) and the average star rating (red line) on two axes (saved as `10_trend_over_time.png`).""",
"""**What the chart shows:** most reviews are from 2025-2026 (57% from 2026 alone), because the "most recent" view is part of the sample. The average rating appears to drop sharply in 2025-2026, but this is mainly because the negative-first view collected many recent complaints, and older months have very few reviews. Flipkart shows only approximate dates ("7 days ago", "Mon, 2024"). **Interpretation:** this chart shows how trend monitoring would work; it should not be read as proof that product quality fell."""),

("def extractive_summary", "### 8.4 Summaries",
"""**What this cell does:** writes a one-sentence "what is wrong" summary for each category (extractive summarisation).

**How:** takes all sentences from 1-3 star reviews in a category, turns them into TF-IDF vectors, finds the average vector (the "centre" of the complaints) and picks the sentence closest to it as the most representative one.""",
"""**What the result shows:** Audio: "Not a good product sound quality is not good"; PC peripherals: "Product is not working and very bad quality"; Power banks: "fast charging is not working"; Smartwatches: "Bad product watch"; Earbuds: "very bad product, sound is not good". The summaries are accurate but general, because negative reviews are short. They match the main complaints (sound, products not working, charging)."""),

("from gensim.models import Word2Vec", "### 8.5 Word similarity (Word2Vec)",
"""**What this cell does:** learns which words are used in similar ways in these reviews (word embeddings).

**How:** trains a Word2Vec model (Gensim) on the lemmatised word lists: each word becomes a 50-number vector, and words that appear in similar contexts get similar vectors. It then prints the 5 most similar words for battery, sound, quality, price and service.""",
"""**What the result shows:** the model learned sensible meanings from our own data: **battery** ~ hrs, hour, backup (and the misspelling "bettery"); **sound** ~ bass, crisp, vocal, audio; **price** ~ budget, offer, cost (and "prize"); **service** ~ seller, visit, refuse, replacement. The "service" neighbours (refuse, replacement) match the warranty/return complaints. The vocabulary is small (2,515 words), so embeddings mainly confirm the aspect keyword lists rather than add new findings."""),

("from sklearn.decomposition import PCA", None,
"""**What this cell does:** draws the Word2Vec vectors of 16 key words on a 2D map (saved as `11_word2vec_pca.png`).

**How:** PCA (Principal Component Analysis) squeezes each 50-number vector into 2 numbers so the words can be plotted; words close together are used in similar ways.""",
"""**What the chart shows:** "sound", "quality", "build", "mic", "good" sit together (general product quality talk); "warranty" and "service" sit together far to the right (after-sales talk); "connect" and "charge" sit together (power and pairing issues); "fit" and "comfort" sit apart on the left. The groups match our aspect categories."""),

("from gensim import corpora", "### 8.6 Complaint topics (LDA)",
"""**What this cell does:** finds hidden complaint themes in the 1-2 star reviews (topic modelling with LDA, Latent Dirichlet Allocation).

**How:** builds a dictionary from the lemmatised words of the 3,319 low-rated reviews (dropping very rare and very common words), then trains a 4-topic LDA model (20 passes, `random_state=42`). Each topic is a list of words with weights.""",
"""**What the result shows:**
- **Topic 0, general dissatisfaction:** bad, product, quality, buy, money, waste.
- **Topic 1, mouse and device problems:** mouse, use, time, issue, problem.
- **Topic 2, early failure:** work, month, day, stop, properly, within ("stopped working within a month").
- **Topic 3, charging and power banks:** charge, battery, power, bank, phone, slow.

The strongest topic (Topic 2) is the same **early-failure** theme found by clustering, which again supports the Build & Durability result."""),

("topic_rows =", None,
"""**What this cell does:** saves the topic words to `outputs_flipkart/lda_topics.csv` and plots the top 8 words of each topic (saved as `06_lda_topics.png`).

**How:** one bar chart per topic, longer bar = more important word.""",
"""**What the chart shows:** Topic 2 has by far the strongest top word ("work", weight 0.20), so "not working" is the single most common complaint idea in low-rated reviews."""),

("from sklearn.model_selection", "## 9. Machine-learning classifier",
"""**What this cell does:** trains two machine-learning models to predict from the text alone whether a review is positive or negative, and compares them with simple baselines.

**How:**
1. Uses the 9,422 reviews with 1, 2, 4 or 5 stars (1-2 = Negative, 4-5 = Positive; 3-star reviews are left out as unclear).
2. Turns text into numbers with TF-IDF (top 500 words and 2-word phrases, negation words like "not" kept).
3. Splits the data: 75% to train (7,066 reviews), 25% to test (2,356). TF-IDF is inside a pipeline, so it learns only from the training part (no information leaks from the test set).
4. Trains **Logistic Regression** (learns a weight per word; balanced for the smaller negative class) and **Naive Bayes** (a simpler model based on word frequencies).
5. Reports accuracy, precision, recall, F1 and the confusion matrix, and repeats the test with 3-fold cross-validation (train and test 3 times on different parts) to make sure the score is stable.""",
"""**What the result shows:**
| Model | Accuracy | Negative recall | Macro-F1 | Cross-validated accuracy |
|---|---|---|---|---|
| Always say "Positive" (baseline) | 0.65 | 0.00 | - | - |
| **Logistic Regression** | **0.94** | **0.93** | 0.93 | 0.94 |
| Naive Bayes | 0.92 | 0.82 | 0.91 | 0.91 |

- **Logistic Regression** correctly finds 774 of 830 negative reviews in the test set (93%) and is right 90% of the time when it flags one. It wrongly flags 87 of 1,526 positive reviews.
- **Naive Bayes** is more cautious: when it flags a review it is right 94% of the time, but it misses 148 complaints (finds 82%).
- Both are far above the 0.65 baseline, and Logistic Regression finds more complaints than VADER (93% vs 63%). Cross-validation gives the same scores, so the results are stable.
- **Limitation:** the labels come from stars, so the model learns to predict stars from text. It is good for sorting new reviews, but it does not say *which aspect* failed; the aspect analysis does that."""),

("# Which TF-IDF terms", None,
"""**What this cell does:** shows which words the Logistic Regression model links most strongly to positive and negative reviews.

**How:** trains Logistic Regression on all 9,422 reviews and lists the 15 words/phrases with the largest positive and largest negative weights (saved as `top_tfidf_terms.csv`).""",
"""**What the result shows:** positive words are praise (good, best, awesome, nice, amazing, excellent, worth, value money). Negative words include **"month", "return", "stopped", "slow"** besides obvious ones (not, poor, worst, bad, "don't buy", waste). These match the defect findings: products stopping within a month and problems with returns."""),

("from wordcloud", "## 10. Word clouds and dashboard",
"""**What this cell does:** draws two word clouds: one for positive reviews (4-5 stars, green) and one for negative reviews (1-2 stars, red). Saved as `05_wordclouds.png`.

**How:** joins the cleaned text of each group and lets WordCloud size each word and common word pair by how often it appears. Common English stop-words ("the", "and", "it") and very general words ("product", "use") are excluded, but negation words ("not", "no", "never") are kept so phrases like "not working" stay visible.""",
"""**What the chart shows:** positive reviews are led by phrases such as "sound quality", "battery backup", "value for money" and "build quality"; negative reviews by "not working", "stopped working", "poor quality", "waste of money" and "don't buy". It is a quick visual summary of the same findings (early failure, quality and service complaints, with price seen as good value), useful for a presentation."""),

("import plotly.graph_objects", None,
"""**What this cell does:** builds an interactive dashboard and saves it as `outputs_flipkart/dashboard.html`.

**How:** Plotly combines four charts on one page: star-rating distribution, aspect mention frequency, the defect heatmap and the 15 lowest-rated products. You can hover to see exact values.""",
"""**What the result shows:** the file opens in any web browser, so a manager can explore the main results without running Python."""),
]

INSIGHTS = """## Key insights (Flipkart: 10,004 reviews, 92 products, 5 categories)

1. **Where the defects are.** Of 16,566 aspect mentions, 20.4% are negative. The highest negative shares are **Customer Service (37.7%)**, **Comfort & Fit (31.3%)** and **Build & Durability (29.2%)**, then Battery & Charging (23.2%). Display & Design (8.6%) and Value for Money (13.5%) are the safest.
2. **Category hot-spots (at least 10 mentions).** TWS earbuds: Customer Service 46.8% and Comfort & Fit 37.4% negative. Smartwatches: Build & Durability 40.4%. PC peripherals: Build & Durability 34.7% (652 mentions, the largest hot-spot) and Customer Service 37.5%.
3. **The main complaint is early failure.** The LDA topic model, the K-Means "stopped working" cluster (829 reviews) and the classifier's negative words ("month", "stopped", "return") all point to products that stop working within days or months.
4. **Sampling changes the level, not the ranking.** The negative share is 8.9% in the "most helpful" view, 24.3% in "most recent" and 55.1% in "negative first", so 20.4% is not a market-wide rate. Using only recent reviews, the top three aspects stay the same (Customer Service 42.8%, Comfort & Fit 38.5%, Build 31.1%).
5. **How far to trust the labels.** In a random sample of 80 VADER-Negative mentions, 53 (66%, 95% range 55%-76%) were genuine complaints. Negative shares are therefore upper limits, not exact defect rates.
6. **Predicting sentiment.** VADER agrees with star-based sentiment on 77.6% of reviews (always-Positive baseline 61.0%). Logistic Regression reaches 0.94 accuracy and finds 93% of negative reviews (3-fold cross-validation 0.94 / 0.93), ahead of Naive Bayes (0.92 / 0.82).
7. **Product spread.** Average ratings range from 2.0 to 4.8 across the 92 products; the weakest are lesser-known brands (a neckband at 2.0, a Type-C earphone at 2.3, a 10000mAh power bank at 2.4), while OPPO, Logitech, Mivi and Beetel products are the strongest.

### Recommendations
| Finding | What to do | What to track |
|---|---|---|
| Build & Durability: early failure, worst for PC peripherals and smartwatches | Longer life and quality testing before sale, starting with the lowest-rated listings | Failure complaints per 1,000 units in the first 6 months |
| Customer Service: highest negative share in every category | Fix the warranty, replacement and return process | % negative Customer Service mentions per month; days to resolve a claim |
| Comfort & Fit: 37.4% negative for earbuds | More ear-tip sizes; state fit limits in the listing | Comfort & Fit negative share for earbuds |
| Price is not the problem (Value for Money 76.9% positive) | Do not cut prices to fix ratings | Positive share of Value mentions |
| Logistic Regression finds 93% of negative reviews | Run the pipeline weekly to flag negative reviews for a person to read | Negative recall on a hand-checked sample |
"""

LIMITS = """## Limitations

- **Non-random sample.** Reviews come from three views (recent, negative-first, most-helpful) of 92 public product pages found by searching Flipkart categories. Percentages describe this sample, not all buyers.
- **Product variants.** Some listings are colour or pack variants of the same model. Copy-paste across listings was removed (2,168 reviews), but a few products are close relatives.
- **Approximate dates.** Flipkart shows relative dates, so trend charts are approximate.
- **Small groups.** Smartwatches have 553 reviews and some category-aspect cells are small (for example Smartwatch Customer Service, 14 mentions); read those with their confidence intervals.
- **Rule-based tagging.** The aspect keyword lists and VADER miss sarcasm, mixed sentences and implied aspects. The 80-mention check puts Negative-label precision at about 66% (53 of 80).
- **Strict cleaning removes real reviews.** Short, junk, duplicate and copy-paste reviews were dropped (23,046 raw to 10,004), which favours longer reviews (median 9 words).
- **Star-based labels.** The classifier learns to reproduce star ratings, so its high score does not mean it understands defects.
- **Only Flipkart.** All reviews come from one marketplace, so results may differ on other sites.
- **No transformer model.** A model such as BERT could understand context better, but it needs more data and computing power, so it was not used.

## Conclusion
The pipeline cleans and analyses about ten thousand real Flipkart reviews across 92 products and shows that customers are mainly unhappy with **after-sales service, comfort and fit, and products that stop working early**, not with price. The results show a product team where to look first; they are not exact defect rates.
"""


def apply(cells):
    """Rebuild all notes around the code cells (cells: list of nbformat cells)."""
    import nbformat
    md = nbformat.v4.new_markdown_cell
    codes = [c for c in cells if c["cell_type"] == "code"]
    new = [md(INTRO), md(DATA)]
    used = set()
    for c in codes:
        first = "".join(c["source"]).lstrip().splitlines()[0]
        hit = [n for n in NOTES if first.startswith(n[0])]
        assert len(hit) == 1, ("no unique note for code cell", first[:60], [h[0] for h in hit])
        key, head, before, after = hit[0]
        assert key not in used, ("note used twice", key)
        used.add(key)
        if head:
            new.append(md(head))
        new.append(md(before))
        new.append(c)
        if after:
            new.append(md(after))
    assert len(used) == len(NOTES), ("notes without a code cell", set(n[0] for n in NOTES) - used)
    new += [md(INSIGHTS), md(LIMITS)]
    cells[:] = new
