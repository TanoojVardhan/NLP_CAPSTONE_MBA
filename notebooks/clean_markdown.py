"""Plain 'what we did' notes for the notebook (no chapter numbers, no course-unit tags).
Used by build_notebook_flipkart.py and by the one-off cleanup of the .ipynb."""
T = {
0: """# E-Commerce Sentiment Analysis & Defect Diagnostics
**Aspect-based sentiment analysis of Flipkart consumer-electronics reviews**
24INFS7101 NLP & Text Analytics in Business (MBA Capstone, A2)

**Goal:** star ratings say a product is "3 out of 5" but not *what* is wrong. We read the review text, split it by product aspect (battery, sound, build, comfort, customer service and so on), score the sentiment of each aspect, and show which aspect is hurting which product category.

**Data:** about 10,000 cleaned, verified-purchase Flipkart reviews of 92 products in 5 categories (earbuds and neckbands, speakers and headphones, smartwatches, power banks and chargers, PC peripherals).

**What this notebook does, in order:** load and check the data, clean the text, tag aspects and score sentiment, check the scores by hand and against star ratings, find defect hot-spots by category and product, then add extra views (entities, clusters, trends, summaries, word similarity, topics), a classifier benchmark, and a dashboard.""",
1: """### Setup
Run the next cell once. It installs the libraries and language data the notebook needs.""",
3: """### Ethics and privacy
We used only publicly visible review pages (no login, no CAPTCHA bypass) and kept only review text, rating, date and helpful votes; reviewer names were not saved. Reviews are a self-selected sample, so results describe this sample, not all buyers. Details are in the Limitations at the end.""",
4: """### Data
We collected verified-purchase reviews from Flipkart product pages (three views per product: most recent, lowest rated first, most helpful), cleaned them (removed garbage, duplicates and copy-pasted reviews, kept reviews of 3+ words) and dropped products with fewer than 30 reviews. The cells below load the file, check it and show the rating mix.""",
9: """### Text cleaning
Lower-casing, removing noise, tokenising and removing stop-words, while keeping negation words such as "not" and "never" because they change the meaning.""",
11: """### Word importance
Counting words (Bag-of-Words) and weighting them by how distinctive they are (TF-IDF) to see which terms stand out in positive and negative reviews.""",
13: """### Aspect tagging and sentiment
Each review is split into clauses. A keyword list assigns each clause to one of 10 aspects, and VADER (a rule-based sentiment scorer) labels it Positive, Neutral or Negative. The result is one row per aspect mention.""",
16: """### Manual check of Negative labels
We read a random sample of 80 mentions that VADER called Negative and judged whether each is a real complaint about its tagged aspect. This gives an estimate of how reliable the Negative label is.""",
19: """**Reading the validation.** Compare the overall agreement with the always-"Positive" baseline: when most reviews are positive, that baseline already scores well. The negative recall and precision printed under the charts are more useful for finding complaints.""",
21: """### Defect hot-spots
Negative share of each aspect, by product category and by product. This is the main business output.""",
27: """**Reading the heatmap.** Percentages are automatic VADER labels. The manual check above shows roughly how many Negative labels are genuine, so treat them as directional, not exact defect rates.""",
29: """### Does the review view change the picture?
The data mixes three views of each product page, so we repeat the aspect table for each view and for the recent view alone.""",
31: """### Named entities
spaCy finds brands, products, organisations and numbers mentioned in reviews.""",
34: """**Reading this:** off-the-shelf entity detection is noisy on short, informal, code-mixed review text, so it is used only as a rough view of which brands and models people name.""",
35: """### Clustering
K-Means groups reviews by their TF-IDF text to see whether natural groups line up with product categories.""",
37: """**Reading this:** clusters that mix categories are driven by shared themes (for example sound or battery) rather than by product type.""",
38: """### Trend over time
Monthly average rating and review volume.""",
40: """**Caveat:** Flipkart shows relative dates, so dates are approximate and some months have few reviews.""",
41: """### Summaries
An extractive summary picks the most representative review sentences per category, as a quick manager-friendly read.""",
43: """### Word similarity
A Word2Vec model learns which words are used in similar contexts in these reviews, shown on a 2D map.""",
46: """**Note:** a transformer model such as BERT could understand context better, but it needs more data and compute than this project needs, so we did not use it.""",
47: """### Complaint topics
LDA topic modelling on 1-2 star reviews to find recurring complaint themes.""",
50: """**Reading this:** topics are built from 1-2 star reviews only; read the top words of each topic to name the theme.""",
51: """### Classifier benchmark
Logistic Regression and Naive Bayes on TF-IDF features predict Positive vs Negative reviews. We report accuracy, negative recall, macro-F1 and cross-validation, and compare with a majority-class baseline.""",
54: """**Reading the classifier result.** TF-IDF is fitted inside a pipeline so it only sees training data. Accuracy alone can mislead because most reviews are positive, so look at negative recall and macro-F1.""",
56: """### Dashboard
The same results as an interactive Plotly dashboard.""",
}
INSIGHTS = """## Key insights (Flipkart: 10,004 reviews, 92 products, 5 categories)

1. **Where the defects are.** Of 16,566 aspect mentions, 20.4% are negative. The highest negative shares are **Customer Service (37.7%)**, **Comfort & Fit (31.3%)** and **Build & Durability (29.2%)**, then Battery & Charging (23.2%). Display & Design (8.6%) and Value for Money (13.5%) are the safest.
2. **Category hot-spots (at least 10 mentions).** TWS earbuds: Customer Service 46.8% and Comfort & Fit 37.4% negative. Smartwatches: Build & Durability 40.4%. PC peripherals: Build & Durability 34.7% (652 mentions, the largest hot-spot) and Customer Service 37.5%.
3. **Sampling changes the picture.** The negative share is 8.9% in the "most helpful" view, 24.3% in the "most recent" view and 55.1% in the "negative first" view. The headline 20.4% depends on this mix, so compare aspects with each other rather than quoting it as a market-wide rate. Using only the recent view, the top aspects stay the same (Customer Service 42.8%, Comfort & Fit 38.5%, Build 31.1%).
4. **Auditing the tags.** In a random sample of 80 VADER-Negative mentions, 53 (66%, 95% interval 55%-76%) were genuine complaints about the tagged aspect. Negative shares are therefore upper-bound signals, not exact defect rates.
5. **Predicting sentiment.** VADER agrees with star-based sentiment on 77.6% of reviews (always-Positive baseline 61.0%). Logistic regression reaches 0.94 accuracy and 0.93 negative recall on the held-out 25% (3-fold CV 0.94 / 0.93), above Naive Bayes (0.92 / 0.82).
6. **Product spread.** Average ratings range from 2.0 to 4.8 across the 92 products; the lowest are a neckband (2.0), a Type-C earphone (2.3) and a 10000mAh power bank (2.4).
"""
LIMITS = """## Limitations

- **Non-random sample.** Reviews come from three views (recent, negative-first, most-helpful) of 92 public product pages found by searching Flipkart categories. Percentages describe this sample, not all buyers.
- **Product variants.** Some listings are colour or pack variants of the same model. Cross-listing copy-paste was removed (2,168 reviews), but a few products are close relatives.
- **Approximate dates.** Flipkart shows relative dates, so trend charts are approximate.
- **Small groups.** Smartwatches have 553 reviews and some category-aspect cells are small (for example Smartwatch Customer Service, 14 mentions); read those with their confidence intervals.
- **Rule-based tagging.** The aspect lexicon and VADER misread sarcasm, mixed sentences and mentions of other products. The 80-mention audit puts Negative-tag precision at about 66% (53 of 80).
- **Strict cleaning removes real reviews.** Short, garbage, duplicate and copy-paste reviews were dropped (23,046 raw to 10,004), which favours longer reviews (median 9 words).
- **Only Flipkart.** Amazon.in needed a login, which we did not bypass.

## Conclusion
The pipeline cleans and tags about ten thousand real reviews across 92 products and points to Customer Service, Comfort & Fit and Build & Durability as the main complaint areas. The results show where to look; they are not exact defect rates.
"""
def apply(cells):
    for i, t in T.items():
        assert cells[i]["cell_type"] == "markdown", i
        cells[i]["source"] = t
    cells[58]["source"] = INSIGHTS
    cells[59]["source"] = LIMITS

# One-line "what this cell does" notes, keyed by the code cell's index in the notebook
# (indices are those of the 63-cell version produced before notes are inserted).
CODE_NOTES = {
2: "Installs the libraries (pandas, scikit-learn, spaCy, gensim, plotly and others) and downloads the language data (spaCy model, NLTK stop-words and VADER lexicon). Run once.",
5: "Sets the folder paths, loads `flipkart_reviews.csv` into a table (`df`) and converts the rating and date columns to numbers and dates.",
6: "Checks the data is clean (no duplicate IDs or texts, every review has 3+ words), prints the cleaning log and shows review counts by view and by category.",
7: "Summarises the star ratings: average, spread and how many reviews each star level has.",
8: "Plots the star-rating distribution and saves it as a figure.",
10: "Cleans every review (removes links and noise, lower-cases, tokenises, removes stop-words while keeping negations, lemmatises) and creates the `clean_text` and `lemmas` columns used later.",
12: "Compares Bag-of-Words (raw word counts) with TF-IDF (distinctive-word weights) by plotting the top 20 terms of each.",
14: "Defines the keyword lists for the 10 aspects, splits each review into clauses, tags each clause with its aspect and scores it with VADER as Positive, Neutral or Negative. Builds `aspect_df`, one row per aspect mention.",
15: "Prints 40 random tagged sentences so we can judge by eye whether the aspect tags make sense.",
17: "Loads our hand-checked sample of 80 Negative mentions and computes how many are genuine complaints, with a 95% confidence interval.",
18: "Gives each whole review a VADER sentiment, converts star ratings to Positive / Neutral / Negative, and measures how often the two agree.",
20: "Plots VADER sentiment next to star-rating sentiment and prints negative-class recall and precision plus the always-Positive baseline.",
22: "Calculates the Positive / Neutral / Negative share of every aspect within every category and lists the top defect signals.",
23: "Saves the aspect and review tables as CSV files in `outputs_flipkart`, and calculates each aspect's overall negative share with its confidence interval.",
24: "Draws the defect heatmap: % negative for each aspect in each category (cells with fewer than 3 mentions are left blank).",
25: "Lists the 10 strongest category-by-aspect hot-spots (at least 10 mentions).",
26: "Plots how often each aspect is mentioned across all reviews.",
28: "Builds a product-level table (average rating, reviews, % negative) and plots average rating against review volume per product.",
30: "Repeats the aspect results for each review view (recent, negative-first, helpful) and for the recent view alone, to show how the sampling changes the numbers.",
32: "Runs spaCy named-entity recognition on every review to pull out brands, products, places and organisations, and counts them.",
33: "Plots the most frequently mentioned named entities.",
36: "Groups reviews into 5 clusters with K-Means on TF-IDF text and compares the clusters with the product categories.",
39: "Plots monthly review volume and average star rating over time.",
42: "Defines and runs an extractive summary: picks the most representative sentence for each category.",
44: "Trains a Word2Vec model on the reviews and prints the words most similar to battery, sound, quality, price and service.",
45: "Projects the Word2Vec word vectors to 2D with PCA and plots key aspect words.",
48: "Trains an LDA topic model with 4 topics on the 1-2 star reviews to find recurring complaint themes.",
49: "Saves the topic words to a file and plots the top words of each topic.",
52: "Trains Logistic Regression and Naive Bayes on TF-IDF features to predict Positive vs Negative. Uses a 75/25 split and 3-fold cross-validation and prints accuracy, negative recall, macro-F1 and the confusion matrix.",
53: "Fits Logistic Regression on all Flipkart reviews and lists the words most associated with positive and with negative reviews.",
55: "Draws word clouds of the words in positive (4-5 star) and negative (1-2 star) reviews.",
57: "Builds the interactive Plotly dashboard (rating distribution, aspect frequency, defect heatmap, lowest-rated products) and saves it as an HTML page.",
}

_apply_text = apply
def apply(cells):
    _apply_text(cells)
    import nbformat as _n
    for i in sorted(CODE_NOTES, reverse=True):
        assert cells[i]["cell_type"] == "code", i
        cells.insert(i, _n.v4.new_markdown_cell("**What this cell does:** " + CODE_NOTES[i]))
