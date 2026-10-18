# E-Commerce Sentiment Analysis & Defect Diagnostics
24INFS7101 NLP & Text Analytics in Business, A2 Capstone.

Aspect-based sentiment analysis (ABSA) of 119 hand-collected verified Amazon.in reviews (15 products, 5 categories).

## Folder layout
- data/ : real_amazon_reviews_sample.csv (the dataset)
- notebooks/ABSA_Capstone_Phase2.ipynb : executed notebook (main analysis)
- notebooks/build_notebook.py : script that generates and executes the notebook
- outputs/ : result tables (CSV), negative_mentions_manual_audit.csv (hand-read Negative labels), dashboard.html
- outputs/figures/ : charts

## How to run
1. Install Python 3.10+ and then: pip install -r requirements.txt
2. python -m spacy download en_core_web_sm
3. Open notebooks/ABSA_Capstone_Phase2.ipynb and Run All (or run build_notebook.py to rebuild it).
NLTK data (vader_lexicon, stopwords, punkt, wordnet) is downloaded by the first cells.

Random seeds are fixed (42). Results will match the report only with the same dataset.

## Known limits
Small sample (119 reviews, 9 one-star); rule-based aspect lexicon and VADER; results are directional. See the Limitations section of the report.

## Flipkart version (main results, 15 Oct 2026)
- Data: `data/flipkart_reviews.csv` (10,005 verified reviews, 91 products, 5 categories), cleaning counts in `data/flipkart_cleaning_log.json`. Raw dumps: `data/flipkart_raw.psv`, `data/flipkart_raw2.psv`. Clean with: `cat data/flipkart_raw.psv data/flipkart_raw2.psv | python3 data/build_flipkart_csv.py x data/flipkart_reviews.csv`.
- Test data only: `data/amazon_electronics5_test_sample.csv` (18,376 reviews sampled from Electronics_5.json by `data/sample_electronics5.py`).
- Notebook: `notebooks/ABSA_Capstone_Flipkart.ipynb`, rebuilt by `python notebooks/build_notebook_flipkart.py` (reads `notebooks/insights_flipkart.md` and `limitations_flipkart.md`). Results in `outputs_flipkart/`.
- The sample is not random (recent, negative-first and most-helpful views of each product page). The 80-mention audit file was a first pass by the assistant; please re-check it.
- The earlier 119-review Amazon.in pilot and the old scraping files were moved out of this folder (see _old_to_delete).
