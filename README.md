# E-Commerce Sentiment Analysis & Defect Diagnostics
24INFS7101 NLP & Text Analytics in Business, A2 Capstone.

Aspect-based sentiment analysis (ABSA) of Flipkart consumer-electronics reviews: 10,004 cleaned verified-purchase reviews, 92 products, 5 categories. It shows which product aspect (battery, build, comfort, customer service and so on) drives negative sentiment in each category.

## Folder layout
- `data/flipkart_raw.csv` : raw scraped reviews (23,046 rows)
- `data/flipkart_reviews.csv` : cleaned dataset used by the notebook
- `data/flipkart_cleaning_log.json` : counts for every cleaning step
- `data/build_flipkart_csv.py` : cleans the raw file: `python3 data/build_flipkart_csv.py data/flipkart_raw.csv data/flipkart_reviews.csv`
- `data/negative_mentions_manual_audit_flipkart.csv` : hand-read sample of 80 Negative labels
- `notebooks/ABSA_Capstone_Flipkart.ipynb` : the analysis; `build_notebook_flipkart.py` rebuilds it (reads `insights_flipkart.md`, `limitations_flipkart.md`, `clean_markdown.py`)
- `outputs_flipkart/` : result tables, figures and the dashboard

## How to run
1. Python 3.10+. Open the notebook and Run All; the first cell installs the libraries and language data with `!pip`.
2. Random seeds are fixed (42), so results match the report with the same data.

## Notes
- The sample is not random (recent, negative-first and most-helpful views of each product page), so compare aspects with each other rather than quoting overall percentages.
- The 80-mention audit file was a first pass; please re-check it as a team.
- Reports and presentations are in `documents/`, which is git-ignored.
