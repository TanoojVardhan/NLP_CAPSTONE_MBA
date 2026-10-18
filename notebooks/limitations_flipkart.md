## 14. Limitations

- **Non-random sample.** Reviews come from three views (recent, negative-first, most-helpful) of 91 public product pages found by searching Flipkart categories. Percentages describe this sample, not all buyers.
- **Product variants.** Some listings are colour or pack variants of the same model. Cross-listing copy-paste was removed (2,165 reviews), but a few products are close relatives. Product names for newly added products are shortened from the web address.
- **Approximate dates.** Flipkart shows relative dates ("7 days ago", "Mon, 2024"), so trend charts are approximate.
- **Small groups.** Smartwatches have 553 reviews and some category-aspect cells are small (e.g. Smartwatch Customer Service, 14 mentions); read those with their confidence intervals.
- **Rule-based tagging.** The aspect lexicon and VADER misread sarcasm, mixed sentences and mentions of other products. The 80-mention audit puts Negative-tag precision at about 66% (53 of 80). That audit was a first pass by the assistant and should be re-checked by the team.
- **Strict cleaning removes real reviews.** Short, garbage, duplicate and copy-paste reviews were dropped (23,046 raw to 10,005), which favours longer reviews (median 9 words).
- **Old test file.** Electronics_5.json is a roughly six-year-old Amazon file with different products and a very positive mix; it is used only as an out-of-domain test.
- **Only Flipkart.** Amazon.in needed a login, which we did not bypass.

## 15. Conclusion
The pipeline cleans and tags ten thousand real reviews across 91 products and points to Customer Service, Comfort & Fit and Build & Durability as the main complaint areas. The results show where to look; they are not exact defect rates.
