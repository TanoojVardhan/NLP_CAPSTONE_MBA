## Limitations

- **Non-random sample.** Reviews come from three views (recent, negative-first, most-helpful) of 92 public product pages found by searching Flipkart categories. Percentages describe this sample, not all buyers.
- **Product variants.** Some listings are colour or pack variants of the same model. Cross-listing copy-paste was removed (2,168 reviews), but a few products are close relatives.
- **Approximate dates.** Flipkart shows relative dates, so trend charts are approximate.
- **Small groups.** Smartwatches have 553 reviews and some category-aspect cells are small (for example Smartwatch Customer Service, 14 mentions); read those with their confidence intervals.
- **Rule-based tagging.** The aspect lexicon and VADER misread sarcasm, mixed sentences and mentions of other products. The 80-mention audit puts Negative-tag precision at about 66% (53 of 80).
- **Strict cleaning removes real reviews.** Short, garbage, duplicate and copy-paste reviews were dropped (23,046 raw to 10,004), which favours longer reviews (median 9 words).
- **Old training file.** Electronics_5.json is a roughly six-year-old Amazon file with different products and a very positive mix (88%); it is used only as outside training data.
- **Only Flipkart.** Amazon.in needed a login, which we did not bypass.

## Conclusion
The pipeline cleans and tags about ten thousand real reviews across 92 products and points to Customer Service, Comfort & Fit and Build & Durability as the main complaint areas. The results show where to look; they are not exact defect rates.
