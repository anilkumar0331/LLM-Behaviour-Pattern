Project Title: LARGE LANGUAGE MODEL BEHAVIOR PATTERN IN DEFINED TEXT OUTPUT

Steps performed on GPTRC Dataset:

1) Data Cleaning - data_cleaning.ipynb (GPTRC_GPT_Responses_Corpus.xlsx -> filtered_gptrc_data.xlsx)
2) Web Searching - web-searching.ipynb (filtered_gptrc_data.xlsx -> gptrc_urls.json)
3) Web Scraping - web-scraping.ipynb (gptrc_urls.json -> gptrc_reliable_urls.json -> gptrc_reliable_urls_filtered.json -> gptrc_urls_context.json)
4) Hallucinations - hallucinations.ipynb (gptrc_urls_context.json -> gptrc_urls_context_with_llm_answers -> hallucinations.json -> hallucinations_filtered.json)
5) Hallucination Patterns - Hallucination_Patterns/hallucinations_patterns.ipynb (hallucinations_filtered.json -> factual_inaccuracy_and_misclassification_hallucinations.json, fabricated_detail_hallucinations.json, speculative_reasoning_hallucinations.json, all_four_patterns_hallucinations.json, miscellaneous_hallucinations.json, no_hallucinations.json)
