# AGG Phase 0 — gold review sheet (operator hand-verification)

Every gold below is `verified: false`. **No Phase-0 run may start until each question used is verified against the filing by a named reviewer; an unverified question is excluded, never assumed** (plan §(c), AGG-0.1).

## How to verify

1. Fill `facts_sheet.csv`: one row per (doc, metric). Set `value_usd_m`, `page` (PDF page), a verbatim `quote`, `reviewer`, and `status` = `filled` (or `na` with a `note` when the filing has no such line item, e.g. an unclassified balance sheet).
2. Run `python3 agg_phase0.py golds` — it recomputes every gold and the FinanceBench anchor checks below. Any `agrees: false` anchor means a sheet error; fix before signing.
3. For each question, tick the `verified` cell here AND set `verified: true`, `verified_by`, `verified_at` in `verification_record.json` (the runner reads that file, not this one).

Corpus: 10 FY2022 10-Ks; missing PDFs: AES_2022_10K, COCACOLA_2022_10K, GENERALMILLS_2022_10K, LOCKHEEDMARTIN_2022_10K, PEPSICO_2022_10K.

## FinanceBench anchor checks

| doc | expr | expect | sheet value | agrees | source |
|---|---|---|---|---|---|
| CORNING_2022_10K | `total_current_assets-total_current_liabilities` | sign 1 | None | None | financebench_id_00005 (+$831M, operating items only) |
| AMERICANWATERWORKS_2022_10K | `total_current_assets-total_current_liabilities` | approx -1561 | None | None | financebench_id_00070 (-$1561M) |
| PAYPAL_2022_10K | `total_current_assets-total_current_liabilities` | sign 1 | None | None | financebench_id_00080 (+$1.6Bn) |
| COCACOLA_2022_10K | `dividends_paid/net_income` | approx 0.8 | None | None | financebench_id_06272 (0.80) |
| AES_2022_10K | `net_income` | sign -1 | None | None | financebench_id_10420 (ROA -0.02) |

## 20 cross-document aggregation questions

| id | question | rule | derived gold | NA-excluded | missing cells | verified | reviewer |
|---|---|---|---|---|---|---|---|
| agg01 | Which company in the corpus reported the highest FY2022 total revenue? | `argmax(revenue)` | null | 0 | 10 | [ ] | |
| agg02 | Which company in the corpus reported the lowest FY2022 total revenue? | `argmin(revenue)` | null | 0 | 10 | [ ] | |
| agg03 | What is the combined FY2022 total revenue of all companies in the corpus, in USD millions? | `sum(revenue)` | null | 0 | 10 | [ ] | |
| agg04 | How many companies in the corpus reported FY2022 total revenue above $50 billion? | `count_gt(revenue, 50000)` | null | 0 | 10 | [ ] | |
| agg05 | List the three companies with the highest FY2022 total revenue, highest first. | `top3(revenue)` | null | 0 | 10 | [ ] | |
| agg06 | Which company reported the highest FY2022 net income attributable to shareholders? | `argmax(net_income)` | null | 0 | 10 | [ ] | |
| agg07 | How many companies in the corpus reported a net loss attributable to shareholders for FY2022? | `count_lt(net_income, 0)` | null | 0 | 10 | [ ] | |
| agg08 | What is the combined FY2022 net income attributable to shareholders across the corpus, in USD millions? | `sum(net_income)` | null | 0 | 10 | [ ] | |
| agg09 | Which company had the largest FY2022 capital expenditures? | `argmax(capex)` | null | 0 | 10 | [ ] | |
| agg10 | What were total FY2022 capital expenditures across all companies in the corpus, in USD millions? | `sum(capex)` | null | 0 | 10 | [ ] | |
| agg11 | How many companies spent more than $5 billion on capital expenditures in FY2022? | `count_gt(capex, 5000)` | null | 0 | 10 | [ ] | |
| agg12 | Which company had the largest total assets at FY2022 year end? | `argmax(total_assets)` | null | 0 | 10 | [ ] | |
| agg13 | List the three companies with the largest total assets at FY2022 year end, largest first. | `top3(total_assets)` | null | 0 | 10 | [ ] | |
| agg14 | Which company generated the most cash from operating activities in FY2022? | `argmax(operating_cash_flow)` | null | 0 | 10 | [ ] | |
| agg15 | For how many companies did FY2022 operating cash flow exceed net income attributable to shareholders? | `count_gt(operating_cash_flow-net_income, 0)` | null | 0 | 10 | [ ] | |
| agg16 | Among companies with a classified balance sheet, how many had positive working capital (total current assets minus total current liabilities) at FY2022 year end? | `count_gt(total_current_assets-total_current_liabilities, 0)` | null | 0 | 10 | [ ] | |
| agg17 | Among companies with a classified balance sheet, which had the most negative working capital at FY2022 year end? | `argmin(total_current_assets-total_current_liabilities)` | null | 0 | 10 | [ ] | |
| agg18 | What were total FY2022 cash dividends paid to common shareholders across the corpus, in USD millions? | `sum(dividends_paid)` | null | 0 | 10 | [ ] | |
| agg19 | Which company had the highest FY2022 capital expenditures as a share of total revenue? | `argmax(capex/revenue)` | null | 0 | 10 | [ ] | |
| agg20 | How many companies had an FY2022 net margin (net income attributable / total revenue) above 10%? | `count_gt(net_income/revenue, 0.1)` | null | 0 | 10 | [ ] | |

## 10 single-fact questions (FinanceBench golds; P-large, P-small, B-today)

| id | doc | question | FinanceBench gold | evidence page(s) | verified | reviewer |
|---|---|---|---|---|---|---|
| financebench_id_00070 | AMERICANWATERWORKS_2022_10K | Does American Water Works have positive working capital based on FY2022 data? If working capital is not a useful or relevant metric for this company, then pleas | No, American Water Works had negative working capital of -$1561M in FY 2022. | [80, 81] | [ ] | |
| financebench_id_00005 | CORNING_2022_10K | Does Corning have positive working capital based on FY2022 data? If working capital is not a useful or relevant metric for this company, then please state that  | Yes. Corning had a positive working capital amount of $831 million by FY 2022 close. This answer considers only operating current assets and current liabilities | [59] | [ ] | |
| financebench_id_01346 | CORNING_2022_10K | How much has the effective tax rate of Corning changed between FY2021 and FY2022? | The effective tax rate of Corning has changed from 20% in FY2021 to 23% in FY 2022. | [23] | [ ] | |
| financebench_id_00080 | PAYPAL_2022_10K | Does Paypal have positive working capital based on FY2022 data? If working capital is not a useful or relevant metric for this company, then please state that a | Yes. Paypal has a positive working capital of $ 1.6Bn as of FY2022 end. | [60] | [ ] | |
| financebench_id_00216 | VERIZON_2022_10K | Does Verizon have a reasonably healthy liquidity profile based on its quick ratio for FY 2022? If the quick ratio is not relevant to measure liquidity, please s | No. The quick ratio was approximately 0.54 for Verizon. It indicated that Verizon does not have a healthy liquidity profile. | [55] | [ ] | |
| financebench_id_00566 | VERIZON_2022_10K | Has Verizon increased its debt on balance sheet between 2022 and the 2021 fiscal period? | No. Verizon's debt decreased by $229 million. | [76] | [ ] | |
| financebench_id_10420 | AES_2022_10K | Based on the information provided primarily in the statement of financial position and the statement of income, what is AES's FY2022 return on assets (ROA)? ROA | -0.02 | [129, 131] | [ ] | |
| financebench_id_06272 | COCACOLA_2022_10K | What is Coca Cola's FY2022 dividend payout ratio (using total cash dividends paid and net income attributable to shareholders)? Round answer to two decimal plac | 0.8 | [62, 65] | [ ] | |
| financebench_id_03718 | LOCKHEEDMARTIN_2022_10K | What is Lockheed Martin's 2 year total revenue CAGR from FY2020 to FY2022 (in units of percents and round to one decimal place)? Provide a response to the quest | 0.4% | [62] | [ ] | |
| financebench_id_04481 | PEPSICO_2022_10K | What is the FY2022 unadjusted EBITDA % margin for PepsiCo? Calculate unadjusted EBITDA using unadjusted operating income and D&A (from cash flow statement). Giv | 16.5% | [61, 63] | [ ] | |
