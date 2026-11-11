"""AGG-1.1 Phase 1 question definitions (stdlib only; no model, no network, no PDF parsing).

20 continuity questions = the Phase 0 question set (ids agg01-agg20, same ops/fields/text, one exception noted
on agg18) re-evaluated over the 28-document corpus, plus the new constructed aggregation questions p1_01...
Ops stay inside the set the Phase 0 deterministic scorer supports (argmax, argmin, sum, count_gt, count_lt,
top3); ``where`` and ``exclude_docs`` only change which companies enter the computation.
"""

from __future__ import annotations

import sys
from pathlib import Path

P0 = Path(__file__).resolve().parents[1] / "agg_phase0"
sys.path.insert(0, str(P0))
import agg_phase0 as A0  # noqa: E402

DIV_EX = ["JPMORGAN_2022_10K", "AMERICANEXPRESS_2022_10K"]
DIV_REASON = ("The cash-flow line 'Dividends paid' of JPMorgan and American Express combines preferred and common "
              "dividends and common-only paid is not stated on a statement face (JPMorgan: operator decision "
              "2026-11-10 #3; American Express: same rule applied, p99/p100)")
DIV_TXT = " (excluding JPMorgan and American Express, whose cash-flow dividend lines combine preferred and common dividends)"
CLS = "Among companies with a classified balance sheet, "

CONTINUITY: list[dict] = []
for q in A0.AGG_QUESTIONS:
    q = dict(q)
    q["family"] = "continuity"
    q["phase0_id"] = q["id"]
    if q["id"] == "agg18":
        # Phase 0 text excluded JPMorgan only; American Express (cash-flow line combines preferred and common)
        # would be silently missing from the 28-document sum, so the exclusion list is extended and said in the text.
        q["q"] = ("What were total FY2022 cash dividends paid to common shareholders by the companies in the corpus "
                  "other than JPMorgan and American Express, in USD millions?")
        q["exclude_docs"] = list(DIV_EX)
        q["exclude_reason"] = DIV_REASON
        q["text_changed_from_phase0"] = True
    CONTINUITY.append(q)

NEW: list[dict] = [
 # --- net income / margin
 {"id": "p1_01", "expr": "net_income/revenue", "op": "argmax", "family": "margin",
  "q": "Which company had the highest FY2022 net margin (net income attributable to shareholders divided by total revenue)?"},
 {"id": "p1_02", "expr": "net_income/revenue", "op": "argmin", "family": "margin",
  "q": "Which company had the lowest (most negative) FY2022 net margin (net income attributable to shareholders divided by total revenue)?"},
 {"id": "p1_03", "expr": "net_income/revenue", "op": "top3", "family": "margin",
  "q": "List the three companies with the highest FY2022 net margin (net income attributable to shareholders divided by total revenue), highest first."},
 {"id": "p1_04", "expr": "net_income/revenue", "op": "count_lt", "t": 0.05, "family": "margin",
  "q": "How many companies had an FY2022 net margin (net income attributable / total revenue) below 5%, counting net losses?"},
 {"id": "p1_05", "expr": "net_income", "op": "count_gt", "t": 10000, "family": "net_income",
  "q": "How many companies reported FY2022 net income attributable to shareholders above $10 billion?"},
 {"id": "p1_06", "expr": "net_income", "op": "top3", "family": "net_income",
  "q": "List the three companies with the highest FY2022 net income attributable to shareholders, highest first."},
 {"id": "p1_07", "expr": "net_income", "op": "sum", "family": "net_income",
  "where": {"expr": "revenue", "cmp": ">", "t": 90000},
  "q": "What is the combined FY2022 net income attributable to shareholders of the companies whose FY2022 total revenue was above $90 billion, in USD millions?"},
 {"id": "p1_08", "expr": "net_income", "op": "count_lt", "t": 2000, "family": "net_income",
  "q": "How many companies reported FY2022 net income attributable to shareholders below $2 billion, counting net losses?"},
 # --- revenue
 {"id": "p1_09", "expr": "revenue", "op": "count_gt", "t": 90000, "family": "revenue",
  "q": "How many companies reported FY2022 total revenue above $90 billion?"},
 {"id": "p1_10", "expr": "revenue", "op": "sum", "family": "revenue",
  "where": {"expr": "net_income", "cmp": "<", "t": 0},
  "q": "What is the combined FY2022 total revenue, in USD millions, of the companies that reported a net loss attributable to shareholders for FY2022?"},
 {"id": "p1_11", "expr": "revenue", "op": "count_lt", "t": 15000, "family": "revenue",
  "q": "How many companies reported FY2022 total revenue below $15 billion?"},
 # --- cash flow
 {"id": "p1_12", "expr": "operating_cash_flow", "op": "top3", "family": "cash_flow",
  "q": "List the three companies with the highest FY2022 cash from operating activities, highest first."},
 {"id": "p1_13", "expr": "operating_cash_flow", "op": "argmin", "family": "cash_flow",
  "q": "Which company had the lowest FY2022 cash from operating activities?"},
 {"id": "p1_14", "expr": "operating_cash_flow", "op": "sum", "family": "cash_flow",
  "where": {"expr": "net_income", "cmp": "<", "t": 0},
  "q": "What is the combined FY2022 cash from operating activities, in USD millions, of the companies that reported a net loss attributable to shareholders for FY2022?"},
 {"id": "p1_15", "expr": "operating_cash_flow-capex", "op": "argmax", "family": "free_cash_flow",
  "q": "Which company had the highest FY2022 free cash flow (cash from operating activities minus capital expenditures)? Companies with no capital-expenditure line (JPMorgan) are excluded."},
 {"id": "p1_16", "expr": "operating_cash_flow-capex", "op": "top3", "family": "free_cash_flow",
  "q": "List the three companies with the highest FY2022 free cash flow (cash from operating activities minus capital expenditures), highest first. Companies with no capital-expenditure line (JPMorgan) are excluded."},
 {"id": "p1_17", "expr": "operating_cash_flow-capex", "op": "count_lt", "t": 0, "family": "free_cash_flow",
  "q": "How many companies had negative FY2022 free cash flow (cash from operating activities minus capital expenditures)? Companies with no capital-expenditure line (JPMorgan) are excluded."},
 {"id": "p1_18", "expr": "operating_cash_flow-capex", "op": "argmin", "family": "free_cash_flow",
  "q": "Which company had the lowest (most negative) FY2022 free cash flow (cash from operating activities minus capital expenditures)? Companies with no capital-expenditure line (JPMorgan) are excluded."},
 # --- capex
 {"id": "p1_19", "expr": "capex", "op": "count_gt", "t": 3000, "family": "capex",
  "q": "How many companies spent more than $3 billion on capital expenditures in FY2022? Companies with no capital-expenditure line (JPMorgan) are excluded."},
 {"id": "p1_20", "expr": "capex", "op": "top3", "family": "capex",
  "q": "List the three companies with the largest FY2022 capital expenditures, largest first."},
 {"id": "p1_21", "expr": "capex/revenue", "op": "argmin", "family": "capex",
  "q": "Which company had the lowest FY2022 capital expenditures as a share of total revenue? Companies with no capital-expenditure line (JPMorgan) are excluded."},
 {"id": "p1_22", "expr": "capex", "op": "sum", "family": "capex",
  "where": {"expr": "revenue", "cmp": ">", "t": 90000},
  "q": "What were the combined FY2022 capital expenditures, in USD millions, of the companies whose FY2022 total revenue was above $90 billion? Companies with no capital-expenditure line (JPMorgan) are excluded."},
 {"id": "p1_23", "expr": "capex/revenue", "op": "count_gt", "t": 0.05, "family": "capex",
  "q": "How many companies spent more than 5% of FY2022 total revenue on capital expenditures? Companies with no capital-expenditure line (JPMorgan) are excluded."},
 # --- balance sheet
 {"id": "p1_24", "expr": "total_assets", "op": "count_gt", "t": 100000, "family": "balance_sheet",
  "q": "How many companies had total assets above $100 billion at FY2022 year end?"},
 {"id": "p1_25", "expr": "total_assets", "op": "argmin", "family": "balance_sheet",
  "q": "Which company had the smallest total assets at FY2022 year end?"},
 {"id": "p1_26", "expr": "total_assets", "op": "sum", "family": "balance_sheet",
  "where": {"expr": "net_income", "cmp": "<", "t": 0},
  "q": "What are the combined total assets at FY2022 year end, in USD millions, of the companies that reported a net loss attributable to shareholders for FY2022?"},
 {"id": "p1_27", "expr": "total_current_assets", "op": "argmax", "family": "balance_sheet",
  "q": CLS + "which company had the largest total current assets at FY2022 year end?"},
 {"id": "p1_28", "expr": "total_current_assets/total_current_liabilities", "op": "count_gt", "t": 1.25, "family": "balance_sheet",
  "q": CLS + "how many had a current ratio (total current assets divided by total current liabilities) above 1.25 at FY2022 year end?"},
 {"id": "p1_29", "expr": "total_current_assets/total_current_liabilities", "op": "argmax", "family": "balance_sheet",
  "q": CLS + "which had the highest current ratio (total current assets divided by total current liabilities) at FY2022 year end?"},
 {"id": "p1_30", "expr": "total_current_assets/total_current_liabilities", "op": "argmin", "family": "balance_sheet",
  "q": CLS + "which had the lowest current ratio (total current assets divided by total current liabilities) at FY2022 year end?"},
 {"id": "p1_31", "expr": "total_current_assets-total_current_liabilities", "op": "sum", "family": "balance_sheet",
  "q": CLS + "what is the combined FY2022 year-end working capital (total current assets minus total current liabilities), in USD millions?"},
 {"id": "p1_32", "expr": "total_current_assets-total_current_liabilities", "op": "top3", "family": "balance_sheet",
  "q": CLS + "which three had the highest FY2022 year-end working capital (total current assets minus total current liabilities), highest first?"},
 # --- dividends (JPMorgan and American Express excluded; companies with no dividend line paid none)
 {"id": "p1_33", "expr": "dividends_paid", "op": "argmax", "family": "dividends",
  "exclude_docs": DIV_EX, "exclude_reason": DIV_REASON,
  "q": "Which company paid the largest FY2022 cash dividends to common shareholders" + DIV_TXT + "?"},
 {"id": "p1_34", "expr": "dividends_paid", "op": "count_gt", "t": 5000, "family": "dividends",
  "exclude_docs": DIV_EX, "exclude_reason": DIV_REASON,
  "q": "How many companies paid more than $5 billion in FY2022 cash dividends to common shareholders" + DIV_TXT + "?"},
 {"id": "p1_35", "expr": "dividends_paid", "op": "top3", "family": "dividends",
  "exclude_docs": DIV_EX, "exclude_reason": DIV_REASON,
  "q": "List the three companies that paid the largest FY2022 cash dividends to common shareholders, largest first" + DIV_TXT + "."},
 {"id": "p1_36", "expr": "dividends_paid", "op": "sum", "family": "dividends",
  "where": {"expr": "revenue", "cmp": ">", "t": 50000},
  "exclude_docs": DIV_EX, "exclude_reason": DIV_REASON,
  "q": "What were the combined FY2022 cash dividends paid to common shareholders, in USD millions, by the companies whose FY2022 total revenue was above $50 billion" + DIV_TXT + "? Companies that paid no cash dividends contribute zero."},
 {"id": "p1_37", "expr": "dividends_paid/net_income", "op": "argmax", "family": "dividends",
  "where": {"expr": "net_income", "cmp": ">", "t": 0},
  "exclude_docs": DIV_EX, "exclude_reason": DIV_REASON,
  "q": "Among profitable companies (positive FY2022 net income attributable to shareholders) that paid cash dividends, which had the highest dividend payout ratio (cash dividends paid divided by net income)" + DIV_TXT + "?"},
 {"id": "p1_38", "expr": "dividends_paid/net_income", "op": "count_gt", "t": 0.6, "family": "dividends",
  "where": {"expr": "net_income", "cmp": ">", "t": 0},
  "exclude_docs": DIV_EX, "exclude_reason": DIV_REASON,
  "q": "Among profitable companies (positive FY2022 net income attributable to shareholders) that paid cash dividends, how many had a dividend payout ratio (cash dividends paid divided by net income) above 60%" + DIV_TXT + "?"},
]

ALL: list[dict] = CONTINUITY + NEW
