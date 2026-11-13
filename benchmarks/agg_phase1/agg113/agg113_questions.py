"""AGG-1.13 step A question generator (stdlib only; no model, no network, no PDF parsing).

The questions are GENERATED FROM THE CELL SHEET (``facts_sheet.csv`` of this directory) by a fixed rule, so no number in a
question comes from a person who has seen any system's answer (no system has run on these documents at all):

* the templates below are the AGG-1.1 ``p1_*`` families (same wording, same ops, ops stay inside the frozen scorer's set:
  argmax, argmin, sum, count_gt, count_lt, top3), re-pointed at the 10 new filings;
* a threshold (``t`` of a count question, ``where.t`` of a filtered question) is picked by ``pick_threshold``: the "round"
  number (m x 10^k, m in 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7.5) that splits the 10 companies' values with 30% to 70% of them on the
  strict side and has the LARGEST minimum relative gap to any company's value (ties: the smaller multiplier, then the smaller t);
* a question is KEPT only if its margin audit passes (``MARGIN_MIN``: the smallest relative gap that decides the answer,
  including the gaps of its ``where`` filter, is >= 1.5%, no near-tie) and every cell it reads is a verified cell;
* the final list is the first ``QUOTA[family]`` templates of each family, in generation order, that survive the audit (a
  template that fails the audit is skipped and recorded in ``dropped``, never edited to pass).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

P0 = Path(__file__).resolve().parents[2] / "agg_phase0"
sys.path.insert(0, str(P0))
import agg_phase0 as A0  # noqa: E402

MARGIN_MIN = 0.015
DIV_EX = ["NIKE_2022_10K"]
DIV_REASON = ("The cash-flow line 'Dividends - common and preferred' of Nike combines common and preferred dividends and "
              "common-only paid is not stated on a statement face (the AGG-1.1 rule for JPMorgan and American Express, "
              "operator decision 2026-11-10 #3)")
DIV_TXT = " (excluding Nike, whose cash-flow dividend line combines common and preferred dividends)"
CLS = "Among the companies in the corpus, "
FCF_NOTE = ""  # every new filing has a capital-expenditure line (no JPMorgan-type exclusion)

GRID_M = (1, 1.5, 2, 2.5, 3, 4, 5, 6, 7.5)


def grid() -> list[float]:
    out = set()
    for k in range(-3, 8):
        for m in GRID_M:
            out.add(round(m * 10.0 ** k, 6))
    return sorted(out)


def rel(a: float, b: float) -> float:
    den = max(abs(a), abs(b))
    return abs(a - b) / den if den else 0.0


def pick_threshold(values: list[float], lo: float = 0.3, hi: float = 0.7) -> float | None:
    """Round number with 30-70% of ``values`` strictly above it and the largest minimum relative gap to any value."""
    n = len(values)
    best = None
    for t in grid():
        frac = sum(1 for v in values if v > t) / n
        if not lo <= frac <= hi:
            continue
        gap = min(rel(v, t) for v in values)
        key = (-round(gap, 6), t)
        if best is None or key < best[0]:
            best = (key, t, gap)
    if best is None:
        return None
    t = best[1]
    return int(t) if float(t).is_integer() else t


def _fmt(t: float) -> str:
    return f"{t:,.0f}" if float(t).is_integer() else f"{t:g}"


def _usd(t: float) -> str:
    """Threshold in USD millions rendered the way the AGG-1.1 questions do ($ billion above 1,000)."""
    if t >= 1000:
        b = t / 1000
        return f"${b:g} billion"
    return f"${t:,.0f} million"


def _pct(t: float) -> str:
    return f"{t * 100:g}%"


class Gen:
    def __init__(self, sheet: dict[tuple[str, str], dict[str, str]], docs: list[str], company: dict[str, str]):
        self.sheet, self.docs, self.company = sheet, docs, company

    def vals(self, expr: str, exclude: list[str] | None = None, where: dict[str, Any] | None = None) -> dict[str, float]:
        out = {}
        for d in self.docs:
            if d in (exclude or []):
                continue
            if where:
                w = A0.eval_expr(where["expr"], self.sheet, d)
                if w in ("missing", None) or not (w > where["t"] if where["cmp"] == ">" else w < where["t"]):
                    continue
            v = A0.eval_expr(expr, self.sheet, d)
            if v not in ("missing", None):
                out[d] = v
        return out

    def thr(self, expr: str, exclude: list[str] | None = None, where: dict[str, Any] | None = None) -> float:
        t = pick_threshold(list(self.vals(expr, exclude, where).values()))
        assert t is not None, expr
        return t

    def where_thr(self, expr: str, exclude: list[str] | None = None) -> float:
        return self.thr(expr, exclude)


def templates(g: Gen) -> list[dict[str, Any]]:
    """Every candidate question, in the fixed generation order (id = ``n13_NN``, NN the template index)."""
    T: list[dict[str, Any]] = []

    def add(**kw: Any) -> None:
        kw["id"] = f"n13_{len(T) + 1:02d}"
        T.append(kw)

    NM = "net_income/revenue"
    # ---- margin
    add(expr=NM, op="argmax", family="margin",
        q="Which company had the highest FY2022 net margin (net income attributable to shareholders divided by total revenue)?")
    add(expr=NM, op="argmin", family="margin",
        q="Which company had the lowest (most negative) FY2022 net margin (net income attributable to shareholders divided by total revenue)?")
    add(expr=NM, op="top3", family="margin",
        q="List the three companies with the highest FY2022 net margin (net income attributable to shareholders divided by total revenue), highest first.")
    t = g.thr(NM)
    add(expr=NM, op="count_gt", t=t, family="margin",
        q=f"How many companies had an FY2022 net margin (net income attributable / total revenue) above {_pct(t)}?")
    # ---- net income
    t = g.thr("net_income")
    add(expr="net_income", op="count_gt", t=t, family="net_income",
        q=f"How many companies reported FY2022 net income attributable to shareholders above {_usd(t)}?")
    add(expr="net_income", op="top3", family="net_income",
        q="List the three companies with the highest FY2022 net income attributable to shareholders, highest first.")
    tw = g.where_thr("revenue")
    add(expr="net_income", op="sum", family="net_income", where={"expr": "revenue", "cmp": ">", "t": tw},
        q=f"What is the combined FY2022 net income attributable to shareholders of the companies whose FY2022 total revenue was above {_usd(tw)}, in USD millions?")
    add(expr="net_income", op="argmin", family="net_income",
        q="Which company had the lowest FY2022 net income attributable to shareholders (a net loss counts as lowest)?")
    t = g.thr("net_income")
    add(expr="net_income", op="count_lt", t=t, family="net_income",
        q=f"How many companies reported FY2022 net income attributable to shareholders below {_usd(t)}, counting net losses?")
    # ---- revenue
    t = g.thr("revenue")
    add(expr="revenue", op="count_gt", t=t, family="revenue",
        q=f"How many companies reported FY2022 total revenue above {_usd(t)}?")
    add(expr="revenue", op="top3", family="revenue",
        q="List the three companies with the highest FY2022 total revenue, highest first.")
    add(expr="revenue", op="argmin", family="revenue",
        q="Which company reported the lowest FY2022 total revenue?")
    add(expr="revenue", op="count_lt", t=t, family="revenue",
        q=f"How many companies reported FY2022 total revenue below {_usd(t)}?")
    tw = g.where_thr("net_income")
    add(expr="revenue", op="sum", family="revenue", where={"expr": "net_income", "cmp": ">", "t": tw},
        q=f"What is the combined FY2022 total revenue, in USD millions, of the companies whose FY2022 net income attributable to shareholders was above {_usd(tw)}?")
    # ---- cash flow
    add(expr="operating_cash_flow", op="top3", family="cash_flow",
        q="List the three companies with the highest FY2022 cash from operating activities, highest first.")
    add(expr="operating_cash_flow", op="argmin", family="cash_flow",
        q="Which company had the lowest FY2022 cash from operating activities?")
    tw = g.where_thr("revenue")
    add(expr="operating_cash_flow", op="sum", family="cash_flow", where={"expr": "revenue", "cmp": ">", "t": tw},
        q=f"What is the combined FY2022 cash from operating activities, in USD millions, of the companies whose FY2022 total revenue was above {_usd(tw)}?")
    t = g.thr("operating_cash_flow")
    add(expr="operating_cash_flow", op="count_gt", t=t, family="cash_flow",
        q=f"How many companies generated more than {_usd(t)} of cash from operating activities in FY2022?")
    # ---- free cash flow
    FCF = "operating_cash_flow-capex"
    add(expr=FCF, op="argmax", family="free_cash_flow",
        q="Which company had the highest FY2022 free cash flow (cash from operating activities minus capital expenditures)?")
    add(expr=FCF, op="top3", family="free_cash_flow",
        q="List the three companies with the highest FY2022 free cash flow (cash from operating activities minus capital expenditures), highest first.")
    add(expr=FCF, op="count_lt", t=0, family="free_cash_flow",
        q="How many companies had negative FY2022 free cash flow (cash from operating activities minus capital expenditures)?")
    add(expr=FCF, op="argmin", family="free_cash_flow",
        q="Which company had the lowest (most negative) FY2022 free cash flow (cash from operating activities minus capital expenditures)?")
    # ---- capex
    t = g.thr("capex")
    add(expr="capex", op="count_gt", t=t, family="capex",
        q=f"How many companies spent more than {_usd(t)} on capital expenditures in FY2022?")
    add(expr="capex", op="top3", family="capex",
        q="List the three companies with the largest FY2022 capital expenditures, largest first.")
    add(expr="capex/revenue", op="argmin", family="capex",
        q="Which company had the lowest FY2022 capital expenditures as a share of total revenue?")
    t = g.thr("capex/revenue")
    add(expr="capex/revenue", op="count_gt", t=t, family="capex",
        q=f"How many companies spent more than {_pct(t)} of FY2022 total revenue on capital expenditures?")
    tw = g.where_thr("revenue")
    add(expr="capex", op="sum", family="capex", where={"expr": "revenue", "cmp": ">", "t": tw},
        q=f"What were the combined FY2022 capital expenditures, in USD millions, of the companies whose FY2022 total revenue was above {_usd(tw)}?")
    # ---- balance sheet
    t = g.thr("total_assets")
    add(expr="total_assets", op="count_gt", t=t, family="balance_sheet",
        q=f"How many companies had total assets above {_usd(t)} at FY2022 year end?")
    add(expr="total_assets", op="argmin", family="balance_sheet",
        q="Which company had the smallest total assets at FY2022 year end?")
    add(expr="total_current_assets", op="argmax", family="balance_sheet",
        q=CLS + "which company had the largest total current assets at FY2022 year end?")
    CR = "total_current_assets/total_current_liabilities"
    t = g.thr(CR)
    add(expr=CR, op="count_gt", t=t, family="balance_sheet",
        q=CLS + f"how many had a current ratio (total current assets divided by total current liabilities) above {t:g} at FY2022 year end?")
    add(expr=CR, op="argmax", family="balance_sheet",
        q=CLS + "which had the highest current ratio (total current assets divided by total current liabilities) at FY2022 year end?")
    add(expr=CR, op="argmin", family="balance_sheet",
        q=CLS + "which had the lowest current ratio (total current assets divided by total current liabilities) at FY2022 year end?")
    WC = "total_current_assets-total_current_liabilities"
    add(expr=WC, op="sum", family="balance_sheet",
        q=CLS + "what is the combined FY2022 year-end working capital (total current assets minus total current liabilities), in USD millions?")
    add(expr=WC, op="top3", family="balance_sheet",
        q=CLS + "which three had the highest FY2022 year-end working capital (total current assets minus total current liabilities), highest first?")
    add(expr=WC, op="count_lt", t=0, family="balance_sheet",
        q=CLS + "how many had negative working capital (total current assets minus total current liabilities) at FY2022 year end?")
    # ---- dividends (Nike excluded: combined common+preferred line; companies with no dividend line paid none)
    ex = {"exclude_docs": DIV_EX, "exclude_reason": DIV_REASON}
    add(expr="dividends_paid", op="argmax", family="dividends", **ex,
        q="Which company paid the largest FY2022 cash dividends to common shareholders" + DIV_TXT + "?")
    t = g.thr("dividends_paid", DIV_EX)
    add(expr="dividends_paid", op="count_gt", t=t, family="dividends", **ex,
        q=f"How many companies paid more than {_usd(t)} in FY2022 cash dividends to common shareholders" + DIV_TXT + "?")
    add(expr="dividends_paid", op="top3", family="dividends", **ex,
        q="List the three companies that paid the largest FY2022 cash dividends to common shareholders, largest first" + DIV_TXT + ".")
    tw = g.where_thr("revenue")
    add(expr="dividends_paid", op="sum", family="dividends", where={"expr": "revenue", "cmp": ">", "t": tw}, **ex,
        q=f"What were the combined FY2022 cash dividends paid to common shareholders, in USD millions, by the companies whose FY2022 total revenue was above {_usd(tw)}" + DIV_TXT + "? Companies that paid no cash dividends contribute zero.")
    add(expr="dividends_paid/net_income", op="argmax", family="dividends",
        where={"expr": "net_income", "cmp": ">", "t": 0}, **ex,
        q="Among profitable companies (positive FY2022 net income attributable to shareholders) that paid cash dividends, which had the highest dividend payout ratio (cash dividends paid divided by net income)" + DIV_TXT + "?")
    pv = list(g.vals("dividends_paid/net_income", DIV_EX, {"expr": "net_income", "cmp": ">", "t": 0}).values())
    t = pick_threshold(pv, 0.3, 0.7)
    add(expr="dividends_paid/net_income", op="count_gt", t=t, family="dividends",
        where={"expr": "net_income", "cmp": ">", "t": 0}, **ex,
        q=f"Among profitable companies (positive FY2022 net income attributable to shareholders) that paid cash dividends, how many had a dividend payout ratio (cash dividends paid divided by net income) above {_pct(t)}" + DIV_TXT + "?")
    return T


# Per-family quota of the final set (30 questions, set by the family structure of AGG-1.1 before the first margin audit was run; the cell values had been read by then, no system output exists): the first QUOTA[f]
# templates of family f, in generation order, that survive the margin audit; a template that fails the audit is skipped and
# recorded as dropped in the verification record (never edited to pass).
QUOTA = {"margin": 4, "net_income": 4, "revenue": 3, "cash_flow": 3, "free_cash_flow": 3, "capex": 3,
         "balance_sheet": 5, "dividends": 5}
N_QUESTIONS = sum(QUOTA.values())
assert N_QUESTIONS == 30
