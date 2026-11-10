# AGG Phase 0 gold: reader diffs and medium-confidence cells

Draft = `facts_sheet.csv` (agent 1, as of draft commit 005950d9d). Reader 2 = `facts_sheet.reader2.csv` (blind, committed c826f95ac before the draft was opened). Definition: the exact statement line item, USD millions, FY2022.

Result: 78 of 80 cells match exactly (value and status/unit) and carry `verified=true`, `verified_by: two-independent-readers 2026-11-10`. 2 cells mismatch (section 1). Medium-confidence cells (either reader) are in section 2. The three line-item re-checks (Corning, PayPal, PepsiCo capex) all agree with the face line, no correction needed.

## 1. Mismatches (verified=false)

| company | metric | draft | reader 2 | draft quote (page) | reader 2 quote (page) |
|---|---|---|---|---|---|
| JPMorgan | dividends_paid | TODO | 13562 | (none) BLANK: ambiguous: cash-flow 'Dividends paid' (13,562) combines preferred and common; common-only paid not stated (declar (p) | Dividends paid (13,562) (12,858) (12,690) (p165) |
| Paypal | dividends_paid | 0 | NA | We have never paid any cash dividends (p32) | We have never paid any cash dividends and we currently do not anticipate paying any cash dividends in the foreseeable future. (p32) |

Disposition: PayPal dividends is 0 vs NA (same sum; draft left unverified). JPMorgan dividends: draft left it blank as ambiguous (preferred+common combined), reader 2 read the face line 13,562; per decision (3) JPMorgan is dropped from agg18 and the cell is set NA with that reason.

## 2. Medium-confidence cells (either reader)

| company | metric | draft | reader 2 | match | draft conf / reader2 conf | draft quote (page) | reader 2 quote (page) | why medium |
|---|---|---|---|---|---|---|---|---|
| JPMorgan | capex | NA | NA | yes | medium / medium | (none) (p) | Net cash (used in) investing activities (137,819) (129,344) (261,912) (p165) | no capex/PP&E purchase line on cash flow statement (NA; only "All other investing activities, net") |
| JPMorgan | total_current_assets | NA | NA | yes | medium / medium | (none) (p) | Total assets(a) $ 3,665,743 $ 3,743,567 (p163) | JPM balance sheet is unclassified; quote is the balance sheet total-assets line, no current-asset/liability subtotals exist; unclassified balance sheet (NA) |
| JPMorgan | total_current_liabilities | NA | NA | yes | medium / medium | (none) (p) | Total assets(a) $ 3,665,743 $ 3,743,567 (p163) | JPM balance sheet is unclassified; quote is the balance sheet total-assets line, no current-asset/liability subtotals exist; unclassified balance sheet (NA) |
| JPMorgan | dividends_paid | TODO | 13562 | NO | none / medium | (none) (p) | Dividends paid (13,562) (12,858) (12,690) (p165) | face line "Dividends paid" covers common AND preferred; not common-only |
| Paypal | dividends_paid | 0 | NA | NO | medium / medium | We have never paid any cash dividends (p32) | We have never paid any cash dividends and we currently do not anticipate paying any cash dividends in the foreseeable future. (p32) | no dividend line: company states it has never paid cash dividends (NA) |
| Verizon | capex | 23087 | 23087 | yes | high / medium | Capital expenditures (including capitalized software) (23,087) (p57) | Capital expenditures (including capitalized software) (23,087) (20,286) (18,192) (p57) | line reads "Capital expenditures (including capitalized software)" |
| Coca-Cola | dividends_paid | 7616 | 7616 | yes | high / medium | Dividends (7,616) (p66) | Dividends (7,616) (7,252) (7,047) (p66) | face line is "Dividends" (no common qualifier) |
| PepsiCo | capex | 5207 | 5207 | yes | medium / high | Capital spending (5,207) (p64) | Capital spending (5,207) (4,625) (4,240) (p64) | gross capital spending line; MD&A cites net capital spending $5.0bn (after sales of PP&E); gross used; magnitude of outflow |
| PepsiCo | dividends_paid | 6172 | 6172 | yes | medium / high | Cash dividends paid (6,172) (p65) | Cash dividends paid (6,172) (5,815) (5,509) (p65) | line is total cash dividends paid (may include small preferred dividends); magnitude of outflow |

## 3. Page-only differences (value matches)

- American Water Works revenue: draft p58, reader 2 p83 (reader 2 page is the income statement face). Value 3792.
- American Water Works net_income: draft p58, reader 2 p83 (reader 2 page is the income statement face). Value 820.
