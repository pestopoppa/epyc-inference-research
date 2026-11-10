# AGG Phase 0 single-fact golds: blind second read (2026-11-10)

Reader 2 read each filing (pypdfium2, face statements, USD millions) before opening the stored FinanceBench gold. Line-item rule: each metric is the exact line item on the face of the statement. Caveat: the golds of 00070, 00005, 01346 were glimpsed in a truncated listing before reading; those values were re-derived from the filing regardless.

| id | reader 2 answer (page: quote) | stored gold | result |
|---|---|---|---|
| 00070 AWK | CA 1,250, CL 2,811 (p81-82) -> WC -1,561, negative | -1561M | match, verified |
| 00005 Corning | Total current assets 7,453; Total current liabilities 5,175 (p60) -> WC +2,278 | +831M | MISMATCH, unverified |
| 01346 Corning | "Effective tax rate 23% 20%" (p24) | 20% -> 23% | match, verified |
| 00080 PayPal | Total current assets 57,517; Total current liabilities 45,101 (p61) -> WC +12,416 | +1.6Bn | MISMATCH, unverified |
| 00216 Verizon | cash 2,605 + AR net 24,506 over CL 50,171 (p56) -> 0.54 | 0.54 | match, verified |
| 00566 Verizon | Total debt 150,639 vs 150,868 (p77) -> decreased 229 | decreased 229M | match, verified |
| 10420 AES | NET INCOME (LOSS) (505) (p132); TOTAL ASSETS 38,363 / 32,963 (p130) -> -1.42% | -0.02 | MISMATCH, unverified (gold matches attributable NI (546): -1.53%) |
| 06272 KO | Dividends (7,616); NI attributable 9,542 (p66-67) -> 0.80 | 0.8 | match, verified |
| 03718 LMT | Total net sales 65,984 vs 65,398 (p63) -> CAGR 0.45% | 0.4% | match, verified |
| 04481 PEP | Operating Profit 11,512 + D&A 2,763 over Net Revenue 86,392 (p62,64) -> 16.52% | 16.5% | match, verified |

## Operator ruling 2026-11-10 (line-item rule)
The 3 mismatches are resolved to the face line items: Corning WC +2,278; PayPal WC +12,416; AES ROA -1.42% (net income (505)). All 10 verified; FinanceBench values kept as `fb_gold_alt` (alternative definition). 00070 and 01346 are `blind: partial` (golds glimpsed before reading); 00005 was glimpsed too but the value was overruled by the filing.
