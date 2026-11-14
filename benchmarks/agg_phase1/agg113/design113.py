"""AGG-1.13 design data that names documents. Lives in this directory on purpose: ``build_agg113.py design_use()`` scans every other
AGG design / runner / test file for document names and EXCLUDES any document it finds there from the candidate set (selection rule 4), so the
step B runner and its tests must not spell a corpus document name; they import these tables instead."""

PILOT_DOCS = ("WALMART_2022_10K", "ULTABEAUTY_2023_10K", "BLOCK_2022_10K")   # the first 3 of the salted order (corpus.json pilot_docs)

STRATA = {   # PREDECLARED *Metrics*: declared strata (a stratum is reported, not tested)
    "thousands_unit_filers": ("BLOCK_2022_10K", "ULTABEAUTY_2023_10K"),
    "unlabelled_subtotal_filer": ("FOOTLOCKER_2023_10K",),
    "combined_registrant_filer": ("PG_E_2022_10K",),
    "january_end_filers": ("BESTBUY_2022_10K", "WALMART_2022_10K", "FOOTLOCKER_2023_10K", "ULTABEAUTY_2023_10K"),
    "may_year_end_filers": ("NIKE_2022_10K", "ORACLE_2022_10K"),
}

# The column headers the four January-end filings PRINT on their primary statements, first column = the fiscal-2022 column. Source: the PDF
# text of the gold pages named in facts_sheet.csv (Best Buy p40/41/43, Walmart p53/55/57, Foot Locker p35/37/39, Ulta p54/55/57), read for the
# step A fiscal-year audit; the l3 row re-checks the same thing on the real parse (``store_fy_check``).
PRINTED_HEADERS: dict[str, dict[str, list[str]]] = {
    "BESTBUY_2022_10K": {"income": ["January 29, 2022", "January 30, 2021", "February 1, 2020"],
                         "cash_flow": ["January 29, 2022", "January 30, 2021", "February 1, 2020"],
                         "balance_sheet": ["January 29, 2022", "January 30, 2021"]},
    "WALMART_2022_10K": {"income": ["2022", "2021", "2020"], "cash_flow": ["2022", "2021", "2020"], "balance_sheet": ["2022", "2021"]},
    "FOOTLOCKER_2023_10K": {"income": ["2022", "2021", "2020"], "cash_flow": ["2022", "2021", "2020"],
                            "balance_sheet": ["January 28, 2023", "January 29, 2022"]},
    "ULTABEAUTY_2023_10K": {"income": ["January 28, 2023", "January 29, 2022", "January 30, 2021"],
                            "cash_flow": ["January 28, 2023", "January 29, 2022", "January 30, 2021"],
                            "balance_sheet": ["January 28, 2023", "January 29, 2022"]},
}
