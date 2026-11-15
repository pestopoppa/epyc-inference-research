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
#
# TWO shapes are declared, because the pipeline reads two shapes:
#   PRINTED_HEADERS  the TEXT-ROW shape (``facts_det.facts_from_page_text``): the header lines AS THE REAL PARSE FLATTENS THEM (verbatim from the page
#                    text of the l3 seed's ODL parse; only the header lines, no data row). A split header (``Fiscal year ended January 28, January 29,
#                    January 30,`` / blank / ``(In thousands) 2023 2022 2021``, or the month-days leading the year line) is kept split. This is what the
#                    real Ulta and Foot Locker statements look like; the AMENDMENT 1 contract fixtures had INTACT dates ("January 28, 2023") and
#                    went through the table reader only, so the contract passed on a shape the real parse never produced (AMENDMENT 2).
#   TABLE_HEADERS    the TABLE shape (``facts_det.facts_from_table``): the intact printed first-column dates, as amendment 1 had them. The real seed holds
#                    no table with a split header (96 tables, none), so this path is kept for parses that DO detect the table.
PRINTED_HEADERS: dict[str, dict[str, str]] = {
    "BESTBUY_2022_10K": {"income": "Fiscal Years Ended January 29, 2022 January 30, 2021 February 1, 2020",
                         "cash_flow": "Fiscal Years Ended January 29, 2022 January 30, 2021 February 1, 2020",
                         "balance_sheet": "January 29, 2022 January 30, 2021"},
    "WALMART_2022_10K": {"income": "Fiscal Years Ended January 31, (Amounts in millions, except per share data) 2022 2021 2020 Revenues:",
                         "cash_flow": "Fiscal Years Ended January 31, (Amounts in millions) 2022 2021 2020 Cash flows from operating activities:",
                         "balance_sheet": "As of January 31, (Amounts in millions) 2022 2021 ASSETS Current assets:"},
    "FOOTLOCKER_2023_10K": {"income": "($ in millions, except per share amounts) 2022 2021 2020",
                            "cash_flow": "($ in millions) 2022 2021 2020 From operating activities:",
                            "balance_sheet": "January 28, January 29, ($ in millions, except share amounts) 2023 2022 ASSETS"},
    "ULTABEAUTY_2023_10K": {"income": "Fiscal year ended January 28, January 29, January 30,\n\n(In thousands, except per share data) 2023 2022 2021",
                            "cash_flow": "Fiscal year ended January 28, January 29, January 30,\n\n(In thousands) 2023 2022 2021",
                            "balance_sheet": "January 28, January 29, (In thousands, except per share data) 2023 2022"},
}
TABLE_HEADERS: dict[str, dict[str, list[str]]] = {
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

# AMENDMENT 2 real-parse contract (PREDECLARED_AGG113_AMENDMENT2_20261115.md): run on the page text of the seed's own parse cache. For every
# January-end document and L3 version: no gold cell may have an L3 value ONLY under another period (the amendment 1 defect), and every cell of
# the filers listed here must carry its value under fiscal 2022 ("all": every gold cell of the document). Foot Locker's balance-sheet totals print
# without a caption (unlabelled subtotals, the declared trap): their cells are reported as ``no_l3_value`` and are NOT part of the floor.
REAL_PARSE_ALL_2022 = ("ULTABEAUTY_2023_10K",)
