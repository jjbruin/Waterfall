"""The 26Q1 TIAA reference document, transcribed cell by cell.

Separated from the comparison logic so the transcription can be reviewed on its
own terms — it is hand-entered from the published pages and is the one part of
the variance workbook that cannot be re-derived from anything.

Pages 2 and 4 of the source are RASTER images (no text layer), so those two were
read off a 3.2x render by eye. Pages 1 and 3 carry real text and were checked
against `pdfplumber` extraction.

Units, uniformly: dollars in $M as printed, percentages in percentage points,
DSCR as a bare multiple. None means the page prints "n/a" or a bare accounting
dash — the two are distinguished per page where it matters.

The PDF prints a "Pegasus Life Storage - Add'l" row under TGA 2022 that has no
counterpart in our data (we carry one Pegasus, in Individual Investments). It is
kept, marked PDF_ONLY, so the workbook shows it rather than quietly dropping it.
"""

NA = "n/a"        # the page prints the literal n/a
DASH = None       # the page prints a bare accounting dash

# ── page 1: allocation ───────────────────────────────────────────────────
PAGE1_ASSET = {
    # bucket: (funded $, committed $) — full dollars, as printed inside the bars
    "Multifamily": (240_410_995, 280_853_025),
    "Retail": (117_414_277, 117_868_982),
    "Self-Storage": (34_172_689, 34_172_689),
    "Office": (12_243_598, 12_243_598),
}
PAGE1_TOTALS = {"funded": 404_200_000, "committed": 445_100_000}
#: bucket -> (pct of funded, funded $) from the narrative
PAGE1_DEAL_TYPE = {
    "Value-Add": (38.0, 153_100_000),
    "Income": (32.0, 131_300_000),
    "New Construction": (30.0, 119_800_000),
}
#: the narrative's asset-mix percentages
PAGE1_ASSET_PCT = {"Multifamily": 59.0, "Retail": 29.0,
                   "Self-Storage": 9.0, "Office": 3.0}

# ── page 2: Financial ────────────────────────────────────────────────────
# vcode -> (debt, total_pref, ptr_equity, total_cap, pct_of_pref, invested,
#           unfunded, total_commitment, itd, net_roe)
FIN_COLS = ("debt", "total_pref", "ptr_equity", "total_cap", "pct_of_pref",
            "invested", "unfunded", "total_commitment", "itd", "net_roe")
PAGE2 = {
    "P0000019": ("Giant 7", 95.1, 21.0, 15.0, 131.1, 57, 11.5, 0.5, 11.9, 5.87, 10.5),
    "P0000017": ("East Manchester", 9.6, 3.6, 2.4, 15.6, 76, 2.7, DASH, 2.7, 0.81, 6.1),
    "P0000021": ("JB Fair Park", 48.98, 14.3, 3.9, 67.1, 85, 6.1, 6.1, 12.2, 1.17, 4.8),
    "P0000030": ("Nottingham Village", 38.9, 9.1, 6.3, 54.3, 41, 3.8, DASH, 3.8, 0.23, 1.4),
    "P0000018": ("Evergreen Plaza", 45.4, 16.4, 8.1, 69.8, 73, 12.0, DASH, 12.0, 5.10, 9.6),
    "PCITWES": ("City West", NA, 5.9, 14.2, 20.2, 84, 5.0, DASH, 5.0, 0.04, NA),
    "P0000065": ("Ascent on Steamboat", 32.5, 21.7, 7.6, 61.8, 69, 15.0, DASH, 15.0, 2.10, 3.2),
    "P0000066": ("Pegasus Life Storage", DASH, 8.1, 2.6, 10.7, 91, 7.4, DASH, 7.4, DASH, 0.0),
    "P0000067": ("Brainerd Place Apartments", 89.5, 31.7, 17.7, 138.9, 74, 11.6, 12.0, 23.6, -0.21, -0.9),
    "P0000068": ("The Point at Plymouth Meeting", 63.1, 26.9, 10.0, 100.0, 71, 19.0, DASH, 19.0, 2.77, 3.8),
    "P0000069": ("Mount Prospect Plaza", 21.2, 9.2, 4.5, 34.9, 90, 8.2, DASH, 8.2, 2.27, 7.4),
    "P0000079": ("Post Commons", 17.9, 8.4, 4.2, 30.5, 90, 7.6, DASH, 7.6, 1.42, 5.4),
    "P0000078": ("Jefferson Waters Creek", 51.7, 23.0, 11.5, 86.2, 90, 20.7, DASH, 20.7, -0.43, -0.9),
    "P0000076": ("The Court at Deptford", 25.0, 9.5, 9.5, 44.1, 90, 8.6, DASH, 8.6, 2.21, 7.6),
    "PDFONLY_PEGADD": ("Pegasus Life Storage - Add'l", DASH, 24.2, 0.0, 24.2, 90, 21.7, DASH, 21.7, 0.91, 2.8),
    "P0000077": ("Jefferson Addison Heights", 43.9, 24.8, 12.6, 81.3, 90, 22.3, DASH, 22.3, -0.48, -1.0),
    "P0000080": ("Prestige Storage Portfolio", 20.3, 11.3, 5.6, 37.2, 45, 5.1, DASH, 5.1, 1.15, 7.3),
    "P0000075": ("Camp Creek", 52.6, 21.9, 7.3, 81.8, 90, 19.7, DASH, 19.7, 4.51, 7.9),
    "P0000081": ("Addison Princeton Meadows", 76.6, 35.0, 18.8, 130.4, 45, 15.8, DASH, 15.8, 2.02, 4.6),
    "P0000084": ("Cocoplum Apartments", 62.2, 23.4, 23.4, 108.9, 45, 10.5, DASH, 10.5, 1.14, 4.1),
    "P0000082": ("Poplar Prairie", 26.7, 12.8, 6.3, 45.9, 90, 11.5, DASH, 11.5, 2.17, 7.2),
    "P0000087": ("The Standard", 30.3, 11.0, 4.6, 45.8, 90, 9.9, DASH, 9.9, 1.33, 5.4),
    "P0000085": ("Jefferson Eastchase", 53.9, 29.4, 14.7, 98.0, 61, 18.0, DASH, 18.0, -0.22, -1.0),
    "P0000086": ("Flats at Dorsett Ridge", 35.2, 15.5, 6.5, 57.1, 90, 13.9, DASH, 13.9, 1.33, 4.7),
    "P0000088": ("Seasons at Bel Air", 88.8, 35.5, 14.7, 138.9, 46, 14.1, 2.2, 16.4, 1.35, 5.9),
    "P0000089": ("45th & Main", 47.0, 18.6, 13.8, 79.4, 90, 16.7, DASH, 16.7, -0.20, -1.0),
    "P0000099": ("Glenmoore Apartments", 36.7, 20.8, 6.9, 64.4, 90, 18.7, DASH, 18.7, 0.74, 3.3),
    "P0000100": ("Green Valley Ranch", 51.5, 20.0, 12.7, 84.2, 90, 18.0, DASH, 18.0, -0.12, -1.0),
    "P0000107": ("Town Fair Tire Portfolio", 20.0, 9.1, 4.5, 33.6, 90, 8.1, DASH, 8.1, 0.67, 6.6),
    "P0000109": ("Burton Retail Portfolio", 75.3, 26.6, 11.4, 113.3, 90, 23.9, DASH, 23.9, 1.67, 10.1),
    "P0000110": ("Trolley Square", 30.8, 6.8, 6.8, 13.5, 90, 3.8, 2.3, 6.1, -0.02, -1.1),
    "P0000114": ("Jefferson Stephens", 50.0, 22.7, 11.3, 84.0, 90, 2.6, 17.9, 20.4, -0.01, -1.1),
    "P0000118": ("Hanestowne Village", 17.8, 3.9, 2.1, 23.8, 90, 3.5, DASH, 3.5, -0.01, -1.1),
    "P0000116": ("Plaza Del Mar", 27.6, 13.6, 6.8, 48.0, 90, 12.2, DASH, 12.2, -0.02, -1.1),
}
PAGE2_SUBTOTALS = {
    "Individual Investments": ("Total Individual Investments", 270.5, 100.1, 60.1, 430.6, 70, 63.4, 6.5, 69.9, 15.33, 6.8),
    "TGA22": ("Total PSC TGA 2022 LLC", 268.4, 133.0, 57.5, 458.8, 82, 97.5, 12.0, 109.5, 8.95, 3.4),
    "TGA23": ("Total PSC TGA 2023 LLC", 366.5, 169.5, 93.3, 629.3, 67, 112.7, DASH, 112.7, 11.62, 4.4),
    "TGA24": ("Total PSC TGA 2024 LLC", 279.2, 119.3, 59.1, 457.5, 77, 89.6, 2.2, 91.8, 3.77, 3.6),
    "TGA25": ("Total PSC TGA 2025 LLC", 201.5, 73.6, 38.4, 282.6, 90, 46.0, 20.2, 66.2, 1.62, 8.2),
}
PAGE2_TOTAL = ("Portfolio Totals", 1386.0, 595.3, 308.4, 2258.9, 68, 404.2, 40.9, 445.1, 41.3, 4.4)
#: only three columns are populated on the excluding-development row
PAGE2_EXDEV = {"total_commitment": 299.3, "itd": 41.3, "net_roe": 5.8}

# ── page 3: Operating ────────────────────────────────────────────────────
# vcode -> (name, occ_at_close, noi_at_close, occ_uw, noi_uw, occ_proj,
#           noi_proj, expected_growth, actual_growth)
OP_COLS = ("occ_at_close", "noi_at_close", "occ_uw", "noi_uw",
           "occ_proj", "noi_proj", "expected_growth", "actual_growth")
PAGE3 = {
    "P0000019": ("Giant 7", 97.9, 8.8, 98.3, 9.3, 97.8, 9.4, 5.3, 6.4),
    "P0000017": ("East Manchester", 80.8, 1.0, 95.5, 1.3, 93.2, 1.5, 36.2, 57.2),
    "P0000021": ("JB Fair Park", NA, DASH, NA, DASH, NA, NA, NA, NA),
    "P0000030": ("Nottingham Village", 84.3, 2.1, 91.1, 3.6, 89.6, 3.2, 71.7, 53.2),
    "P0000018": ("Evergreen Plaza", 94.0, 5.3, 92.9, 6.5, 96.2, 6.3, 21.7, 17.6),
    "PCITWES": ("City West", 84.3, 3.0, NA, NA, NA, NA, NA, NA),
    "P0000065": ("Ascent on Steamboat", 89.7, 2.4, 92.8, 3.7, 94.4, 2.9, 49.9, 19.0),
    "P0000066": ("Pegasus Life Storage", 48.0, DASH, 88.3, 1.7, 90.3, 1.1, NA, NA),
    "P0000067": ("Brainerd Place Apartments", NA, DASH, NA, NA, NA, NA, NA, NA),
    "P0000068": ("The Point at Plymouth Meeting", 95.0, 4.2, 94.8, 6.6, 91.6, 4.7, 55.1, 11.9),
    "P0000069": ("Mount Prospect Plaza", 93.5, 2.6, 95.6, 2.4, 96.7, 2.8, -8.8, 7.7),
    "P0000079": ("Post Commons", 92.0, 1.8, 97.9, 2.3, 99.6, 2.2, 28.2, 22.4),
    "P0000078": ("Jefferson Waters Creek", NA, DASH, NA, 2.9, NA, 2.3, NA, NA),
    "P0000076": ("The Court at Deptford", 87.4, 3.6, 95.6, 3.9, 92.7, 4.0, 8.8, 10.8),
    "PDFONLY_PEGADD": ("Pegasus Life Storage - Add'l", 48.0, DASH, 88.3, 1.7, NA, 1.1, NA, NA),
    "P0000077": ("Jefferson Addison Heights", NA, DASH, NA, DASH, NA, NA, NA, NA),
    "P0000080": ("Prestige Storage Portfolio", 92.3, 2.1, 91.4, 2.7, 83.3, 2.4, 28.2, 17.1),
    "P0000075": ("Camp Creek", 94.2, 6.4, 95.5, 6.9, 94.9, 6.8, 9.0, 6.8),
    "P0000081": ("Addison Princeton Meadows", 92.3, 6.1, 88.6, 8.0, 91.2, 6.4, 30.7, 4.7),
    "P0000084": ("Cocoplum Apartments", 90.2, 5.4, 92.9, 7.3, 90.0, 5.5, 34.5, 1.5),
    "P0000082": ("Poplar Prairie", 89.2, 3.3, 94.7, 3.9, 91.4, 3.9, 20.0, 19.7),
    "P0000087": ("The Standard", 87.7, 2.6, 91.5, 3.7, 90.9, 3.1, 43.9, 20.4),
    "P0000085": ("Jefferson Eastchase", NA, DASH, NA, DASH, NA, NA, NA, NA),
    "P0000086": ("Flats at Dorsett Ridge", 90.0, 3.2, 94.0, 3.9, 91.1, 3.3, 19.9, 1.5),
    "P0000088": ("Seasons at Bel Air", 90.7, 7.3, 86.4, 8.4, 89.9, 8.2, 15.0, 12.1),
    "P0000089": ("45th & Main", NA, DASH, NA, DASH, NA, NA, NA, NA),
    "P0000099": ("Glenmoore Apartments", 89.1, 2.9, 89.9, 3.9, 90.4, 3.2, 33.1, 3.8),
    "P0000100": ("Green Valley Ranch", NA, DASH, NA, DASH, NA, NA, NA, NA),
    "P0000107": ("Town Fair Tire Portfolio", 92.0, 2.3, 98.7, 2.9, 96.8, 2.4, 26.2, 5.7),
    "P0000109": ("Burton Retail Portfolio", 95.1, 8.9, 95.1, 8.9, 95.8, 8.9, 0.1, 0.1),
    "P0000110": ("Trolley Square", NA, DASH, NA, DASH, NA, NA, NA, NA),
    "P0000114": ("Jefferson Stephens", NA, DASH, NA, DASH, NA, NA, NA, NA),
    "P0000118": ("Hanestowne Village", NA, DASH, NA, DASH, NA, NA, NA, NA),
    "P0000116": ("Plaza Del Mar", NA, DASH, NA, DASH, NA, NA, NA, NA),
}
PAGE3_SUBTOTALS = {
    "Individual Investments": ("Total Individual Investments", 90.8, 22.6, 94.6, 26.0, 95.1, 24.4, 15.0, 7.9),
    "TGA22": ("Total PSC TGA 2022 LLC", 88.1, 12.2, 94.8, 18.0, 93.8, 17.1, 47.4, 40.7),
    "TGA23": ("Total PSC TGA 2023 LLC", 91.4, 25.9, 92.0, 32.6, 90.9, 28.2, 25.9, 9.0),
    "TGA24": ("Total PSC TGA 2024 LLC", 90.4, 15.8, 90.0, 19.1, 91.0, 17.1, 21.0, 8.1),
    "TGA25": ("Total PSC TGA 2025 LLC", 95.1, 8.9, 95.1, 8.9, 95.8, 8.9, 0.1, 0.1),
}
PAGE3_TOTAL = ("Portfolio Totals", 91.8, 62.7, 93.1, 104.6, 92.9, 95.7, 66.7, 52.6)

# ── page 4: Loan ─────────────────────────────────────────────────────────
# vcode -> (name, debt, ltv, dscr, debt_yield, rate, maturity)
LOAN_COLS = ("debt", "ltv", "ytd_dscr", "debt_yield", "rate", "maturity")
PAGE4 = {
    "P0000019": ("Giant 7", 95.1, 70.9, NA, NA, "3.9% fixed", "12/6/2029"),
    "P0000017": ("East Manchester", 9.6, 48.9, NA, NA, "3.7% fixed", "1/11/2031"),
    "P0000021": ("JB Fair Park", 48.98, "Dev", "Dev", "Dev", "0.0% fixed", NA),
    "P0000030": ("Nottingham Village", 38.9, 79.1, 1.1, 7.0, "SOFR + 350", "6/1/2026"),
    "P0000018": ("Evergreen Plaza", 45.4, 56.0, 2.9, 10.1, "3.5% fixed", "9/23/2031"),
    "PCITWES": ("City West", NA, NA, NA, NA, "SOFR + 400", NA),
    "P0000065": ("Ascent on Steamboat", 32.5, 64.5, 2.1, 8.8, "3.7% fixed", "7/1/2026"),
    "P0000066": ("Pegasus Life Storage", DASH, NA, NA, NA, NA, NA),
    "P0000067": ("Brainerd Place Apartments", 89.5, "Dev", "Dev", "Dev", "Various", "Various"),
    "P0000068": ("The Point at Plymouth Meeting", 63.1, 71.1, 1.7, 7.5, "4.4% fixed", "7/12/2032"),
    "P0000069": ("Mount Prospect Plaza", 21.2, 53.7, 1.3, 9.0, "5.4% fixed", "8/18/2027"),
    "P0000079": ("Post Commons", 17.9, 53.2, 1.2, 10.7, "6.3% fixed", "11/1/2027"),
    "P0000078": ("Jefferson Waters Creek", 51.7, 57.5, "Dev", "Dev", "SOFR + 300", "12/6/2026"),
    "P0000076": ("The Court at Deptford", 25.0, 50.1, 2.3, 14.9, "6.5% fixed", "1/1/2028"),
    "PDFONLY_PEGADD": ("Pegasus Life Storage - Add'l", DASH, NA, NA, NA, NA, NA),
    "P0000077": ("Jefferson Addison Heights", 43.9, "Dev", "Dev", "Dev", "SOFR + 300", "3/3/2027"),
    "P0000080": ("Prestige Storage Portfolio", 20.3, 53.1, 2.5, 12.5, "5.0% fixed", "3/23/2028"),
    "P0000075": ("Camp Creek", 52.6, 58.2, 1.7, 13.5, "6.9% fixed", "6/2/2028"),
    "P0000081": ("Addison Princeton Meadows", 76.6, 59.9, 1.4, 7.7, "5.3% fixed", "8/1/2033"),
    "P0000084": ("Cocoplum Apartments", 62.2, 62.4, 1.3, 7.3, "5.5% fixed", "10/1/2028"),
    "P0000082": ("Poplar Prairie", 26.7, 50.4, 1.4, 11.5, "7.5% fixed", "9/28/2027"),
    "P0000087": ("The Standard", 30.3, 66.7, 1.1, 9.5, "6.7% fixed", "11/1/2030"),
    "P0000085": ("Jefferson Eastchase", 53.9, "Dev", "Dev", "Dev", "WSJ + 0", "11/22/2027"),
    "P0000086": ("Flats at Dorsett Ridge", 35.2, 63.4, 1.4, 8.5, "6.2% fixed", "5/1/2029"),
    "P0000088": ("Seasons at Bel Air", 88.8, 62.0, 1.4, 8.0, "5.7% fixed", "8/1/2034"),
    "P0000089": ("45th & Main", 47.0, "Dev", "Dev", "Dev", "SOFR + 350", "8/9/2028"),
    "P0000099": ("Glenmoore Apartments", 36.7, 63.3, 1.3, 6.9, "5.5% fixed", "3/1/2030"),
    "P0000100": ("Green Valley Ranch", 51.5, "Dev", "Dev", "Dev", "SOFR + 350", "8/20/2029"),
    "P0000107": ("Town Fair Tire Portfolio", 20.0, 57.7, 3.1, 14.7, "SOFR + 350", "2/14/2032"),
    "P0000109": ("Burton Retail Portfolio", 75.3, 69.1, 2.3, 13.1, "5.7% fixed", "8/28/2032"),
    "P0000110": ("Trolley Square", 30.8, "Dev", "Dev", "Dev", "6.3% fixed", "3/12/2029"),
    "P0000114": ("Jefferson Stephens", 50.0, "Dev", "Dev", "Dev", "SOFR + 265", "4/17/2029"),
    "P0000118": ("Hanestowne Village", 17.8, NA, NA, NA, "5.6% fixed", "9/24/2030"),
    "P0000116": ("Plaza Del Mar", 27.6, NA, NA, NA, "SOFR + 400", "3/16/2029"),
}
PAGE4_SUBTOTALS = {
    "Individual Investments": ("Total Individual Investments", 270.5, 67.4, NA, 4.1, None, None),
    "TGA22": ("Total PSC TGA 2022 LLC", 268.4, 60.4, 1.6, 9.6, None, None),
    "TGA23": ("Total PSC TGA 2023 LLC", 366.5, 59.4, 1.5, 9.7, None, None),
    "TGA24": ("Total PSC TGA 2024 LLC", 279.2, 62.1, 1.5, 8.6, None, None),
    "TGA25": ("Total PSC TGA 2025 LLC", 201.5, 69.1, 2.3, 4.9, None, None),
}
PAGE4_TOTAL = ("Portfolio Totals", 1386.0, 62.8, 1.6, 8.7, None, None)

#: Names the PDF uses that differ from ours, for the workbook's Deal column.
ALIASES = {
    "P0000099": "ReNew Glenmoore",
    "P0000100": "Green Valley Ranch & Telluride",
    "P0000118": "Hanestowne Waterstone",
    "P0000109": "Burton Portfolio",
}
