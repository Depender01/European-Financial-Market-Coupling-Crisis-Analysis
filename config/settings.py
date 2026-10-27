"""Project Settings and Configuration.

European Financial Market Coupling & Crisis Analysis
Defines instrument metadata, directory paths, default parameters, and historical crisis windows.
"""

from pathlib import Path
from typing import Dict, Any, Optional

# -----------------------------------------------------------------------------
# Base Directory Structure (using pathlib for cross-platform reliability)
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Ensure runtime directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# European Equity Market Instruments Manifest
# -----------------------------------------------------------------------------
# CORE BASKET: 8 major European cash indices with continuous coverage from >= 1998 to present.
# Note on Return Types:
# - ^GDAXI (DAX) is a Performance Index (Total Return, dividends reinvested).
# - ^FTSE, ^FCHI, ^SSMI, ^AEX, ^IBEX, FTSEMIB, ^BFX are Price Return Indices (dividends excluded).
# This distinction is documented and accounted for in return analytics.
CORE_MARKETS: Dict[str, Dict[str, Any]] = {
    "^FTSE": {
        "name": "FTSE 100",
        "country": "United Kingdom",
        "currency": "GBP",
        "index_type": "Price Return",
        "exchange": "London Stock Exchange (LSE)",
        "earliest_available": "1984-01-03",
        "notes": "UK large-cap benchmark. High weighting toward commodities and global financials.",
    },
    "^GDAXI": {
        "name": "DAX 40 (Total Return)",
        "country": "Germany",
        "currency": "EUR",
        "index_type": "Total Return / Performance",
        "exchange": "Deutsche Börse (Xetra)",
        "earliest_available": "1987-12-30",
        "notes": (
            "German blue-chip benchmark. Reinvests gross dividends into index level. "
            "Empirical audit confirms daily log return correlation with DAX Price Return (^GDAXIP) is 0.9989, "
            "with a mean daily dividend drift of +1.15 bps (~2.89% annualized)."
        ),
    },
    "^FCHI": {
        "name": "CAC 40",
        "country": "France",
        "currency": "EUR",
        "index_type": "Price Return",
        "exchange": "Euronext Paris",
        "earliest_available": "1990-03-01",
        "notes": "French benchmark. High weighting in luxury, energy, and aerospace.",
    },
    "^SSMI": {
        "name": "SMI 20",
        "country": "Switzerland",
        "currency": "CHF",
        "index_type": "Price Return",
        "exchange": "SIX Swiss Exchange",
        "earliest_available": "1990-11-09",
        "notes": "Swiss blue-chip benchmark. Defensively weighted toward healthcare and consumer staples.",
    },
    "^AEX": {
        "name": "AEX 25",
        "country": "Netherlands",
        "currency": "EUR",
        "index_type": "Price Return",
        "exchange": "Euronext Amsterdam",
        "earliest_available": "1992-10-12",
        "notes": "Dutch benchmark. High technology and semiconductor manufacturing weight (ASML).",
    },
    "^IBEX": {
        "name": "IBEX 35",
        "country": "Spain",
        "currency": "EUR",
        "index_type": "Price Return",
        "exchange": "Bolsa de Madrid (BME)",
        "earliest_available": "1993-07-12",
        "notes": "Spanish benchmark. Heavily weighted toward retail banking and utilities.",
    },
    "FTSEMIB.MI": {
        "name": "FTSE MIB",
        "country": "Italy",
        "currency": "EUR",
        "index_type": "Price Return",
        "exchange": "Borsa Italiana (Milan)",
        "earliest_available": "1997-12-31",
        "notes": "Italian benchmark. Heavy financial and industrial exposure.",
    },
    "^BFX": {
        "name": "BEL 20",
        "country": "Belgium",
        "currency": "EUR",
        "index_type": "Price Return",
        "exchange": "Euronext Brussels",
        "earliest_available": "1991-04-09",
        "notes": "Belgian benchmark. Healthcare, chemicals, and financial exposure.",
    },
}

# EXTENDED INSTRUMENTS (with specific historical coverage or proxy caveats):
EXTENDED_MARKETS: Dict[str, Dict[str, Any]] = {
    "^OMX": {
        "name": "OMX Stockholm 30",
        "country": "Sweden",
        "currency": "SEK",
        "index_type": "Price Return",
        "exchange": "Nasdaq Stockholm",
        "earliest_available": "2008-11-20",
        "notes": "Local cash index. Limited Yahoo Finance history before November 2008.",
    },
    "EWD": {
        "name": "iShares MSCI Sweden ETF",
        "country": "Sweden (Proxy ETF)",
        "currency": "USD",
        "index_type": "Net Total Return ETF",
        "exchange": "NYSE Arca",
        "earliest_available": "1996-03-18",
        "notes": "WARNING: Traded on US hours in USD; subject to USD/SEK FX noise and tracking error.",
    },
    "^OMXC25": {
        "name": "OMX Copenhagen 25",
        "country": "Denmark",
        "currency": "DKK",
        "index_type": "Price Return",
        "exchange": "Nasdaq Copenhagen",
        "earliest_available": "2016-12-19",
        "notes": "Replaced OMXC20 in Dec 2016. Ticker ^OMXC20PI is defunct/delisted on Yahoo Finance.",
    },
    "^GDAXIP": {
        "name": "DAX 40 (Price Return / Kursindex)",
        "country": "Germany (Price Return)",
        "currency": "EUR",
        "index_type": "Price Return",
        "exchange": "Deutsche Börse (Xetra)",
        "earliest_available": "2013-03-05",
        "notes": "Pure price-return version of the DAX (dividends excluded). Available from March 2013 onward.",
    },
}

# Pan-European Benchmark Options:
BENCHMARK_CONFIG: Dict[str, Dict[str, Any]] = {
    "EQUAL_WEIGHTED": {
        "name": "Equal-Weighted European Basket (Synthetic)",
        "currency": "Local/Blended",
        "earliest_available": "1998-01-01",
        "notes": "Calculated as daily mean return of all available core markets. Recommended for unbroken coverage.",
    },
    "^STOXX50E": {
        "name": "EURO STOXX 50",
        "country": "Eurozone",
        "currency": "EUR",
        "index_type": "Price Return",
        "exchange": "Eurex / STOXX",
        "earliest_available": "2007-03-30",
        "notes": "Premier Eurozone blue-chip index. Coverage on Yahoo Finance begins March 2007.",
    },
}

# -----------------------------------------------------------------------------
# Default Date Configuration
# -----------------------------------------------------------------------------
DEFAULT_START_DATE: str = "2000-01-01"
DEFAULT_END_DATE: Optional[str] = None  # None indicates up to latest available date

# -----------------------------------------------------------------------------
# Data Processing Configuration
# -----------------------------------------------------------------------------
# Calendar alignment policies:
# - "intersection": Keep only dates where ALL selected markets were actively trading (recommended to prevent synthetic zero returns).
# - "drop_all_nan": Keep dates where at least one market traded, without filling missing prices.
DEFAULT_CALENDAR_ALIGNMENT: str = "intersection"

# Maximum missing data proportion permitted for an instrument before dropping it
MAX_MISSING_TOLERANCE: float = 0.25

# -----------------------------------------------------------------------------
# Historical Crisis and Stress Event Windows (for Event-State Analysis)
# -----------------------------------------------------------------------------
# Documented European and global macro-financial events:
HISTORICAL_CRISIS_WINDOWS: Dict[str, Dict[str, Any]] = {
    "DotCom_Crash": {
        "name": "Dot-Com Crash / Telecom Bust",
        "start": "2000-03-10",
        "end": "2002-10-09",
        "description": "Peak of technology valuation bubble followed by multi-year equity bear market.",
    },
    "GFC_Lehman": {
        "name": "Global Financial Crisis (Lehman Collapse)",
        "start": "2007-08-09",
        "end": "2009-03-09",
        "description": "Subprime mortgage contagion, Lehman Brothers collapse, and global liquidity freeze.",
    },
    "Euro_Debt_Crisis": {
        "name": "European Sovereign Debt Crisis (PIIGS)",
        "start": "2010-04-23",
        "end": "2012-07-26",
        "description": "Greek bailout, contagion to Ireland, Portugal, Spain, Italy; ends with Draghi 'Whatever it takes'.",
    },
    "Grexit_China_Slowdown": {
        "name": "Grexit Referendum & China Yuan Devaluation",
        "start": "2015-06-01",
        "end": "2016-02-11",
        "description": "Greek capital controls, commodity price collapse, and PBOC currency devaluation.",
    },
    "Brexit_Referendum": {
        "name": "UK Brexit Referendum Shock",
        "start": "2016-06-23",
        "end": "2016-07-15",
        "description": "Shock UK referendum vote to exit the European Union; sterling depreciation.",
    },
    "COVID_Crash": {
        "name": "COVID-19 Pandemic Market Crash",
        "start": "2020-02-19",
        "end": "2020-03-23",
        "description": "Global economic lockdown panic, extreme volatility spike, and swift liquidity collapse.",
    },
    "Ukraine_War_Inflation": {
        "name": "Russia-Ukraine War & European Energy Shock",
        "start": "2022-01-03",
        "end": "2022-10-14",
        "description": "Russian invasion of Ukraine, natural gas supply shock, and rapid ECB/BoE rate hikes.",
    },
}

# -----------------------------------------------------------------------------
# Rolling Analysis Parameters
# -----------------------------------------------------------------------------
DEFAULT_ROLLING_WINDOW_M: int = 20     # Rolling epoch duration in trading days (~1 calendar month)
DEFAULT_ROLLING_SHIFT_DELTA: int = 1   # Rolling stride in trading days (daily continuous trajectory)
DEFAULT_LONG_WINDOW: int = 63          # Quarterly rolling window (~3 calendar months)
RANDOM_SEED: int = 42                  # Global seed for stochastic routines
