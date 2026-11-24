#!/usr/bin/env python
# rs_data.py — Fixed version + Questrade API support
# Changes vs original:
#   1. Python 3.10+ compatible (removed FTP, fixed imports)
#   2. NASDAQ ticker list via HTTPS instead of FTP
#   3. Batch yfinance downloads (100 tickers at a time) → ~25 min instead of hours
#   4. Removed broken session/user-agent hack (curl_cffi handles this transparently)
#   5. pandas 2.x compatible (MultiIndex column handling)
#   6. Questrade API data source (DATA_SOURCE: QUESTRADE) with incremental candle updates,
#      symbolId caching, direct access-token auth, and preserved Yahoo sector/industry.

import requests
import json
import time
import datetime as dt
import os
import yaml
import yfinance as yf
import pandas as pd
import dateutil.relativedelta
import numpy as np
import re
from io import StringIO
from time import sleep

from datetime import date
from datetime import datetime, timezone, timedelta

DIR = os.path.dirname(os.path.realpath(__file__))

if not os.path.exists(os.path.join(DIR, 'data')):
    os.makedirs(os.path.join(DIR, 'data'))
if not os.path.exists(os.path.join(DIR, 'data_persist')):
    os.makedirs(os.path.join(DIR, 'data_persist'))
if not os.path.exists(os.path.join(DIR, 'tmp')):
    os.makedirs(os.path.join(DIR, 'tmp'))

# ── Config ──────────────────────────────────────────────────────────────────

try:
    with open(os.path.join(DIR, 'config_private.yaml'), 'r') as stream:
        private_config = yaml.safe_load(stream)
except FileNotFoundError:
    private_config = None
except yaml.YAMLError as exc:
    print(exc)

try:
    with open('config.yaml', 'r') as stream:
        config = yaml.safe_load(stream)
except FileNotFoundError:
    config = None
except yaml.YAMLError as exc:
    print(exc)

def cfg(key):
    try:
        return private_config[key]
    except Exception:
        try:
            return config[key]
        except Exception:
            return None

def read_json(json_file):
    if not os.path.exists(json_file):
        return {}
    with open(json_file, "r", encoding="utf-8") as fp:
        return json.load(fp)

# ── Constants ────────────────────────────────────────────────────────────────

API_KEY = cfg("API_KEY")
TD_API = "https://api.tdameritrade.com/v1/marketdata/%s/pricehistory"
PRICE_DATA_FILE = os.path.join(DIR, "data", "price_history.json")
REFERENCE_TICKER = cfg("REFERENCE_TICKER")
DATA_SOURCE = cfg("DATA_SOURCE")
ALL_STOCKS = cfg("USE_ALL_LISTED_STOCKS")
TICKER_INFO_FILE = os.path.join(DIR, "data_persist", "ticker_info.json")
TICKER_INFO_DICT = read_json(TICKER_INFO_FILE)
QUESTRADE_SYMBOLS_FILE = os.path.join(DIR, "data_persist", "questrade_symbols.json")
REF_TICKER = {
    "ticker": REFERENCE_TICKER,
    "sector": "--- Reference ---",
    "industry": "--- Reference ---",
    "universe": "--- Reference ---"
}

UNKNOWN = "unknown"

# Exchange code mapping: yfinance codes → human-readable exchange names
YFINANCE_EXCHANGE_MAP = {
    "NYQ": "NYSE",
    "NMS": "NASDAQ",
    "NCM": "NASDAQ",
    "NGM": "NASDAQ",
    "ASE": "AMEX",
    "ARC": "NYSE ARCA",
    "BTS": "BATS",
}

# Batch size for yfinance bulk downloads — 100 is a sweet spot (speed vs reliability)
BATCH_SIZE = 100

# ── Ticker list retrieval ────────────────────────────────────────────────────

def _find_ticker_sector_cols(df, ticker_candidates, sector_candidates, industry_candidates):
    """
    Detect column names dynamically from a list of possible names.
    Wikipedia occasionally renames columns — this makes the code resilient to that.
    """
    cols = [str(c).strip() for c in df.columns]

    def find(candidates):
        for c in candidates:
            for col in cols:
                if c.lower() in col.lower():
                    return col
        return None

    ticker_col    = find(ticker_candidates)
    sector_col    = find(sector_candidates)
    industry_col  = find(industry_candidates)
    return ticker_col, sector_col, industry_col


def get_sp500_tickers(universe="S&P 500"):
    """
    Parse S&P 500 list from Wikipedia using pd.read_html() — robust against HTML structure changes.
    Columns on the page: Symbol | Security | GICS Sector | GICS Sub-Industry | ...
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; rs-rating-bot/1.0)"}
    try:
        tables = pd.read_html(url, header=0, attrs={"class": "wikitable"}, storage_options={"User-Agent": headers["User-Agent"]})
    except Exception:
        resp = requests.get(url, headers=headers, timeout=30)
        tables = pd.read_html(StringIO(resp.text), header=0)

    df = tables[0]
    ticker_col, sector_col, industry_col = _find_ticker_sector_cols(
        df,
        ticker_candidates=["Symbol", "Ticker", "Tick"],
        sector_candidates=["GICS Sector", "Sector"],
        industry_candidates=["GICS Sub-Industry", "Sub-Industry", "Industry"]
    )

    secs = {}
    for _, row in df.iterrows():
        ticker = str(row[ticker_col]).strip().replace(".", "-") if ticker_col else None
        if not ticker or not re.match(r'^[A-Z\-]+$', ticker):
            continue
        secs[ticker] = {
            "ticker": ticker,
            "sector":   str(row[sector_col]).strip()   if sector_col   else UNKNOWN,
            "industry": str(row[industry_col]).strip() if industry_col else UNKNOWN,
            "universe": universe
        }
    print(f"  S&P 500: {len(secs)} tickers loaded from Wikipedia.")
    return secs


def get_nq100_tickers(universe="Nasdaq 100"):
    """
    Parse Nasdaq-100 list from Wikipedia.
    Columns: Company | Ticker | GICS Sector | GICS Sub-Industry | ...
    """
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; rs-rating-bot/1.0)"}
    try:
        tables = pd.read_html(url, header=0, storage_options={"User-Agent": headers["User-Agent"]})
    except Exception:
        resp = requests.get(url, headers=headers, timeout=30)
        tables = pd.read_html(StringIO(resp.text), header=0)

    df = max(tables, key=lambda t: len(t))
    ticker_col, sector_col, industry_col = _find_ticker_sector_cols(
        df,
        ticker_candidates=["Ticker", "Symbol", "Tick"],
        sector_candidates=["GICS Sector", "Sector"],
        industry_candidates=["GICS Sub-Industry", "Sub-Industry", "Industry"]
    )

    secs = {}
    for _, row in df.iterrows():
        ticker = str(row[ticker_col]).strip().replace(".", "-") if ticker_col else None
        if not ticker or not re.match(r'^[A-Z\-]+$', ticker):
            continue
        secs[ticker] = {
            "ticker": ticker,
            "sector":   str(row[sector_col]).strip()   if sector_col   else UNKNOWN,
            "industry": str(row[industry_col]).strip() if industry_col else UNKNOWN,
            "universe": universe
        }
    print(f"  Nasdaq 100: {len(secs)} tickers loaded from Wikipedia.")
    return secs


def get_sp_midsmall_tickers(url, universe):
    """
    Generic parser for S&P 400 and S&P 600 Wikipedia pages.
    Columns vary but always contain ticker, sector, and industry info.
    """
    headers = {"User-Agent": "Mozilla/5.0 (compatible; rs-rating-bot/1.0)"}
    try:
        tables = pd.read_html(url, header=0, storage_options={"User-Agent": headers["User-Agent"]})
    except Exception:
        resp = requests.get(url, headers=headers, timeout=30)
        tables = pd.read_html(StringIO(resp.text), header=0)

    df = max(tables, key=lambda t: len(t))
    ticker_col, sector_col, industry_col = _find_ticker_sector_cols(
        df,
        ticker_candidates=["Ticker", "Symbol", "Tick"],
        sector_candidates=["GICS Sector", "Sector"],
        industry_candidates=["GICS Sub-Industry", "Sub-Industry", "Industry"]
    )

    secs = {}
    for _, row in df.iterrows():
        ticker = str(row[ticker_col]).strip().replace(".", "-") if ticker_col else None
        if not ticker or not re.match(r'^[A-Z\-]+$', ticker):
            continue
        secs[ticker] = {
            "ticker": ticker,
            "sector":   str(row[sector_col]).strip()   if sector_col   else UNKNOWN,
            "industry": str(row[industry_col]).strip() if industry_col else UNKNOWN,
            "universe": universe
        }
    print(f"  {universe}: {len(secs)} tickers loaded from Wikipedia.")
    return secs


def get_resolved_securities():
    tickers = {REFERENCE_TICKER: REF_TICKER}
    if ALL_STOCKS:
        return get_tickers_from_nasdaq(tickers)
    else:
        return get_tickers_from_wikipedia(tickers)

def get_tickers_from_wikipedia(tickers):
    """
    Load index components from Wikipedia using pd.read_html() with dynamic column detection.
    """
    if cfg("NQ100"):
        try:
            tickers.update(get_nq100_tickers())
        except Exception as e:
            print(f"⚠ Could not load Nasdaq-100 from Wikipedia: {e}")
    if cfg("SP500"):
        try:
            tickers.update(get_sp500_tickers())
        except Exception as e:
            print(f"⚠ Could not load S&P 500 from Wikipedia: {e}")
    if cfg("SP400"):
        try:
            tickers.update(get_sp_midsmall_tickers(
                "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies", "S&P 400"
            ))
        except Exception as e:
            print(f"⚠ Could not load S&P 400 from Wikipedia: {e}")
    if cfg("SP600"):
        try:
            tickers.update(get_sp_midsmall_tickers(
                "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies", "S&P 600"
            ))
        except Exception as e:
            print(f"⚠ Could not load S&P 600 from Wikipedia: {e}")
    return tickers

def exchange_from_symbol(symbol):
    """Map NASDAQ trader file single-letter exchange codes to display names."""
    mapping = {"Q": "NASDAQ", "A": "NYSE ARCA", "N": "NYSE", "P": "NYSE ARCA", "Z": "BATS", "V": "IEXG"}
    return mapping.get(symbol, "n/a")

def exchange_from_yfinance(code):
    """Map yfinance exchange codes (e.g. NYQ, NMS) to display names."""
    return YFINANCE_EXCHANGE_MAP.get(code, code if code else "n/a")

def get_tickers_from_nasdaq(tickers):
    """
    Download the NASDAQ ticker list via HTTPS (replaces old FTP method which was fragile).
    URL: https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt
    """
    print("Downloading NASDAQ ticker list via HTTPS...")
    url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt"
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        lines = r.text.splitlines()
    except Exception as e:
        print(f"HTTPS download failed ({e}), trying fallback URL...")
        url2 = "https://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqtraded.txt"
        r = requests.get(url2, timeout=60)
        lines = r.text.splitlines()

    ticker_col = 1
    etf_col = 5
    exchange_col = 3
    test_col = 7

    count_added = 0
    for entry in lines[1:]:
        values = entry.split('|')
        if len(values) <= max(ticker_col, etf_col, exchange_col, test_col):
            continue
        ticker = values[ticker_col].strip()
        if (re.match(r'^[A-Z]+$', ticker)
                and values[etf_col].strip() == "N"
                and values[test_col].strip() == "N"):
            sec = {
                "ticker": ticker,
                "sector": UNKNOWN,
                "industry": UNKNOWN,
                "universe": exchange_from_symbol(values[exchange_col].strip())
            }
            tickers[ticker] = sec
            count_added += 1

    print(f"NASDAQ list: {count_added} tickers loaded.")
    return tickers

SECURITIES = get_resolved_securities().values()

# ── File helpers ─────────────────────────────────────────────────────────────

def write_to_file(dict_data, file):
    with open(file, "w", encoding='utf8') as fp:
        json.dump(dict_data, fp, ensure_ascii=False)

def write_price_history_file(tickers_dict):
    write_to_file(tickers_dict, PRICE_DATA_FILE)

def write_ticker_info_file(info_dict):
    write_to_file(info_dict, TICKER_INFO_FILE)

def enrich_ticker_data(ticker_response, security):
    ticker_response["sector"] = security["sector"]
    ticker_response["industry"] = security["industry"]
    ticker_response["universe"] = security["universe"]

# ── Progress helpers ──────────────────────────────────────────────────────────

def print_data_progress(label, idx, total, elapsed_s, remaining_s):
    dt_ref = datetime.fromtimestamp(0)
    dt_e = datetime.fromtimestamp(elapsed_s)
    elapsed = dateutil.relativedelta.relativedelta(dt_e, dt_ref)
    if remaining_s and not np.isnan(remaining_s):
        dt_r = datetime.fromtimestamp(remaining_s)
        remaining = dateutil.relativedelta.relativedelta(dt_r, dt_ref)
        remaining_string = f'{remaining.hours}h {remaining.minutes}m {remaining.seconds}s'
    else:
        remaining_string = "?"
    print(f'[{idx}/{total}] {label} — Elapsed: {elapsed.hours}h {elapsed.minutes}m {elapsed.seconds}s | Remaining: {remaining_string}')

def get_remaining_seconds(all_load_times, idx, total):
    if idx == 0:
        return float('nan')
    load_time_ma = pd.Series(all_load_times).rolling(min(idx + 1, 25)).mean().tail(1).item()
    return (total - idx) * load_time_ma

def escape_ticker(ticker):
    return ticker.replace(".", "-")

# ── Industry/sector info ──────────────────────────────────────────────────────

def get_info_from_dict(d, key):
    return d[key] if key in d else "n/a"

def _safe_float(d, key):
    """Return float value from dict or None if missing/invalid."""
    try:
        v = d.get(key) if hasattr(d, 'get') else getattr(d, key, None)
        return float(v) if v is not None else None
    except (TypeError, ValueError, AttributeError):
        return None

def load_ticker_info(ticker, info_dict):
    """
    Fetch and cache ticker metadata from yfinance.
    """
    escaped = escape_ticker(ticker)
    result = {
        "industry": "n/a", "sector": "n/a",
        "marketCap": None, "floatShares": None,
        "shortPercentOfFloat": None,
        "fiftyTwoWeekHigh": None, "fiftyTwoWeekLow": None,
        "revenueGrowth": None,
    }

    try:
        t = yf.Ticker(escaped)

        try:
            fi = t.fast_info
            result["marketCap"]        = _safe_float(fi, "marketCap")
            result["fiftyTwoWeekHigh"] = _safe_float(fi, "yearHigh")
            result["fiftyTwoWeekLow"]  = _safe_float(fi, "yearLow")
            result["floatShares"]      = _safe_float(fi, "shares")
        except Exception:
            pass

        try:
            info_obj = t.info
            result["industry"]            = get_info_from_dict(info_obj, "industry")
            result["sector"]              = get_info_from_dict(info_obj, "sector")
            result["shortPercentOfFloat"] = (
                _safe_float(info_obj, "shortPercentOfFloat") or
                _safe_float(info_obj, "shortRatio")
            )
            result["revenueGrowth"] = _safe_float(info_obj, "revenueGrowth")
            if result["marketCap"] is None:
                result["marketCap"] = _safe_float(info_obj, "marketCap")
            result["exchange"] = exchange_from_yfinance(info_obj.get("exchange", ""))
        except Exception:
            pass

    except Exception:
        pass

    info_dict[ticker] = {"info": result}

def needs_refresh(ticker, info_dict):
    """
    True if cached entry is missing the new market data fields.
    """
    try:
        info = info_dict[ticker]["info"]
        return "marketCap" not in info or "revenueGrowth" not in info or "exchange" not in info
    except (KeyError, TypeError):
        return True

# ── Core: Questrade API client & price loader ─────────────────────────────────

class QuestradeClient:
    """
    Minimal Questrade REST client using a manually supplied access_token & api_server.
      - Rate-limit pacing (safe for 20 req/s market data, 30 req/s symbol calls)
      - 429 retry honoring X-RateLimit-Reset
    """
    def __init__(self, access_token, api_server="https://api01.iq.questrade.com/"):
        self.access_token = (access_token or "").strip()
        if not self.access_token:
            raise ValueError(
                "Missing Questrade access token. Set QUESTRADE_ACCESS_TOKEN env var "
                "(or in config_private.yaml)."
            )
        self.api_server = (api_server or "https://api01.iq.questrade.com/").rstrip("/") + "/"
        self.session = requests.Session()
        self._last_market_call = 0.0

    def _get(self, path, params=None, is_market_data=True, max_retries=4):
        url = f"{self.api_server}{path.lstrip('/')}"
        headers = {"Authorization": f"Bearer {self.access_token}"}

        for attempt in range(max_retries):
            # Pace market-data calls to ~15 req/sec (well below 20 req/sec limit)
            if is_market_data:
                elapsed = time.time() - self._last_market_call
                if elapsed < 0.068:
                    sleep(0.068 - elapsed)
                self._last_market_call = time.time()

            resp = self.session.get(url, headers=headers, params=params, timeout=30)

            if resp.status_code == 200:
                return resp.json()

            if resp.status_code == 401:
                raise RuntimeError(
                    "Questrade access token expired or invalid (HTTP 401). "
                    "Generate a fresh access_token and retry."
                )

            if resp.status_code == 429:
                reset_ts = resp.headers.get("X-RateLimit-Reset")
                wait_s = 2.0
                if reset_ts:
                    try:
                        wait_s = max(1.0, min(15.0, float(reset_ts) - time.time() + 0.5))
                    except ValueError:
                        wait_s = 2.0
                print(f"  ⏳ Questrade 429 rate limit — waiting {wait_s:.1f}s...")
                sleep(wait_s)
                continue

            if resp.status_code >= 500:
                sleep(1.5 * (attempt + 1))
                continue

            # 4xx (e.g., 404 symbol not found) — don't retry
            return None

        return None

    def get_symbols_by_names(self, names):
        """Batch lookup up to ~50 symbol names at once via GET v1/symbols?names=..."""
        data = self._get("v1/symbols", params={"names": ",".join(names)}, is_market_data=False)
        return data.get("symbols", []) if data else []

    def get_symbols_by_ids(self, symbol_ids):
        """Batch lookup up to ~50 symbol IDs at once via GET v1/symbols?ids=..."""
        ids_str = ",".join(str(sid) for sid in symbol_ids)
        data = self._get("v1/symbols", params={"ids": ids_str}, is_market_data=False)
        return data.get("symbols", []) if data else []

    def get_daily_candles(self, symbol_id, start_dt, end_dt):
        """Fetch OneDay OHLCV candles for a numeric symbolId."""
        params = {
            "startTime": start_dt.strftime("%Y-%m-%dT00:00:00-05:00"),
            "endTime": end_dt.strftime("%Y-%m-%dT23:59:59-05:00"),
            "interval": "OneDay",
        }
        data = self._get(f"v1/markets/candles/{symbol_id}", params=params, is_market_data=True)
        return data.get("candles", []) if data else []


def _to_questrade_symbol(ticker):
    """Wikipedia uses BRK-B; Questrade uses BRK.B."""
    return ticker.replace("-", ".")


def _parse_qt_candles(raw_candles):
    """Convert Questrade candle objects to the exact schema expected by rs_ranking.py."""
    candles = []
    for c in raw_candles:
        close = c.get("close")
        if close is None:
            continue
        start_str = c.get("start", "")
        try:
            ts = int(datetime.fromisoformat(start_str).timestamp())
        except Exception:
            ts = 0
        candles.append({
            "open":     float(c["open"])   if c.get("open")   is not None else None,
            "close":    float(close),
            "low":      float(c["low"])    if c.get("low")    is not None else None,
            "high":     float(c["high"])   if c.get("high")   is not None else None,
            "volume":   float(c["volume"]) if c.get("volume") is not None else None,
            "datetime": ts,
        })
    return candles


def _merge_candles(existing_candles, new_candles, max_days=400):
    """Merge existing and newly fetched daily candles by calendar date, keeping last max_days."""
    by_day = {}
    for c in existing_candles:
        day_key = datetime.fromtimestamp(c["datetime"], tz=timezone.utc).strftime("%Y-%m-%d")
        by_day[day_key] = c
    for c in new_candles:
        day_key = datetime.fromtimestamp(c["datetime"], tz=timezone.utc).strftime("%Y-%m-%d")
        by_day[day_key] = c
    merged = [by_day[k] for k in sorted(by_day)]
    return merged[-max_days:]


def load_prices_from_questrade(securities, info={}):
    """
    Fetch closing prices and volumes from Questrade API.
    - Reuses existing price_history.json and does incremental 7-day updates for known tickers.
    - Preserves existing sector/industry in ticker_info.json (only falls back to yfinance
      for brand-new tickers not yet in ticker_info.json).
    - Refreshes marketCap and fiftyTwoWeekHigh from Questrade's batch /v1/symbols endpoint.
    """
    print("*** Loading Stocks from Questrade API ***")
    access_token = os.environ.get("QUESTRADE_ACCESS_TOKEN") or cfg("QUESTRADE_ACCESS_TOKEN")
    api_server = os.environ.get("QUESTRADE_API_SERVER") or cfg("QUESTRADE_API_SERVER") or "https://api01.iq.questrade.com/"
    qt = QuestradeClient(access_token, api_server)

    securities_list = list(securities)
    ref_sec = next((s for s in securities_list if s["ticker"] == REFERENCE_TICKER), None)
    non_ref = [s for s in securities_list if s["ticker"] != REFERENCE_TICKER]
    ordered = ([ref_sec] if ref_sec else []) + non_ref

    existing_prices = read_json(PRICE_DATA_FILE)
    symbol_map = read_json(QUESTRADE_SYMBOLS_FILE)  # {ticker: symbolId}

    # 1. Resolve missing ticker -> symbolId mappings in batches of 50
    missing_tickers = [s["ticker"] for s in ordered if s["ticker"] not in symbol_map]
    if missing_tickers:
        print(f"Resolving {len(missing_tickers)} new symbol IDs from Questrade...")
        for i in range(0, len(missing_tickers), 50):
            chunk = missing_tickers[i:i + 50]
            qt_names = [_to_questrade_symbol(t) for t in chunk]
            qt_to_orig = {_to_questrade_symbol(t).upper(): t for t in chunk}
            for sym in qt.get_symbols_by_names(qt_names):
                name = str(sym.get("symbol", "")).upper()
                orig = qt_to_orig.get(name)
                if orig and sym.get("symbolId"):
                    symbol_map[orig] = sym["symbolId"]
        write_to_file(symbol_map, QUESTRADE_SYMBOLS_FILE)
        print(f"✓ Symbol ID cache updated ({len(symbol_map)} total).")

    # 2. Refresh marketCap & fiftyTwoWeekHigh in batches of 50 WITHOUT touching sector/industry
    resolved_sids = [symbol_map[s["ticker"]] for s in ordered if s["ticker"] in symbol_map]
    sid_to_ticker = {symbol_map[s["ticker"]]: s["ticker"] for s in ordered if s["ticker"] in symbol_map}
    print(f"Refreshing marketCap & 52wk high for {len(resolved_sids)} symbols...")
    for i in range(0, len(resolved_sids), 50):
        chunk_ids = resolved_sids[i:i + 50]
        for sym in qt.get_symbols_by_ids(chunk_ids):
            ticker = sid_to_ticker.get(sym.get("symbolId"))
            if not ticker:
                continue
            if ticker not in TICKER_INFO_DICT:
                load_ticker_info(ticker, TICKER_INFO_DICT)
            info_entry = TICKER_INFO_DICT.setdefault(ticker, {}).setdefault("info", {
                "industry": "n/a", "sector": "n/a",
                "marketCap": None, "floatShares": None,
                "shortPercentOfFloat": None,
                "fiftyTwoWeekHigh": None, "fiftyTwoWeekLow": None,
                "revenueGrowth": None,
            })
            if sym.get("marketCap") is not None:
                info_entry["marketCap"] = float(sym["marketCap"])
            if sym.get("highPrice52") is not None:
                info_entry["fiftyTwoWeekHigh"] = float(sym["highPrice52"])
            if sym.get("lowPrice52") is not None:
                info_entry["fiftyTwoWeekLow"] = float(sym["lowPrice52"])

    # 3. Fetch daily candles (incremental 7-day fetch if history exists, 18-month backfill if new)
    today = date.today()
    full_start = today - timedelta(days=1 * 365 + 183)
    incr_start = today - timedelta(days=7)

    tickers_dict = {}
    failed_tickers = []
    total = len(ordered)
    global_start = time.time()

    for idx, sec in enumerate(ordered, start=1):
        ticker = sec["ticker"]
        sid = symbol_map.get(ticker)
        if not sid:
            failed_tickers.append(ticker)
            continue

        prev_candles = existing_prices.get(ticker, {}).get("candles", [])
        use_incremental = len(prev_candles) >= 120
        start_dt = incr_start if use_incremental else full_start

        raw_candles = qt.get_daily_candles(sid, start_dt, today)
        new_candles = _parse_qt_candles(raw_candles)

        if use_incremental:
            candles = _merge_candles(prev_candles, new_candles)
        else:
            candles = new_candles

        if not candles:
            failed_tickers.append(ticker)
            continue

        ticker_data = {"candles": candles}
        enrich_ticker_data(ticker_data, sec)

        if ticker not in TICKER_INFO_DICT:
            load_ticker_info(ticker, TICKER_INFO_DICT)

        try:
            ticker_data["industry"] = TICKER_INFO_DICT[ticker]["info"]["industry"]
            ticker_data["sector"]   = TICKER_INFO_DICT[ticker]["info"]["sector"]
        except (KeyError, TypeError):
            ticker_data["industry"] = "Unknown"

        try:
            yf_exchange = TICKER_INFO_DICT[ticker]["info"].get("exchange")
            if yf_exchange and ticker_data.get("universe") in (None, "n/a", UNKNOWN):
                ticker_data["universe"] = yf_exchange
        except (KeyError, TypeError):
            pass

        tickers_dict[ticker] = ticker_data

        if idx % 250 == 0 or idx == total:
            elapsed = time.time() - global_start
            rate = idx / elapsed if elapsed > 0 else 1
            rem_s = (total - idx) / rate
            print(f"[{idx}/{total}] {len(tickers_dict)} OK | Elapsed: {int(elapsed // 60)}m {int(elapsed % 60)}s | Remaining: {int(rem_s // 60)}m {int(rem_s % 60)}s")

        if idx % 500 == 0:
            write_price_history_file(tickers_dict)
            write_ticker_info_file(TICKER_INFO_DICT)

    write_price_history_file(tickers_dict)
    write_ticker_info_file(TICKER_INFO_DICT)

    total_time = time.time() - global_start
    print(f"\n✓ Questrade done: {len(tickers_dict)} tickers updated in {int(total_time // 60)}m {int(total_time % 60)}s")
    if failed_tickers:
        print(f"✗ {len(failed_tickers)} tickers had no Questrade data: {failed_tickers[:20]}{'...' if len(failed_tickers) > 20 else ''}")
        with open(os.path.join(DIR, "failed_tickers.txt"), "w") as f:
            f.write("\n".join(failed_tickers))

    return tickers_dict

# ── Core: batch price download from Yahoo ─────────────────────────────────────

def candles_from_df_row(row):
    """Convert a single-row dict from the batch DataFrame into a candle dict."""
    return {
        "open":     float(row["Open"])   if not pd.isna(row["Open"])   else None,
        "close":    float(row["Close"])  if not pd.isna(row["Close"])  else None,
        "low":      float(row["Low"])    if not pd.isna(row["Low"])    else None,
        "high":     float(row["High"])   if not pd.isna(row["High"])   else None,
        "volume":   float(row["Volume"]) if not pd.isna(row["Volume"]) else None,
        "datetime": int(row.name.timestamp()) if hasattr(row.name, 'timestamp') else 0,
    }

def parse_batch_download(df, batch_tickers):
    """
    Parse the result of yf.download() for multiple tickers.
    yfinance returns a MultiIndex DataFrame: columns = (Field, Ticker).
    Returns a dict: {ticker: [candle, ...]}
    """
    result = {}

    if df is None or df.empty:
        return result

    if not isinstance(df.columns, pd.MultiIndex):
        ticker = batch_tickers[0]
        df_t = df.copy()
        if df_t.empty:
            return result
        candles = []
        for ts, row in df_t.iterrows():
            if pd.isna(row.get("Close")):
                continue
            candle = {
                "open":     float(row["Open"])   if not pd.isna(row["Open"])   else None,
                "close":    float(row["Close"]),
                "low":      float(row["Low"])    if not pd.isna(row["Low"])    else None,
                "high":     float(row["High"])   if not pd.isna(row["High"])   else None,
                "volume":   float(row["Volume"]) if not pd.isna(row["Volume"]) else None,
                "datetime": int(ts.timestamp()),
            }
            candles.append(candle)
        if candles:
            result[ticker] = candles
        return result

    df_swapped = df.swaplevel(axis=1)

    for ticker in batch_tickers:
        yf_ticker = escape_ticker(ticker)
        try:
            df_t = df_swapped[yf_ticker] if yf_ticker in df_swapped.columns.get_level_values(0) else df_swapped.get(yf_ticker)
            if df_t is None or df_t.empty:
                continue
            df_t = df_t.dropna(subset=["Close"])
            if df_t.empty:
                continue
            candles = []
            for ts, row in df_t.iterrows():
                candle = {
                    "open":     float(row["Open"])   if "Open"   in row and not pd.isna(row["Open"])   else None,
                    "close":    float(row["Close"]),
                    "low":      float(row["Low"])    if "Low"    in row and not pd.isna(row["Low"])    else None,
                    "high":     float(row["High"])   if "High"   in row and not pd.isna(row["High"])   else None,
                    "volume":   float(row["Volume"]) if "Volume" in row and not pd.isna(row["Volume"]) else None,
                    "datetime": int(ts.timestamp()),
                }
                candles.append(candle)
            if candles:
                result[ticker] = candles
        except (KeyError, TypeError):
            pass

    return result

def load_prices_from_yahoo(securities, info={}):
    """
    Main download function for Yahoo Finance.
    """
    print("*** Loading Stocks from Yahoo Finance (batch mode) ***")
    today = date.today()
    start_date = today - dt.timedelta(days=1 * 365 + 183)  # 18 months of data
    end_date = today + dt.timedelta(days=1)  # end is exclusive in yfinance

    securities_list = list(securities)
    tickers_dict = {}
    failed_tickers = []

    ref_ticker = REFERENCE_TICKER
    ref_sec = next((s for s in securities_list if s["ticker"] == ref_ticker), None)
    non_ref = [s for s in securities_list if s["ticker"] != ref_ticker]

    ordered = ([ref_sec] if ref_sec else []) + non_ref

    batches = []
    batch_securities = []
    for sec in ordered:
        batch_securities.append(sec)
        if len(batch_securities) >= BATCH_SIZE:
            batches.append(batch_securities)
            batch_securities = []
    if batch_securities:
        batches.append(batch_securities)

    total_batches = len(batches)
    global_start = time.time()
    batch_times = []

    for batch_idx, batch in enumerate(batches):
        batch_start = time.time()
        batch_tickers = [escape_ticker(s["ticker"]) for s in batch]
        original_tickers = [s["ticker"] for s in batch]

        print(f"\n── Batch {batch_idx + 1}/{total_batches}: {len(batch)} tickers ──")

        try:
            df = yf.download(
                tickers=batch_tickers,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False,
                threads=True,
                ignore_tz=True,
            )
        except Exception as e:
            print(f"Batch {batch_idx + 1} failed: {e}")
            failed_tickers.extend(original_tickers)
            continue

        candles_by_ticker = parse_batch_download(df, batch_tickers)
        escaped_to_original = {escape_ticker(t): t for t in original_tickers}

        for esc_ticker, candles in candles_by_ticker.items():
            original = escaped_to_original.get(esc_ticker, esc_ticker)
            sec = next((s for s in batch if s["ticker"] == original), None)
            if sec is None:
                continue

            ticker_data = {"candles": candles}
            enrich_ticker_data(ticker_data, sec)

            if original not in TICKER_INFO_DICT or needs_refresh(original, TICKER_INFO_DICT):
                try:
                    load_ticker_info(original, TICKER_INFO_DICT)
                except Exception as e:
                    print(f"  Could not load info for {original}: {e}")

            try:
                ticker_data["industry"] = TICKER_INFO_DICT[original]["info"]["industry"]
                ticker_data["sector"]   = TICKER_INFO_DICT[original]["info"]["sector"]
            except (KeyError, TypeError):
                ticker_data["industry"] = "Unknown"

            try:
                yf_exchange = TICKER_INFO_DICT[original]["info"].get("exchange")
                if yf_exchange and ticker_data.get("universe") in (None, "n/a", UNKNOWN):
                    ticker_data["universe"] = yf_exchange
            except (KeyError, TypeError):
                pass

            tickers_dict[original] = ticker_data

        for t in original_tickers:
            if t not in tickers_dict:
                failed_tickers.append(t)

        batch_elapsed = time.time() - batch_start
        batch_times.append(batch_elapsed)
        total_elapsed = time.time() - global_start
        processed_batches = batch_idx + 1
        avg_batch_time = total_elapsed / processed_batches
        remaining_batches = total_batches - processed_batches
        remaining_s = avg_batch_time * remaining_batches

        remaining_str = f"{int(remaining_s // 60)}m {int(remaining_s % 60)}s" if remaining_s else "?"
        tickers_ok = sum(1 for t in original_tickers if t in tickers_dict)
        print(f"  {tickers_ok}/{len(batch)} tickers OK | Batch time: {batch_elapsed:.1f}s | Remaining: {remaining_str}")

        if (batch_idx + 1) % 10 == 0:
            print(f"  → Saving intermediate results ({len(tickers_dict)} tickers so far)...")
            write_price_history_file(tickers_dict)
            write_ticker_info_file(TICKER_INFO_DICT)

    write_price_history_file(tickers_dict)
    write_ticker_info_file(TICKER_INFO_DICT)

    total_time = time.time() - global_start
    print(f"\n✓ Done: {len(tickers_dict)} tickers downloaded in {int(total_time // 60)}m {int(total_time % 60)}s")
    if failed_tickers:
        unique_failed = list(dict.fromkeys(failed_tickers))
        print(f"✗ {len(unique_failed)} tickers had no data (likely delisted): {unique_failed[:20]}{'...' if len(unique_failed) > 20 else ''}")
        with open(os.path.join(DIR, "failed_tickers.txt"), "w") as f:
            f.write("\n".join(unique_failed))

    return tickers_dict

# ── TD Ameritrade (kept for compatibility, not actively maintained) ────────────

def tda_params(apikey, period_type="year", period=2, frequency_type="daily", frequency=1):
    return (
        ("apikey", apikey),
        ("periodType", period_type),
        ("period", period),
        ("frequencyType", frequency_type),
        ("frequency", frequency)
    )

def load_prices_from_tda(securities, api_key, info={}):
    print("*** Loading Stocks from TD Ameritrade ***")
    headers = {"Cache-Control": "no-cache"}
    params = tda_params(api_key)
    tickers_dict = {}
    start = time.time()
    load_times = []

    for idx, sec in enumerate(securities):
        ticker = sec["ticker"]
        r_start = time.time()
        response = requests.get(TD_API % ticker, params=params, headers=headers)
        ticker_data = response.json()
        if ticker not in TICKER_INFO_DICT:
            load_ticker_info(ticker, TICKER_INFO_DICT)
            write_ticker_info_file(TICKER_INFO_DICT)
        ticker_data["industry"] = TICKER_INFO_DICT[ticker]["info"]["industry"]
        now = time.time()
        load_times.append(now - r_start)
        enrich_ticker_data(ticker_data, sec)
        tickers_dict[sec["ticker"]] = ticker_data
        error_text = f' Error {response.status_code}' if response.status_code != 200 else ''
        remaining_s = get_remaining_seconds(load_times, idx, len(list(securities)))
        print_data_progress(f"{ticker}{error_text}", idx + 1, len(list(securities)), now - start, remaining_s)
        if info.get("forceTDA"):
            sleep(0.4)

    write_price_history_file(tickers_dict)

# ── Entry point ───────────────────────────────────────────────────────────────

def save_data(source, securities, api_key, info={}):
    if source == "YAHOO":
        load_prices_from_yahoo(securities, info)
    elif source == "TD_AMERITRADE":
        load_prices_from_tda(securities, api_key, info)
    elif source == "QUESTRADE":
        load_prices_from_questrade(securities, info)

def main(forceTDA=False, api_key=API_KEY):
    dataSource = DATA_SOURCE if not forceTDA else "TD_AMERITRADE"
    save_data(dataSource, SECURITIES, api_key, {"forceTDA": forceTDA})
    write_ticker_info_file(TICKER_INFO_DICT)

if __name__ == "__main__":
    main()