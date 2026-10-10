import os
import re
import difflib
from datetime import datetime
from io import StringIO

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from dotenv import load_dotenv
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
load_dotenv()

# On Streamlit Community Cloud, secrets are configured in the app settings.
# Copy them to environment variables so Groq/OpenFIGI clients can read them.
for _secret_name in ("GROQ_API_KEY", "OPENFIGI_API_KEY"):
    try:
        if not os.getenv(_secret_name) and _secret_name in st.secrets:
            os.environ[_secret_name] = str(st.secrets[_secret_name])
    except Exception:
        # Local development may not have Streamlit secrets configured.
        pass

st.set_page_config(
    page_title="AI Financial Analyst Agent",
    page_icon="📈",
    layout="wide",
)

st.title("📈 AI Financial Analyst Agent")
st.caption(
    "Research stocks, compare fundamentals, explore historical performance, "
    "and monitor a simple portfolio. Market data may be delayed."
)

# --------------------------------------------------
# STATE HELPERS
# --------------------------------------------------
def now_local():
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def initialize_state():
    defaults = {
        "messages": [],
        "prompt_draft": "",
        "submitted_prompt": None,
        "last_report": "",
        "last_report_csv": "",
        "last_report_filename": "financial_research_report.md",
        "holdings": {},
        "alerts": [],
        "activity_log": [],
        "pending_chat_corrections": None,
        "pending_chart_corrections": None,
        "pending_holding_corrections": None,
        "pending_alert_corrections": None,
        "ticker_verification_overrides": [],
        "global_lookup_results": [],
        "global_lookup_query": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_state()


def select_suggestion(prompt_text):
    """Put a suggestion into the editable prompt box without sending it."""
    st.session_state["prompt_draft"] = prompt_text


def submit_prompt():
    """Save the prompt from the form so it can be processed after rerun."""
    draft = st.session_state.get("prompt_draft", "").strip()
    st.session_state["submitted_prompt"] = draft or None
    st.session_state["prompt_draft"] = ""


def log_activity(lines, message):
    lines.append(f"{now_local()} — {message}")


# --------------------------------------------------
# AI FINANCE AGENT
# --------------------------------------------------
@st.cache_resource
def create_agent():
    return Agent(
        name="Finance Agent",
        model=Groq(id="openai/gpt-oss-120b"),
        tools=[
            YFinanceTools(
                stock_price=True,
                analyst_recommendations=True,
                stock_fundamentals=True,
                company_news=True,
            )
        ],
        instructions=[
            "You are a financial research assistant, not a broker.",
            "Use available tools to retrieve current or recent financial information.",
            "Separate sourced facts from your interpretation and calculations.",
            "Present financial comparisons in Markdown tables when useful.",
            "Include source URLs and the relevant period/date when available.",
            "Never invent missing financial figures, events, analyst ratings, or citations.",
            "Mention when data is unavailable, delayed, or not directly comparable.",
            "Explain important risks and uncertainty; do not promise returns.",
            "Do not present the response as individualized investment advice.",
        ],
        markdown=True,
        show_tool_calls=True,
    )


agent = create_agent()

# --------------------------------------------------
# YAHOO FINANCE DATA FUNCTIONS
# --------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_history(symbol, period="1y"):
    """Fetch adjusted price history. Results are cached for five minutes."""
    ticker = yf.Ticker(symbol)
    history = ticker.history(period=period, auto_adjust=True)
    if history is None or history.empty or "Close" not in history.columns:
        return pd.DataFrame()
    return history.dropna(subset=["Close"])


@st.cache_data(ttl=60, show_spinner=False)
def get_latest_quote(symbol):
    """Return latest available close, its timestamp, and currency if available."""
    history = yf.Ticker(symbol).history(period="5d", auto_adjust=False)
    if history is None or history.empty or "Close" not in history.columns:
        raise ValueError(f"No recent price data was returned for {symbol}.")

    close_values = history["Close"].dropna()
    if close_values.empty:
        raise ValueError(f"No valid closing price was returned for {symbol}.")

    price = float(close_values.iloc[-1])
    timestamp = str(close_values.index[-1])
    currency = "Unknown"
    try:
        fast_info = yf.Ticker(symbol).fast_info
        possible_currency = fast_info.get("currency")
        if possible_currency:
            currency = str(possible_currency)
    except Exception:
        pass

    return price, timestamp, currency


@st.cache_data(ttl=300, show_spinner=False)
def get_fundamentals(tickers):
    """Retrieve a selected set of company fundamentals for comparison."""
    metrics = {
        "Company": "longName",
        "Sector": "sector",
        "Industry": "industry",
        "Currency": "currency",
        "Market Capitalization": "marketCap",
        "Revenue (TTM)": "totalRevenue",
        "Net Income (TTM)": "netIncomeToCommon",
        "Gross Profit": "grossProfits",
        "Gross Margin": "grossMargins",
        "Operating Margin": "operatingMargins",
        "Net Profit Margin": "profitMargins",
        "Return on Equity": "returnOnEquity",
        "Return on Assets": "returnOnAssets",
        "Total Debt": "totalDebt",
        "Cash and Equivalents": "totalCash",
        "Free Cash Flow": "freeCashflow",
        "Trailing P/E": "trailingPE",
        "Forward P/E": "forwardPE",
        "Price to Book": "priceToBook",
        "Enterprise Value": "enterpriseValue",
        "EV / EBITDA": "enterpriseToEbitda",
        "Earnings Per Share (TTM)": "trailingEps",
        "Dividend Yield": "dividendYield",
        "Beta": "beta",
        "52-Week High": "fiftyTwoWeekHigh",
        "52-Week Low": "fiftyTwoWeekLow",
        "Current Price": "currentPrice",
    }
    records = []
    errors = []

    for symbol in tickers:
        try:
            info = yf.Ticker(symbol).get_info()
            if not info:
                errors.append(f"{symbol}: Yahoo Finance returned no company information.")
                continue
            for label, key in metrics.items():
                records.append({
                    "Metric": label,
                    "Ticker": symbol,
                    "Value": info.get(key),
                })
        except Exception as exc:
            errors.append(f"{symbol}: {exc}")

    if not records:
        return pd.DataFrame(), errors

    data = pd.DataFrame(records)
    comparison = data.pivot(index="Metric", columns="Ticker", values="Value")
    ordered = [symbol for symbol in tickers if symbol in comparison.columns]
    return comparison[ordered], errors


def format_value(metric, value):
    if pd.isna(value):
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)

    if metric in {
        "Gross Margin", "Operating Margin", "Net Profit Margin",
        "Return on Equity", "Return on Assets", "Dividend Yield",
    }:
        return f"{number:.2%}"
    if metric in {
        "Market Capitalization", "Revenue (TTM)", "Net Income (TTM)",
        "Gross Profit", "Total Debt", "Cash and Equivalents",
        "Free Cash Flow", "Enterprise Value",
    }:
        return f"{number:,.2f}"
    if metric in {"Trailing P/E", "Forward P/E", "Price to Book", "EV / EBITDA", "Beta"}:
        return f"{number:.2f}"
    if metric in {"Current Price", "52-Week High", "52-Week Low", "Earnings Per Share (TTM)"}:
        return f"{number:,.2f}"
    return str(value)


def is_stock_comparison(prompt):
    lower = prompt.lower()
    return any(word in lower for word in ("compare", "comparison", "versus", " vs ", " vs. "))


def extract_tickers(prompt):
    excluded = {
        "AI", "I", "US", "USA", "TTM", "PE", "EPS", "ROE", "ROA", "EV", "FCF",
        "CEO", "GDP", "THE", "AND", "FOR", "ARE", "WHAT", "HOW", "CAN", "YOU", "VS",
        "NYSE", "NASDAQ", "NSE", "BSE", "ETF", "USD", "INR", "RSI", "EBITDA",
    }
    # Support longer symbols and exchange suffixes such as RELIANCE.NS or BRK.B.
    candidates = re.findall(
        r"(?<![A-Za-z0-9])\$?([A-Z][A-Z0-9.^=-]{0,14})(?![A-Za-z0-9])",
        prompt,
    )
    result = []
    for symbol in candidates:
        if symbol not in excluded and symbol not in result:
            result.append(symbol)
    return result[:4]


# Common symbols help suggest a likely correction even if Yahoo's search is unavailable.
# Yahoo Finance search results are also used when available, so this is not the only universe.
COMMON_TICKERS = {
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "TSLA",
    "AMD", "INTC", "QCOM", "AVGO", "ORCL", "IBM", "CRM", "ADBE", "NFLX",
    "COST", "WMT", "TGT", "HD", "LOW", "KO", "PEP", "MCD", "NKE", "DIS",
    "JPM", "BAC", "WFC", "C", "GS", "MS", "V", "MA", "AXP", "BRK-B",
    "LLY", "JNJ", "MRK", "PFE", "ABBV", "UNH", "CVS", "ISRG", "TMO",
    "XOM", "CVX", "COP", "SLB", "CAT", "DE", "GE", "BA", "LMT", "RTX",
    "F", "GM", "UBER", "ABNB", "PYPL", "SQ", "SHOP", "PLTR", "SNOW", "PANW",
    "CRWD", "COIN", "MSTR", "SPOT", "BABA", "TSM", "NVO", "SAP", "SONY",
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "ITC.NS",
    "BHARTIARTL.NS", "SBIN.NS", "LT.NS", "TATAMOTORS.NS", "WIPRO.NS", "TATACONSUM.NS",
}
TICKER_PATTERN = re.compile(r"[A-Z0-9.^=-]{1,15}")


def normalize_ticker(symbol):
    """Normalize user-entered ticker text without guessing the intended company."""
    return str(symbol or "").strip().upper().replace("$", "").replace(" ", "")


# Yahoo Finance covers many international exchanges, but it is not an official
# exhaustive registry for every listing. OpenFIGI provides an additional global
# securities reference/search layer. Neither source guarantees every market/security.
YAHOO_SUFFIXES = (
    ".NS", ".BO", ".L", ".TO", ".V", ".AX", ".HK", ".SS", ".SZ",
    ".T", ".SI", ".KS", ".KQ", ".TW", ".PA", ".DE", ".MI", ".MC",
    ".AS", ".SW", ".ST", ".CO", ".HE", ".OL", ".BR", ".IR", ".WA",
    ".BD", ".JO", ".SR", ".DU", ".QA", ".BK", ".JK", ".KL", ".NZ",
)

# OpenFIGI exchange identifiers are not the same thing as Yahoo Finance suffixes.
# Only apply a suffix suggestion for exchange codes we explicitly recognize.
OPENFIGI_TO_YAHOO_SUFFIX = {
    "NSE": ".NS", "XNSE": ".NS", "BSE": ".BO", "XBOM": ".BO",
    "LSE": ".L", "XLON": ".L", "TSX": ".TO", "XTSE": ".TO",
    "TSXV": ".V", "XTSX": ".V", "ASX": ".AX", "XASX": ".AX",
    "HKG": ".HK", "XHKG": ".HK", "TKS": ".T", "XTKS": ".T",
    "SGX": ".SI", "XSES": ".SI", "TAI": ".TW", "XTAI": ".TW",
    "PAR": ".PA", "XPAR": ".PA", "ETR": ".DE", "XETR": ".DE",
    "MIL": ".MI", "XMIL": ".MI", "MAD": ".MC", "XMAD": ".MC",
    "AMS": ".AS", "XAMS": ".AS", "SWX": ".SW", "XSWX": ".SW",
    "STO": ".ST", "XSTO": ".ST", "CPH": ".CO", "XCSE": ".CO",
    "HEL": ".HE", "XHEL": ".HE", "OSL": ".OL", "XOSL": ".OL",
    "JSE": ".JO", "XJSE": ".JO", "IDX": ".JK", "XIDX": ".JK",
    "KLS": ".KL", "XKLS": ".KL", "BKK": ".BK", "XBKK": ".BK",
    "WSE": ".WA", "XWAR": ".WA", "SAU": ".SR", "XSAU": ".SR",
    "DFM": ".DU", "XDFM": ".DU", "QSE": ".QA", "XDOH": ".QA",
}


def yahoo_base_symbol(symbol):
    """Remove a recognized Yahoo exchange suffix, preserving share-class dots."""
    normalized = normalize_ticker(symbol)
    for suffix in YAHOO_SUFFIXES:
        if normalized.endswith(suffix):
            return normalized[:-len(suffix)]
    return normalized


@st.cache_data(ttl=900, show_spinner=False)
def search_yahoo_directory(query, max_results=20):
    """Search Yahoo Finance's global quote directory; results are best-effort."""
    query = str(query or "").strip()
    if not query:
        return []

    raw_quotes = []
    try:
        search_class = getattr(yf, "Search", None)
        if search_class is not None:
            result = search_class(query, max_results=max_results, news_count=0)
            raw_quotes = list(getattr(result, "quotes", None) or [])
    except Exception:
        raw_quotes = []

    # Fallback for yfinance versions that do not expose yf.Search, or if its
    # wrapper call fails. This Yahoo endpoint is not an official public API.
    if not raw_quotes:
        try:
            response = requests.get(
                "https://query1.finance.yahoo.com/v1/finance/search",
                params={"q": query, "quotesCount": max_results, "newsCount": 0, "enableFuzzyQuery": "true"},
                headers={"User-Agent": "Mozilla/5.0 financial-analyst-app"},
                timeout=5,
            )
            response.raise_for_status()
            raw_quotes = response.json().get("quotes", [])
        except Exception:
            raw_quotes = []

    cleaned = []
    seen = set()
    for quote in raw_quotes:
        if not isinstance(quote, dict):
            continue
        symbol = normalize_ticker(quote.get("symbol", ""))
        if not symbol or not TICKER_PATTERN.fullmatch(symbol) or symbol in seen:
            continue
        seen.add(symbol)
        quote_type = str(quote.get("quoteType") or quote.get("typeDisp") or "").strip()
        # Exclude obvious non-security results, but tolerate missing type metadata.
        if quote_type and any(token in quote_type.lower() for token in (
            "currency", "cryptocurrency", "future", "option", "index", "fund family"
        )):
            continue
        cleaned.append({
            "symbol": symbol,
            "company": str(quote.get("longname") or quote.get("shortname") or quote.get("name") or ""),
            "exchange": str(quote.get("exchDisp") or quote.get("exchange") or quote.get("exchangeName") or "Exchange not supplied"),
            "security_type": quote_type or "Not supplied",
            "source": "Yahoo Finance search",
        })
    return cleaned[:max_results]


@st.cache_data(ttl=900, show_spinner=False)
def search_openfigi_directory(query, max_results=10):
    """Search OpenFIGI's global security reference directory.

    An OPENFIGI_API_KEY in .env is optional and provides a higher rate limit.
    Without it, the public endpoint can be more rate-limited. This is a reference
    lookup, not a quote feed and not a guaranteed complete exchange-by-exchange registry.
    """
    query = str(query or "").strip()
    if not query:
        return []

    headers = {"Content-Type": "application/json"}
    api_key = os.getenv("OPENFIGI_API_KEY", "").strip()
    if api_key:
        headers["X-OPENFIGI-APIKEY"] = api_key

    try:
        response = requests.post(
            "https://api.openfigi.com/v3/search",
            json={"query": query, "marketSecDes": "Equity"},
            headers=headers,
            timeout=6,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        # The UI must remain usable if this secondary directory is unavailable.
        return []

    raw_results = payload.get("data", []) if isinstance(payload, dict) else []
    cleaned = []
    seen = set()
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        symbol = normalize_ticker(item.get("ticker", ""))
        if not symbol:
            continue
        name = str(item.get("name") or item.get("securityDescription") or "")
        exchange = str(item.get("exchCode") or item.get("micCode") or "Exchange not supplied")
        security_type = str(item.get("securityType2") or item.get("securityType") or "Equity")
        key = (symbol, name, exchange)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append({
            "symbol": symbol,
            "company": name,
            "exchange": exchange,
            "security_type": security_type,
            "source": "OpenFIGI directory",
        })
    return cleaned[:max_results]


def search_global_ticker_directory(query, limit=30):
    """Combine global Yahoo quote search and OpenFIGI reference search."""
    query = str(query or "").strip()
    if not query:
        return []

    combined = []
    seen = set()
    for result in search_yahoo_directory(query, max_results=20) + search_openfigi_directory(query, max_results=15):
        key = (result.get("symbol", ""), result.get("exchange", ""), result.get("source", ""))
        if key not in seen:
            seen.add(key)
            combined.append(result)
    return combined[:limit]


def _matching_global_directory_records(symbol, records):
    """Return exact or same-base-symbol matches without treating fuzzy matches as verification."""
    symbol = normalize_ticker(symbol)
    base = yahoo_base_symbol(symbol)
    exact = []
    for record in records:
        candidate = normalize_ticker(record.get("symbol", ""))
        if not candidate:
            continue
        candidate_base = yahoo_base_symbol(candidate)
        if candidate == symbol or ("." not in symbol and candidate_base == base):
            exact.append(record)
    return exact


@st.cache_data(ttl=300, show_spinner=False)
def _probe_ticker_symbol(symbol):
    """Verify recent price data first, then consult broader global directories.

    A directory match means a security reference was found, not that Yahoo Finance
    can currently supply usable prices. Failed lookups are not proof of delisting.
    """
    symbol = normalize_ticker(symbol)
    ticker = yf.Ticker(symbol)
    errors = []
    company_name = ""

    try:
        history = ticker.history(period="1mo", auto_adjust=False)
        if history is not None and not history.empty and "Close" in history.columns:
            closes = history["Close"].dropna()
            if not closes.empty:
                last_observation = closes.index[-1]
                return {
                    "status": "verified",
                    "symbol": symbol,
                    "company": "",
                    "exchange": "",
                    "matches": [],
                    "reason": f"Yahoo Finance returned price history through {last_observation.date()}.",
                }
    except Exception as exc:
        errors.append(type(exc).__name__)

    info = {}
    try:
        maybe_info = ticker.get_info()
        if isinstance(maybe_info, dict):
            info = maybe_info
        company_name = str(info.get("longName") or info.get("shortName") or "")
        returned_symbol = normalize_ticker(info.get("symbol", ""))
        if returned_symbol and returned_symbol != symbol:
            errors.append(f"metadata resolved to {returned_symbol}, not {symbol}")
    except Exception as exc:
        errors.append(type(exc).__name__)
        returned_symbol = ""

    # Query the two broader directories only when direct recent prices failed.
    matches = search_global_ticker_directory(symbol, limit=25)
    exact_matches = _matching_global_directory_records(symbol, matches)
    if exact_matches:
        match_names = list(dict.fromkeys(
            f"{m.get('company') or m.get('symbol')} ({m.get('exchange')}, {m.get('source')})"
            for m in exact_matches if m.get("symbol")
        ))
        return {
            "status": "directory_match",
            "symbol": symbol,
            "company": company_name or (exact_matches[0].get("company", "")),
            "exchange": str(info.get("exchange") or info.get("fullExchangeName") or ""),
            "matches": exact_matches[:10],
            "reason": (
                "A global security directory returned a matching listing, but Yahoo Finance did not "
                "provide recent prices for this exact symbol. Check the exchange-specific ticker "
                "suggestions below. Directory matches: " + "; ".join(match_names[:4])
            ),
        }

    if info and company_name:
        return {
            "status": "unknown",
            "symbol": symbol,
            "company": company_name,
            "exchange": str(info.get("exchange") or info.get("fullExchangeName") or ""),
            "matches": matches[:10],
            "reason": (
                f"Yahoo Finance returned metadata for {company_name}, but no recent price history. "
                "The quote may be stale, the listing may be inactive, or the exchange suffix may be "
                "wrong. This does not prove the security is delisted."
            ),
        }

    if errors:
        reason = (
            "Direct Yahoo Finance verification failed. The global directories may return possible "
            "matches, but an unavailable response can also be caused by rate limits or network issues."
        )
        status = "unknown"
    else:
        reason = (
            "No recent Yahoo Finance price history or exact global-directory match was found. "
            "Check the spelling, company name, and exchange suffix. This is not proof of delisting."
        )
        status = "unverified"

    return {
        "status": status,
        "symbol": symbol,
        "company": company_name,
        "exchange": str(info.get("exchange") or info.get("fullExchangeName") or ""),
        "matches": matches[:10],
        "reason": reason,
    }


def validate_ticker_symbol(symbol):
    """Central ticker validator reused by chat, charts, holdings, and alerts."""
    normalized = normalize_ticker(symbol)
    if not normalized or not TICKER_PATTERN.fullmatch(normalized):
        return {
            "status": "invalid_format",
            "symbol": normalized,
            "company": "",
            "exchange": "",
            "matches": [],
            "reason": "Ticker format is not supported. Use a symbol such as AAPL, TCS.NS, or 7203.T.",
        }

    if normalized in st.session_state.get("ticker_verification_overrides", []):
        return {
            "status": "accepted_unverified",
            "symbol": normalized,
            "company": "",
            "exchange": "",
            "matches": [],
            "reason": "You explicitly chose to continue even though the data provider could not verify this ticker earlier.",
        }

    return _probe_ticker_symbol(normalized)


INDIAN_EXCHANGE_SUFFIX_HINTS = {
    "TCS": ["TCS.NS", "TCS.BO"],
    "RELIANCE": ["RELIANCE.NS", "RELIANCE.BO"],
    "INFY": ["INFY.NS", "INFY.BO"],
    "HDFCBANK": ["HDFCBANK.NS", "HDFCBANK.BO"],
    "ICICIBANK": ["ICICIBANK.NS", "ICICIBANK.BO"],
    "SBIN": ["SBIN.NS", "SBIN.BO"],
    "ITC": ["ITC.NS", "ITC.BO"],
    "WIPRO": ["WIPRO.NS", "WIPRO.BO"],
    "LT": ["LT.NS", "LT.BO"],
}


@st.cache_data(ttl=600, show_spinner=False)
def suggest_tickers(symbol, limit=5):
    """Suggest exchange-specific Yahoo symbols using global search + typo similarity."""
    symbol = normalize_ticker(symbol)
    candidates = set(COMMON_TICKERS)
    preferred = INDIAN_EXCHANGE_SUFFIX_HINTS.get(symbol, [])
    global_results = search_global_ticker_directory(symbol, limit=30)

    # Yahoo directory results give directly usable quote symbols where available.
    yahoo_results = [r for r in global_results if r.get("source") == "Yahoo Finance search"]
    for result in yahoo_results:
        candidate = normalize_ticker(result.get("symbol", ""))
        if candidate and TICKER_PATTERN.fullmatch(candidate):
            candidates.add(candidate)

    # OpenFIGI uses exchange identifiers rather than Yahoo suffixes. Suggest suffixes
    # only for explicitly mapped exchanges, and only for direct/same ticker matches.
    for result in global_results:
        if result.get("source") != "OpenFIGI directory":
            continue
        candidate = normalize_ticker(result.get("symbol", ""))
        exchange = str(result.get("exchange", "")).upper()
        suffix = OPENFIGI_TO_YAHOO_SUFFIX.get(exchange)
        if suffix and candidate and (candidate == symbol or candidate == yahoo_base_symbol(symbol)):
            candidates.add(f"{candidate}{suffix}")

    ranked = []
    for candidate in candidates:
        if candidate == symbol:
            continue
        base_candidate = yahoo_base_symbol(candidate)
        ratio = max(
            difflib.SequenceMatcher(None, symbol, candidate).ratio(),
            difflib.SequenceMatcher(None, symbol, base_candidate).ratio(),
        )
        exact_base = base_candidate == symbol or candidate == symbol
        # Keep exact base/share-exchange matches even if suffix makes similarity lower.
        if exact_base or ratio >= 0.55:
            ranked.append((1 if exact_base else 0, ratio, candidate))

    ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
    ordered = []
    for candidate in preferred + [candidate for _, _, candidate in ranked]:
        if candidate != symbol and candidate not in ordered:
            ordered.append(candidate)
    return ordered[:limit]

def make_ticker_issue(symbol):
    result = validate_ticker_symbol(symbol)
    if result["status"] in {"verified", "accepted_unverified"}:
        return None
    suggestions = suggest_tickers(symbol) if result["status"] != "invalid_format" else []
    return {
        "symbol": normalize_ticker(symbol),
        "status": result["status"],
        "reason": result["reason"],
        "suggestions": suggestions,
        "matches": result.get("matches", []),
    }


def replace_ticker_in_text(text, old_symbol, new_symbol):
    """Replace only a whole ticker token, preserving the rest of the user's wording."""
    pattern = re.compile(rf"(?<![A-Za-z0-9])\$?{re.escape(old_symbol)}(?![A-Za-z0-9])")
    return pattern.sub(new_symbol, text)


def remove_pending_issue(target, symbol):
    state_key = f"pending_{target}_corrections"
    pending = st.session_state.get(state_key)
    if not pending:
        return
    pending["items"] = [
        item for item in pending.get("items", [])
        if item.get("symbol") != symbol
    ]
    st.session_state[state_key] = pending if pending["items"] else None


def apply_ticker_suggestion(target, old_symbol, new_symbol):
    """Callback: explicitly apply a suggested symbol to the relevant input/draft."""
    state_key = f"pending_{target}_corrections"
    pending = st.session_state.get(state_key)
    if not pending:
        return

    if target == "chat":
        corrected_text = replace_ticker_in_text(pending.get("text", ""), old_symbol, new_symbol)
        pending["text"] = corrected_text
        st.session_state["prompt_draft"] = corrected_text
    elif target == "chart":
        symbols = pending.get("symbols", [])
        pending["symbols"] = [new_symbol if item == old_symbol else item for item in symbols]
        st.session_state["chart_symbols_input"] = ", ".join(pending["symbols"])
    elif target == "holding":
        st.session_state["holding_symbol_input"] = new_symbol
    elif target == "alert":
        st.session_state["alert_symbol_input"] = new_symbol

    remove_pending_issue(target, old_symbol)


def allow_unverified_ticker(target, symbol):
    """Callback: allow a ticker explicitly when provider verification is inconclusive."""
    normalized = normalize_ticker(symbol)
    overrides = list(st.session_state.get("ticker_verification_overrides", []))
    if normalized not in overrides:
        overrides.append(normalized)
    st.session_state["ticker_verification_overrides"] = overrides

    pending = st.session_state.get(f"pending_{target}_corrections")
    if pending:
        if target == "chat":
            st.session_state["prompt_draft"] = pending.get("text", "")
        elif target == "chart":
            st.session_state["chart_symbols_input"] = ", ".join(pending.get("symbols", []))
        elif target == "holding":
            st.session_state["holding_symbol_input"] = normalized
        elif target == "alert":
            st.session_state["alert_symbol_input"] = normalized
    remove_pending_issue(target, normalized)


def dismiss_ticker_issue(target, symbol):
    """Dismiss a suggestion without changing the ticker, so it can be edited manually."""
    pending = st.session_state.get(f"pending_{target}_corrections")
    if pending:
        if target == "chat":
            st.session_state["prompt_draft"] = pending.get("text", "")
        elif target == "chart":
            st.session_state["chart_symbols_input"] = ", ".join(pending.get("symbols", []))
    remove_pending_issue(target, symbol)


def render_ticker_correction_panel(target, heading="Ticker verification"):
    """Render persistent, explicit correction controls for unverified ticker symbols."""
    pending = st.session_state.get(f"pending_{target}_corrections")
    if not pending:
        return

    with st.container(border=True):
        st.markdown(f"#### 🔎 {heading}")
        st.warning(
            "No symbol has been changed automatically. Review a suggestion below, "
            "or edit the ticker manually. A failed lookup can also be caused by a temporary data-provider issue."
        )
        for item in pending.get("items", []):
            symbol = item.get("symbol", "")
            st.markdown(f"**{symbol}** — {item.get('reason', 'Could not verify ticker.')} ")
            directory_matches = item.get("matches", [])
            if directory_matches:
                with st.expander(f"Global directory matches for {symbol}", expanded=True):
                    st.caption("These are reference/search results, not a guarantee that live prices are available.")
                    for match in directory_matches[:8]:
                        company = match.get("company") or "Company name unavailable"
                        match_symbol = match.get("symbol") or "Ticker not supplied"
                        exchange = match.get("exchange") or "Exchange not supplied"
                        source = match.get("source") or "Reference source"
                        st.write(f"• **{match_symbol}** — {company} · {exchange} · {source}")
            suggestions = item.get("suggestions", [])
            if suggestions:
                st.caption("Possible matches — click one to populate the relevant input; submit again to run the lookup.")
                suggestion_cols = st.columns(min(len(suggestions), 3))
                for index, suggestion in enumerate(suggestions):
                    with suggestion_cols[index % len(suggestion_cols)]:
                        st.button(
                            f"Use {suggestion}",
                            key=f"ticker_fix_{target}_{re.sub(r'[^A-Za-z0-9]', '_', symbol)}_{suggestion}",
                            on_click=apply_ticker_suggestion,
                            args=(target, symbol, suggestion),
                            use_container_width=True,
                        )
            else:
                st.caption("No close automatic suggestion was found. Check the exchange suffix or edit the symbol manually.")
            if item.get("status") != "invalid_format":
                st.button(
                    f"Continue with {symbol} anyway",
                    key=f"ticker_override_{target}_{re.sub(r'[^A-Za-z0-9]', '_', symbol)}",
                    on_click=allow_unverified_ticker,
                    args=(target, symbol),
                    help="Use this only if you believe the ticker is correct; the data provider may still return no data.",
                )
            if target in {"chat", "chart"}:
                st.button(
                    f"Edit {symbol} manually",
                    key=f"ticker_manual_{target}_{re.sub(r'[^A-Za-z0-9]', '_', symbol)}",
                    on_click=dismiss_ticker_issue,
                    args=(target, symbol),
                )


def ticker_issues(symbols):
    """Return detailed issues for the symbols that could not be verified."""
    issues = []
    for symbol in symbols:
        issue = make_ticker_issue(symbol)
        if issue:
            issues.append(issue)
    return issues


def ticker_source_url(symbol):
    return f"https://finance.yahoo.com/quote/{symbol}/"


def markdown_table(df):
    """Create a Markdown table without requiring the optional tabulate package."""
    if df.empty:
        return "No table data available."
    headers = [str(x) for x in [df.index.name or "Metric", *df.columns.tolist()]]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for index_value, row in df.iterrows():
        cells = [str(index_value)] + [str(row[col]) for col in df.columns]
        cells = [cell.replace("|", "\\|").replace("\n", " ") for cell in cells]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def make_report(prompt, answer, sources=None, extra_sections=None):
    sources = sources or []
    extra_sections = extra_sections or []
    parts = [
        "# AI Financial Analyst — Research Report",
        f"**Generated:** {now_local()}",
        "**Data note:** Market data may be delayed. Verify material figures against official company filings.",
        "",
        "## Research question",
        prompt,
        "",
        "## Findings",
        answer,
    ]
    for heading, content in extra_sections:
        parts.extend(["", f"## {heading}", content])
    if sources:
        parts.extend(["", "## Data sources"])
        parts.extend([f"- [{label}]({url})" for label, url in sources])
    parts.extend([
        "",
        "## Important note",
        "This report is for informational and educational purposes only; it is not individualized investment advice.",
    ])
    return "\n".join(parts)


def build_conversation_prompt(current_prompt, prior_messages):
    """Pass recent session messages as context; the user-visible prompt remains unchanged."""
    if not prior_messages:
        return current_prompt
    lines = []
    for message in prior_messages[-10:]:
        role = "User" if message.get("role") == "user" else "Assistant"
        lines.append(f"{role}: {message.get('content', '')}")
    history = "\n".join(lines)
    return (
        "Use the recent conversation below as context for follow-up questions. "
        "Treat it as conversational context, not as verified financial data. "
        "If the current question is ambiguous, resolve it using the conversation where reasonable.\n\n"
        f"Recent conversation:\n{history}\n\nCurrent question: {current_prompt}"
    )


def describe_tool_event(chunk):
    """Best-effort extraction of real Phidata tool events from a streamed response."""
    event = getattr(chunk, "event", None)
    event_text = str(getattr(event, "value", event) or "")
    event_lower = event_text.lower()
    tools = getattr(chunk, "tools", None) or []
    names = []

    for tool in tools:
        name = None
        if isinstance(tool, dict):
            name = tool.get("tool_name") or tool.get("name")
            function = tool.get("function")
            if not name and isinstance(function, dict):
                name = function.get("name")
        else:
            name = getattr(tool, "tool_name", None) or getattr(tool, "name", None)
        if name:
            names.append(str(name))

    if "toolcallstarted" in event_lower or "tool_call_started" in event_lower:
        return "Started tool call: " + (", ".join(names) if names else "tool name not exposed")
    if "toolcallcompleted" in event_lower or "tool_call_completed" in event_lower:
        return "Completed tool call: " + (", ".join(names) if names else "tool name not exposed")
    if "toolcallerror" in event_lower or "tool_call_error" in event_lower:
        return "Tool call reported an error: " + (", ".join(names) if names else "tool name not exposed")
    return None


def answer_from_agent(context_prompt, answer_placeholder, activity_status):
    answer_parts = []
    saw_tool_event = False
    for chunk in agent.run(context_prompt, stream=True):
        tool_note = describe_tool_event(chunk)
        if tool_note:
            saw_tool_event = True
            activity_status.write(f"🔧 {tool_note}")
        content = getattr(chunk, "content", None)
        if isinstance(content, str) and content:
            answer_parts.append(content)
            answer_placeholder.markdown("".join(answer_parts))

    answer = "".join(answer_parts).strip()
    if not answer:
        answer = "The agent returned no text. Please try again."
    answer_placeholder.markdown(answer)
    if not saw_tool_event:
        activity_status.write(
            "The agent run finished. This installed Phidata version did not expose individual tool events in the stream; "
            "the agent is configured with YFinanceTools and show_tool_calls=True."
        )
    return answer


# --------------------------------------------------
# NAVIGATION
# --------------------------------------------------
tab_chat, tab_charts, tab_portfolio, tab_directory = st.tabs([
    "💬 Research Chat",
    "📉 Interactive Charts",
    "💼 Portfolio & Alerts",
    "🌍 Global Ticker Search",
])

# ==================================================
# TAB 1 — CHAT, CONVERSATIONAL CONTEXT, REPORT EXPORT
# ==================================================
with tab_chat:
    st.subheader("Research assistant")
    st.caption("Follow-up questions use recent messages from this browser session as context.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    render_ticker_correction_panel("chat", "Review ticker suggestions for your question")

    if st.session_state.last_report:
        with st.expander("📄 Latest research report", expanded=False):
            st.download_button(
                "Download report (Markdown)",
                data=st.session_state.last_report,
                file_name=st.session_state.last_report_filename,
                mime="text/markdown",
                key="download_latest_report_md",
            )
            if st.session_state.last_report_csv:
                st.download_button(
                    "Download comparison data (CSV)",
                    data=st.session_state.last_report_csv,
                    file_name="fundamental_comparison.csv",
                    mime="text/csv",
                    key="download_latest_report_csv",
                )

    st.markdown("### ✨ Suggested prompts")
    suggestions = [
        ("📊 Compare stocks", "Compare AAPL vs MSFT fundamentals"),
        ("📰 Market news", "Summarize the latest news about NVDA"),
        ("📈 Stock analysis", "Analyze TSLA's financial performance"),
        ("⚠️ Investment risks", "Explain the key risks of investing in AMZN"),
    ]
    suggestion_cols = st.columns(4)
    for index, (label, prompt_text) in enumerate(suggestions):
        with suggestion_cols[index]:
            st.button(
                label,
                key=f"suggestion_{index}",
                help=prompt_text,
                use_container_width=True,
                on_click=select_suggestion,
                args=(prompt_text,),
            )

    with st.form("chat_form", clear_on_submit=False):
        st.text_area(
            "Your prompt",
            key="prompt_draft",
            placeholder="Ask about a stock, compare companies, or analyze financial news...",
            height=100,
        )
        st.form_submit_button(
            "➤ Send",
            type="primary",
            use_container_width=True,
            on_click=submit_prompt,
        )

    prompt = st.session_state.get("submitted_prompt")
    if prompt:
        # Clear the submission before execution to reduce accidental duplicate runs.
        st.session_state.submitted_prompt = None
        st.session_state.pending_chat_corrections = None
        previous_messages = list(st.session_state.messages)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        activity_lines = []
        answer = ""
        data_sources = []
        extra_report_sections = []
        comparison_csv = ""

        with st.chat_message("assistant"):
            answer_placeholder = st.empty()
            try:
                if is_stock_comparison(prompt):
                    tickers = extract_tickers(prompt)
                    with st.status("Running comparison workflow", expanded=True) as status:
                        status.write("Step 1/4 — Parsed ticker symbols from the prompt.")
                        if len(tickers) < 2:
                            status.update(label="Comparison needs more ticker symbols", state="error")
                            st.warning("Please specify at least two stock ticker symbols, for example: Compare AAPL vs MSFT.")
                            answer = "Please enter at least two ticker symbols, such as AAPL and MSFT."
                        else:
                            log_activity(activity_lines, f"Ticker extraction completed: {', '.join(tickers)}")
                            status.write(f"Step 2/4 — Verifying ticker symbols with Yahoo Finance: {', '.join(tickers)}.")
                            issues = ticker_issues(tickers)
                            if issues:
                                st.session_state.pending_chat_corrections = {
                                    "text": prompt,
                                    "items": issues,
                                }
                                issue_text = ", ".join(item["symbol"] for item in issues)
                                status.update(label="Ticker verification needs your review", state="error")
                                st.warning(
                                    f"Yahoo Finance could not verify: {issue_text}. "
                                    "No comparison was run for this request. Review suggested corrections below the chat and submit again."
                                )
                                answer = (
                                    f"I couldn't reliably verify these ticker symbols: {issue_text}. "
                                    "Please review the suggested corrections and submit the question again. "
                                    "A failed lookup can be caused by a typo, missing listing data, or a temporary provider issue."
                                )
                            else:
                                status.write(f"Step 3/4 — Retrieving fundamentals for {', '.join(tickers)}.")
                                comparison, errors = get_fundamentals(tuple(tickers))
                                for symbol in tickers:
                                    data_sources.append((f"Yahoo Finance: {symbol}", ticker_source_url(symbol)))
                                log_activity(activity_lines, "Fundamentals retrieval function completed")

                                if comparison.empty:
                                    status.update(label="No fundamentals returned", state="error")
                                    answer = f"I couldn't retrieve fundamental data for {', '.join(tickers)}. The ticker check passed, but fundamental data was unavailable. Please retry or check the source links."
                                    if errors:
                                        st.error("; ".join(errors))
                                else:
                                    status.write("Step 4/4 — Formatting units and checking for missing fields.")
                                    display_df = comparison.copy()
                                    for metric in display_df.index:
                                        for symbol in display_df.columns:
                                            display_df.loc[metric, symbol] = format_value(metric, display_df.loc[metric, symbol])
                                    missing_count = int(display_df.isin(["N/A"]).sum().sum())
                                    check_message = (
                                        f"Data check: {missing_count} metric cells are unavailable (N/A); "
                                        "values are sourced from Yahoo Finance and may be delayed."
                                    )
                                    st.subheader(f"Fundamental comparison: {' vs '.join(tickers)}")
                                    st.caption(f"Retrieved: {now_local()} · Source: Yahoo Finance via yfinance · {check_message}")
                                    st.dataframe(display_df, use_container_width=True)
                                    if errors:
                                        with st.expander("Data retrieval warnings"):
                                            for error in errors:
                                                st.write(error)
                                    answer = (
                                        f"Retrieved available fundamental metrics for {', '.join(tickers)}. "
                                        f"The table has {missing_count} unavailable metric cells. "
                                        "Compare accounting periods and currencies before interpreting differences; verify material figures against company filings."
                                    )
                                    extra_report_sections.append(("Fundamental comparison table", markdown_table(display_df)))
                                    extra_report_sections.append(("Data verification", check_message + " Missing fields are shown as N/A, not zero."))
                                    comparison_csv = comparison.to_csv()
                                    log_activity(activity_lines, "Formatted comparison and checked missing values")
                                    status.update(label="Comparison workflow completed", state="complete")
                    answer_placeholder.markdown(answer)
                else:
                    # Explicit dollar-prefixed symbols are unambiguous enough to validate before tool use.
                    explicit_symbols = list(dict.fromkeys(
                        normalize_ticker(item)
                        for item in re.findall(r"\$([A-Za-z0-9.^=-]{1,15})", prompt)
                    ))
                    explicit_issues = ticker_issues(explicit_symbols) if explicit_symbols else []
                    if explicit_issues:
                        st.session_state.pending_chat_corrections = {
                            "text": prompt,
                            "items": explicit_issues,
                        }
                        issue_text = ", ".join(item["symbol"] for item in explicit_issues)
                        st.warning(
                            f"I couldn't verify these explicitly specified ticker symbols: {issue_text}. "
                            "Review suggestions below, or remove the $ prefix if it was not intended as a ticker."
                        )
                        answer = (
                            f"Ticker verification is needed for {issue_text}. "
                            "Please review the suggested symbols and submit the question again."
                        )
                        answer_placeholder.markdown(answer)
                    else:
                        context_prompt = build_conversation_prompt(prompt, previous_messages)
                        with st.status("AI research workflow", expanded=True) as status:
                            status.write("Step 1 — Preparing the request with recent conversation context.")
                            status.write("Step 2 — Running the finance agent; tool calls are reported by the framework when its stream exposes them.")
                            log_activity(activity_lines, "Submitted context-aware prompt to Finance Agent")
                            answer = answer_from_agent(context_prompt, answer_placeholder, status)
                            status.write("Step 3 — Preparing source and timestamp notes for the downloadable report.")
                            status.update(label="AI research workflow completed", state="complete")
                        # Add quote-page references only for symbols explicitly written with a $ prefix.
                        for symbol in explicit_symbols:
                            data_sources.append((f"Yahoo Finance: {symbol}", ticker_source_url(symbol)))

                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.session_state.activity_log = activity_lines
                st.session_state.last_report = make_report(
                    prompt,
                    answer,
                    sources=data_sources,
                    extra_sections=extra_report_sections,
                )
                st.session_state.last_report_csv = comparison_csv
                st.session_state.last_report_filename = "financial_research_report.md"

            except Exception as exc:
                answer = f"Something went wrong: {exc}"
                st.error(answer)
                st.info("Check your internet connection, API configuration, ticker symbols, and terminal output.")
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.session_state.last_report = make_report(prompt, answer, sources=data_sources)
                st.session_state.last_report_csv = ""

        if st.session_state.activity_log:
            with st.expander("🧭 Latest workflow activity", expanded=False):
                for line in st.session_state.activity_log:
                    st.write(line)

        st.download_button(
            "⬇️ Download this research report",
            data=st.session_state.last_report,
            file_name=st.session_state.last_report_filename,
            mime="text/markdown",
            key="download_report_after_query",
        )
        if st.session_state.last_report_csv:
            st.download_button(
                "⬇️ Download fundamentals CSV",
                data=st.session_state.last_report_csv,
                file_name="fundamental_comparison.csv",
                mime="text/csv",
                key="download_csv_after_query",
            )

        if st.session_state.pending_chat_corrections:
            st.rerun()

# ==================================================
# TAB 2 — INTERACTIVE HISTORICAL CHARTS
# ==================================================
with tab_charts:
    st.subheader("Interactive stock performance")
    st.write("Enter ticker symbols separated by commas. Price charts show historical adjusted closes; the normalized chart starts every stock at 100 to compare percentage performance.")

    with st.form("chart_form"):
        chart_symbols_text = st.text_input("Ticker symbols", value="AAPL, MSFT", key="chart_symbols_input")
        chart_period = st.selectbox(
            "Historical period",
            options=["1mo", "3mo", "6mo", "1y", "2y", "5y"],
            index=3,
            key="chart_period_input",
        )
        charts_submitted = st.form_submit_button("Load charts", type="primary")

    if charts_submitted:
        st.session_state.pending_chart_corrections = None
        symbols = []
        for token in chart_symbols_text.split(","):
            symbol = normalize_ticker(token)
            # Keep malformed non-empty tokens so the shared validator can explain the problem
            # rather than silently dropping them from a multi-ticker request.
            if symbol and symbol not in symbols:
                symbols.append(symbol)
        symbols = symbols[:6]

        if not symbols:
            st.error("Enter at least one ticker symbol, such as AAPL or MSFT.")
        else:
            issues = ticker_issues(symbols)
            issue_symbols = {item["symbol"] for item in issues}
            if issues:
                st.session_state.pending_chart_corrections = {
                    "text": chart_symbols_text,
                    "symbols": symbols,
                    "items": issues,
                }
                st.warning(
                    "The app will chart only symbols that Yahoo Finance verifies. "
                    "Review the correction panel below for other symbols."
                )
            verified_symbols = [symbol for symbol in symbols if symbol not in issue_symbols]
            price_series = {}
            chart_errors = []
            if verified_symbols:
                with st.status("Retrieving historical prices", expanded=True) as status:
                    for symbol in verified_symbols:
                        status.write(f"Fetching adjusted historical prices for {symbol}…")
                        try:
                            history = fetch_history(symbol, chart_period)
                            if history.empty:
                                chart_errors.append(f"{symbol}: ticker verification passed, but no historical data was returned for this period")
                                continue
                            series = history["Close"].dropna()
                            if series.empty:
                                chart_errors.append(f"{symbol}: no valid closing prices returned")
                                continue
                            price_series[symbol] = series
                            status.write(f"{symbol}: received {len(series)} observations; latest date {series.index[-1]}.")
                        except Exception as exc:
                            chart_errors.append(f"{symbol}: historical data lookup failed: {exc}")
                    status.update(label="Historical data retrieval finished", state="complete" if price_series else "error")
            else:
                st.info("No chart was generated because none of the entered tickers could be verified.")

            if chart_errors:
                with st.expander("Retrieval warnings"):
                    for message in chart_errors:
                        st.write(message)

            if price_series:
                st.caption(f"Source: Yahoo Finance via yfinance · Retrieved: {now_local()} · Adjusted historical close; may not be real-time.")
                price_fig = go.Figure()
                normalized_fig = go.Figure()
                summary_rows = []
                chart_sources = []

                for symbol, series in price_series.items():
                    price_fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines", name=symbol))
                    normalized = (series / series.iloc[0]) * 100.0
                    normalized_fig.add_trace(go.Scatter(x=normalized.index, y=normalized.values, mode="lines", name=symbol))
                    total_return = ((series.iloc[-1] / series.iloc[0]) - 1) * 100
                    summary_rows.append({
                        "Ticker": symbol,
                        "First close": float(series.iloc[0]),
                        "Latest close": float(series.iloc[-1]),
                        "Price change (%)": float(total_return),
                        "First date": str(series.index[0]),
                        "Latest date": str(series.index[-1]),
                        "Observations": int(series.count()),
                    })
                    chart_sources.append(f"{symbol}: {ticker_source_url(symbol)}")

                price_fig.update_layout(title="Adjusted closing price", xaxis_title="Date", yaxis_title="Price (listing currency)", hovermode="x unified", legend_title="Ticker")
                normalized_fig.update_layout(title="Relative performance (start = 100)", xaxis_title="Date", yaxis_title="Indexed value", hovermode="x unified", legend_title="Ticker")
                st.plotly_chart(price_fig, use_container_width=True)
                st.plotly_chart(normalized_fig, use_container_width=True)

                summary_df = pd.DataFrame(summary_rows)
                st.markdown("#### Performance and data checks")
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
                st.caption("Calculation check: price change = (latest adjusted close / first adjusted close − 1) × 100. The series can cover slightly different trading dates for different exchanges.")
                st.markdown("**Sources**")
                for source in chart_sources:
                    st.markdown(f"- {source}")
                st.download_button(
                    "Download chart data (CSV)",
                    data=summary_df.to_csv(index=False),
                    file_name="stock_performance_summary.csv",
                    mime="text/csv",
                    key="download_chart_summary",
                )

    render_ticker_correction_panel("chart", "Review chart ticker suggestions")

# ==================================================
# TAB 3 — PORTFOLIO AND PRICE THRESHOLD ALERTS
# ==================================================
with tab_portfolio:
    st.subheader("Portfolio tracker")
    st.caption("Holdings and alert rules are stored in this Streamlit browser session. This version does not send email, SMS, or background notifications.")

    with st.form("add_holding_form"):
        st.markdown("#### Add or update a holding")
        hold_col1, hold_col2, hold_col3 = st.columns(3)
        with hold_col1:
            holding_symbol = st.text_input("Ticker", placeholder="AAPL", key="holding_symbol_input")
        with hold_col2:
            holding_shares = st.number_input("Shares", min_value=0.0, value=1.0, step=1.0, format="%.4f", key="holding_shares_input")
        with hold_col3:
            holding_cost = st.number_input("Average purchase price per share", min_value=0.0, value=0.0, step=1.0, format="%.4f", key="holding_cost_input")
        holding_submitted = st.form_submit_button("Save holding", type="primary")

    if holding_submitted:
        st.session_state.pending_holding_corrections = None
        symbol = normalize_ticker(holding_symbol)
        if not TICKER_PATTERN.fullmatch(symbol):
            st.error("Enter a valid ticker symbol, such as AAPL or RELIANCE.NS.")
        elif holding_shares <= 0:
            st.error("Shares must be greater than zero.")
        elif holding_cost < 0:
            st.error("Average purchase price cannot be negative.")
        else:
            check = validate_ticker_symbol(symbol)
            if check["status"] not in {"verified", "accepted_unverified"}:
                st.session_state.pending_holding_corrections = {
                    "text": symbol,
                    "items": [make_ticker_issue(symbol)],
                }
                st.warning(f"No holding was saved because {symbol} could not be verified. Review the suggestions below, then submit again.")
            else:
                st.session_state.holdings[symbol] = {
                    "shares": float(holding_shares),
                    "avg_cost": float(holding_cost),
                }
                if check["status"] == "accepted_unverified":
                    st.warning(f"Saved {symbol} using your manual override. Price data may remain unavailable.")
                else:
                    st.success(f"Saved {symbol}. Adding the same ticker again replaces its saved quantity and average cost.")

    render_ticker_correction_panel("holding", "Review holding ticker suggestions")

    holding_symbols = list(st.session_state.holdings.keys())
    current_prices = {}
    price_errors = []

    if holding_symbols:
        st.markdown("#### Current portfolio")
        refresh_col, note_col = st.columns([1, 3])
        with refresh_col:
            refresh_prices = st.button("🔄 Refresh prices", key="refresh_portfolio_prices")
        if refresh_prices:
            get_latest_quote.clear()
            fetch_history.clear()

        with st.status("Checking the latest available prices", expanded=False) as status:
            for symbol in holding_symbols:
                try:
                    price, timestamp, currency = get_latest_quote(symbol)
                    current_prices[symbol] = (price, timestamp, currency)
                    status.write(f"{symbol}: latest available price fetched from Yahoo Finance.")
                except Exception as exc:
                    price_errors.append(f"{symbol}: {exc}")
            status.update(label="Portfolio price check finished", state="complete" if current_prices else "error")

        portfolio_rows = []
        for symbol, holding in st.session_state.holdings.items():
            shares = holding["shares"]
            avg_cost = holding["avg_cost"]
            cost_basis = shares * avg_cost
            if symbol in current_prices:
                price, timestamp, currency = current_prices[symbol]
                market_value = shares * price
                pnl = market_value - cost_basis
                return_pct = (pnl / cost_basis * 100) if cost_basis else None
                portfolio_rows.append({
                    "Ticker": symbol,
                    "Shares": shares,
                    "Avg cost": avg_cost,
                    "Latest price": price,
                    "Currency": currency,
                    "Market value": market_value,
                    "Cost basis": cost_basis,
                    "Unrealized P/L": pnl,
                    "Return (%)": return_pct,
                    "Price timestamp": timestamp,
                })
            else:
                portfolio_rows.append({
                    "Ticker": symbol,
                    "Shares": shares,
                    "Avg cost": avg_cost,
                    "Latest price": None,
                    "Currency": "N/A",
                    "Market value": None,
                    "Cost basis": cost_basis,
                    "Unrealized P/L": None,
                    "Return (%)": None,
                    "Price timestamp": "Unavailable",
                })

        portfolio_df = pd.DataFrame(portfolio_rows)
        st.dataframe(portfolio_df, use_container_width=True, hide_index=True)
        if price_errors:
            for error in price_errors:
                st.warning(error)

        available_currencies = {row["Currency"] for row in portfolio_rows if row["Latest price"] is not None}
        if len(available_currencies) == 1:
            known_rows = [row for row in portfolio_rows if row["Market value"] is not None]
            total_value = sum(row["Market value"] for row in known_rows)
            total_cost = sum(row["Cost basis"] for row in known_rows)
            total_pnl = total_value - total_cost
            total_return = (total_pnl / total_cost * 100) if total_cost else None
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Market value", f"{total_value:,.2f} {next(iter(available_currencies))}")
            m2.metric("Cost basis", f"{total_cost:,.2f}")
            m3.metric("Unrealized P/L", f"{total_pnl:,.2f}")
            m4.metric("Return", f"{total_return:.2f}%" if total_return is not None else "N/A")
        elif len(available_currencies) > 1:
            st.warning("Holdings include multiple currencies. Totals are hidden because this app does not convert currencies.")

        st.caption(f"Price source: Yahoo Finance via yfinance · Checked: {now_local()} · Prices use the latest available row and may not be live market quotes.")

        remove_symbol = st.selectbox("Remove a holding", options=["(select)"] + holding_symbols, key="remove_holding_select")
        if st.button("Remove selected holding", key="remove_holding_button"):
            if remove_symbol != "(select)":
                st.session_state.holdings.pop(remove_symbol, None)
                st.rerun()
    else:
        st.info("No holdings yet. Add a ticker above to start tracking a portfolio.")

    st.divider()
    st.subheader("Price threshold alerts")
    st.caption("Alerts are checked when this tab reruns or you refresh prices. They are not background notifications.")

    with st.form("add_alert_form"):
        alert_col1, alert_col2, alert_col3 = st.columns(3)
        with alert_col1:
            alert_symbol_input = st.text_input("Alert ticker", placeholder="NVDA", key="alert_symbol_input")
        with alert_col2:
            alert_condition = st.selectbox("Trigger when price is", ["at or above", "at or below"], key="alert_condition_input")
        with alert_col3:
            alert_threshold = st.number_input("Threshold price", min_value=0.01, value=100.0, step=1.0, key="alert_threshold_input")
        alert_submitted = st.form_submit_button("Add alert")

    if alert_submitted:
        st.session_state.pending_alert_corrections = None
        symbol = normalize_ticker(alert_symbol_input)
        if not TICKER_PATTERN.fullmatch(symbol):
            st.error("Enter a valid ticker symbol for the alert.")
        elif alert_threshold <= 0:
            st.error("The threshold must be greater than zero.")
        else:
            check = validate_ticker_symbol(symbol)
            if check["status"] not in {"verified", "accepted_unverified"}:
                st.session_state.pending_alert_corrections = {
                    "text": symbol,
                    "items": [make_ticker_issue(symbol)],
                }
                st.warning(f"No alert was saved because {symbol} could not be verified. Review the suggestions below, then submit again.")
            else:
                st.session_state.alerts.append({
                    "symbol": symbol,
                    "condition": alert_condition,
                    "threshold": float(alert_threshold),
                })
                if check["status"] == "accepted_unverified":
                    st.warning(f"Added alert for {symbol} using your manual override. Price data may remain unavailable.")
                else:
                    st.success(f"Added alert for {symbol}.")

    render_ticker_correction_panel("alert", "Review alert ticker suggestions")

    if st.session_state.alerts:
        alert_rows = []
        for index, alert in enumerate(st.session_state.alerts):
            symbol = alert["symbol"]
            quote = current_prices.get(symbol)
            if quote is None:
                try:
                    quote = get_latest_quote(symbol)
                    current_prices[symbol] = quote
                except Exception:
                    quote = None
            price = quote[0] if quote else None
            threshold = alert["threshold"]
            if price is None:
                state = "Price unavailable"
            elif alert["condition"] == "at or above" and price >= threshold:
                state = "🔔 Triggered"
            elif alert["condition"] == "at or below" and price <= threshold:
                state = "🔔 Triggered"
            else:
                state = "Waiting"
            alert_rows.append({
                "#": index + 1,
                "Ticker": symbol,
                "Condition": alert["condition"],
                "Threshold": threshold,
                "Latest price": price,
                "Status": state,
            })
        st.dataframe(pd.DataFrame(alert_rows), use_container_width=True, hide_index=True)
        triggered = [row for row in alert_rows if row["Status"] == "🔔 Triggered"]
        if triggered:
            st.warning("One or more price thresholds are currently met. This notice appears only when the app checks the prices.")

        remove_alert_index = st.selectbox(
            "Remove an alert",
            options=["(select)"] + [f"{i + 1}. {a['symbol']} {a['condition']} {a['threshold']:g}" for i, a in enumerate(st.session_state.alerts)],
            key="remove_alert_select",
        )
        if st.button("Remove selected alert", key="remove_alert_button"):
            if remove_alert_index != "(select)":
                selected_number = int(remove_alert_index.split(".", 1)[0]) - 1
                if 0 <= selected_number < len(st.session_state.alerts):
                    st.session_state.alerts.pop(selected_number)
                    st.rerun()
    else:
        st.info("No alerts configured. Add a threshold above to create one.")

    st.caption("Portfolio holdings and alerts currently live in Streamlit session state. Restarting the session/browser connection may clear them; use a database if you later need durable accounts and alerts.")

# ==================================================
# TAB 4 — GLOBAL TICKER DIRECTORY SEARCH
# ==================================================
with tab_directory:
    st.subheader("Global ticker directory search")
    st.caption(
        "Search across Yahoo Finance's supported international market symbols and OpenFIGI's "
        "global security reference directory. This combines two sources; it is not a promise of "
        "complete, real-time coverage of every exchange worldwide."
    )

    with st.form("global_directory_search_form"):
        lookup_query = st.text_input(
            "Ticker or company name",
            key="global_lookup_query",
            placeholder="Try TCS, Tata Consultancy Services, 7203.T, or Samsung Electronics",
        )
        run_directory_search = st.form_submit_button("Search global directories", type="primary")

    if run_directory_search:
        query = str(lookup_query or "").strip()
        if len(query) < 1:
            st.warning("Enter a ticker symbol or company name to search.")
            st.session_state.global_lookup_results = []
        else:
            with st.spinner("Searching global security directories…"):
                st.session_state.global_lookup_results = search_global_ticker_directory(query, limit=30)
            if not st.session_state.global_lookup_results:
                st.warning(
                    "No results were returned by the available directories. This can happen because "
                    "of coverage limits, rate limiting, or a temporary connection problem; it does not "
                    "prove that the security is not listed."
                )

    lookup_results = st.session_state.get("global_lookup_results", [])
    if lookup_results:
        lookup_df = pd.DataFrame([
            {
                "Symbol": row.get("symbol", ""),
                "Company / security": row.get("company", ""),
                "Exchange / code": row.get("exchange", ""),
                "Security type": row.get("security_type", ""),
                "Directory source": row.get("source", ""),
                "Yahoo Finance quote page": (
                    ticker_source_url(row.get("symbol", ""))
                    if row.get("source") == "Yahoo Finance search" and row.get("symbol")
                    else ""
                ),
            }
            for row in lookup_results
        ])
        st.dataframe(lookup_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download global search results (CSV)",
            data=lookup_df.to_csv(index=False),
            file_name="global_ticker_search_results.csv",
            mime="text/csv",
            key="download_global_ticker_results",
        )
        st.info(
            "For charting or portfolio tracking, choose a Yahoo Finance symbol where available, "
            "usually including its exchange suffix (for example, TCS.NS for NSE India). "
            "An OpenFIGI match confirms a reference-directory result, not Yahoo quote availability."
        )

# --------------------------------------------------
# SIDEBAR — SESSION CONTROLS AND DATA NOTES
# --------------------------------------------------
with st.sidebar:
    st.header("Agent controls")
    st.write(f"Conversation messages: {len(st.session_state.messages)}")
    st.write(f"Tracked holdings: {len(st.session_state.holdings)}")
    st.write(f"Price alerts: {len(st.session_state.alerts)}")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_report = ""
        st.session_state.last_report_csv = ""
        st.session_state.activity_log = []
        st.session_state.pending_chat_corrections = None
        st.rerun()
    st.divider()
    st.caption("Data & limitations")
    st.caption("• Market information comes from Yahoo Finance through yfinance.")
    st.caption("• Ticker discovery combines Yahoo Finance search and OpenFIGI reference results where available.")
    st.caption("• No single public database guarantees every active listing on every exchange worldwide.")
    st.caption("• Agent answers depend on Groq and the enabled finance tools.")
    st.caption("• Memory, holdings, and alerts are session-local in this version.")
    st.caption("• This tool is for research and education, not individualized financial advice.")
