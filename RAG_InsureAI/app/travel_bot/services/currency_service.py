"""
Display-only currency conversion for quote prices.

Protego (the actual quoting/underwriting/payment provider) always prices
and issues policies in AED — that never changes, and this module doesn't
touch anything that reaches Protego. It only converts the AED amount
chat.py already has for display, so a user who said they're travelling
from India sees "≈ 898.45 INR" alongside the real "34.57 AED" instead of
only ever seeing AED regardless of where they're from.

Two independent lookups:
1. Country -> ISO 4217 currency code (COUNTRY_TO_CURRENCY). Stable
   reference data (currency codes essentially never change), so a static
   table is the right call here — unlike exchange RATES, this doesn't go
   stale.
2. AED -> target-currency exchange rate, fetched live from a free, no-API-
   key FX-rate service (open.er-api.com) and cached for _RATE_CACHE_TTL
   seconds, since a rate is only ever a snapshot and re-fetching per quote
   line would be both slow and unfriendly to a third-party free API. Any
   failure (network, non-200, malformed body) degrades to AED-only display
   — a stale or missing conversion is a cosmetic loss, not a reason to
   break the actual quote.
"""
from __future__ import annotations

import time
from typing import Optional

import requests

_RATE_API_URL = "https://open.er-api.com/v6/latest/{base}"
_RATE_CACHE_TTL = 3600  # 1 hour — exchange rates don't move fast enough to need fresher than this
_rate_cache: dict = {"base": None, "rates": {}, "fetched_at": 0.0}


def get_currency_for_country(country: str) -> Optional[str]:
    """Returns the ISO 4217 currency code for a (already normalize_country()'d,
    full-name) country, or None if unrecognized — callers should fall back
    to AED-only display in that case, never guess."""
    if not country:
        return None
    return COUNTRY_TO_CURRENCY.get(country.strip())


def _fetch_rates(base: str = "AED") -> dict:
    now = time.time()
    if _rate_cache["base"] == base and (now - _rate_cache["fetched_at"]) < _RATE_CACHE_TTL:
        return _rate_cache["rates"]
    try:
        resp = requests.get(_RATE_API_URL.format(base=base), timeout=8)
        resp.raise_for_status()
        data = resp.json()
        rates = data.get("rates") or {}
        if data.get("result") == "success" and rates:
            _rate_cache["base"] = base
            _rate_cache["rates"] = rates
            _rate_cache["fetched_at"] = now
            return rates
    except Exception as exc:
        print(f"[currency_service] rate fetch failed for base={base}: {exc}")
    # Keep serving a stale cache over a hard failure if we have one at all —
    # a rate that's a few hours stale is still far more useful than none.
    return _rate_cache["rates"]


def convert_from_aed(amount_aed: float, to_currency: str) -> Optional[float]:
    if not to_currency or to_currency == "AED":
        return None
    rates = _fetch_rates("AED")
    rate = rates.get(to_currency)
    if rate is None or not isinstance(amount_aed, (int, float)):
        return None
    return amount_aed * rate


def format_dual_currency(amount_aed: float, departure_country: str) -> str:
    """"34.57 AED" normally, "34.57 AED (≈ 898.45 INR)" when the departure
    country maps to a recognized non-AED currency and a live rate was
    available. Never raises — any lookup/conversion failure just yields the
    plain AED string."""
    base = f"{amount_aed:,.2f} AED" if isinstance(amount_aed, (int, float)) else f"{amount_aed} AED"
    to_currency = get_currency_for_country(departure_country)
    if not to_currency:
        return base
    converted = convert_from_aed(amount_aed, to_currency)
    if converted is None:
        return base
    return f"{base} (≈ {converted:,.2f} {to_currency})"


# Full-country-name -> ISO 4217 currency code. Keys match the canonical
# names normalize_country() / COUNTRY_ALIASES produce (schemas/travel.py) —
# an unaliased country the user typed passes through unchanged, so this
# covers the common English full names a user or the LLM extractor would
# actually produce, not ISO country codes.
COUNTRY_TO_CURRENCY = {
    "United Arab Emirates": "AED", "Saudi Arabia": "SAR", "Qatar": "QAR",
    "Kuwait": "KWD", "Bahrain": "BHD", "Oman": "OMR",
    "India": "INR", "Pakistan": "PKR", "Bangladesh": "BDT", "Sri Lanka": "LKR",
    "Nepal": "NPR", "Philippines": "PHP", "Indonesia": "IDR", "Malaysia": "MYR",
    "Singapore": "SGD", "Thailand": "THB", "Vietnam": "VND", "China": "CNY",
    "Hong Kong": "HKD", "Japan": "JPY", "South Korea": "KRW", "Taiwan": "TWD",
    "United Kingdom": "GBP", "Ireland": "EUR", "France": "EUR", "Germany": "EUR",
    "Italy": "EUR", "Spain": "EUR", "Portugal": "EUR", "Netherlands": "EUR",
    "Belgium": "EUR", "Austria": "EUR", "Greece": "EUR", "Finland": "EUR",
    "Luxembourg": "EUR", "Slovakia": "EUR", "Slovenia": "EUR", "Estonia": "EUR",
    "Latvia": "EUR", "Lithuania": "EUR", "Cyprus": "EUR", "Malta": "EUR",
    "Croatia": "EUR", "Switzerland": "CHF", "Norway": "NOK", "Sweden": "SEK",
    "Denmark": "DKK", "Iceland": "ISK", "Poland": "PLN", "Czech Republic": "CZK",
    "Hungary": "HUF", "Romania": "RON", "Bulgaria": "BGN", "Turkey": "TRY",
    "Russia": "RUB", "Ukraine": "UAH",
    "United States": "USD", "Canada": "CAD", "Mexico": "MXN", "Brazil": "BRL",
    "Argentina": "ARS", "Chile": "CLP", "Colombia": "COP", "Peru": "PEN",
    "Egypt": "EGP", "Jordan": "JOD", "Lebanon": "LBP", "Iraq": "IQD",
    "Israel": "ILS", "Morocco": "MAD", "Tunisia": "TND", "Algeria": "DZD",
    "South Africa": "ZAR", "Nigeria": "NGN", "Kenya": "KES", "Ghana": "GHS",
    "Ethiopia": "ETB", "Tanzania": "TZS", "Uganda": "UGX",
    "Australia": "AUD", "New Zealand": "NZD",
    "Afghanistan": "AFN", "Iran": "IRR", "Yemen": "YER",
    "Azerbaijan": "AZN", "Georgia": "GEL", "Armenia": "AMD",
    "Kazakhstan": "KZT", "Uzbekistan": "UZS",
    "Maldives": "MVR", "Bhutan": "BTN", "Myanmar": "MMK", "Cambodia": "KHR",
    "Laos": "LAK", "Mongolia": "MNT",
    # Added alongside the phone-country picker's expansion to full world
    # coverage — same underlying "all world currencies" requirement.
    "Palestine": "ILS", "Syria": "SYP", "North Korea": "KPW", "Macau": "MOP",
    "Timor-Leste": "USD", "Brunei": "BND",
    "Kyrgyzstan": "KGS", "Tajikistan": "TJS", "Turkmenistan": "TMT",
    "Belarus": "BYN", "Moldova": "MDL", "Bosnia and Herzegovina": "BAM",
    "Serbia": "RSD", "Montenegro": "EUR", "North Macedonia": "MKD",
    "Kosovo": "EUR", "Albania": "ALL", "San Marino": "EUR",
    "Vatican City": "EUR", "Liechtenstein": "CHF", "Andorra": "EUR",
    "Monaco": "EUR",
    "Venezuela": "VES", "Ecuador": "USD", "Bolivia": "BOB", "Paraguay": "PYG",
    "Uruguay": "UYU", "Guyana": "GYD", "Suriname": "SRD",
    "Guatemala": "GTQ", "Belize": "BZD", "Honduras": "HNL",
    "El Salvador": "USD", "Nicaragua": "NIO", "Costa Rica": "CRC",
    "Panama": "PAB", "Cuba": "CUP", "Jamaica": "JMD", "Haiti": "HTG",
    "Dominican Republic": "DOP", "Bahamas": "BSD", "Trinidad and Tobago": "TTD",
    "Barbados": "BBD", "Antigua and Barbuda": "XCD", "Dominica": "XCD",
    "Grenada": "XCD", "Saint Kitts and Nevis": "XCD", "Saint Lucia": "XCD",
    "Saint Vincent and the Grenadines": "XCD",
    "Libya": "LYD", "Sudan": "SDG", "South Sudan": "SSP", "Rwanda": "RWF",
    "Burundi": "BIF", "Angola": "AOA", "Zambia": "ZMW", "Zimbabwe": "ZWL",
    "Malawi": "MWK", "Mozambique": "MZN", "Namibia": "NAD", "Botswana": "BWP",
    "Eswatini": "SZL", "Lesotho": "LSL", "Madagascar": "MGA",
    "Mauritius": "MUR", "Seychelles": "SCR", "Comoros": "KMF",
    "Djibouti": "DJF", "Eritrea": "ERN", "Somalia": "SOS",
    "Cameroon": "XAF", "Central African Republic": "XAF", "Chad": "XAF",
    "Congo": "XAF", "DR Congo": "CDF", "Gabon": "XAF",
    "Equatorial Guinea": "XAF", "Sao Tome and Principe": "STN",
    "Senegal": "XOF", "Gambia": "GMD", "Guinea": "GNF",
    "Guinea-Bissau": "XOF", "Mali": "XOF", "Mauritania": "MRU",
    "Niger": "XOF", "Burkina Faso": "XOF", "Ivory Coast": "XOF",
    "Sierra Leone": "SLE", "Liberia": "LRD", "Togo": "XOF", "Benin": "XOF",
    "Cape Verde": "CVE",
    "Fiji": "FJD", "Papua New Guinea": "PGK", "Samoa": "WST", "Tonga": "TOP",
    "Vanuatu": "VUV", "Solomon Islands": "SBD", "Kiribati": "AUD",
    "Micronesia": "USD", "Marshall Islands": "USD", "Palau": "USD",
    "Nauru": "AUD", "Tuvalu": "AUD",
}
