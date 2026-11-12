#!/usr/bin/env python3
"""
Wheel-Screener: sucht täglich Cash-Secured Puts, die deine Wheel-Kriterien erfüllen.

Automatisch geprüft:
  Marktumfeld (VIX, S&P 500 über 200-Tage-Linie) -> maximales Delta
  Einstiegssignal im Wochenchart (wie in TradingView), erste Woche "grün":
    TEMA 8 über TEMA 21, Stoch RSI (3,3,14,14) K über D und steigend,
    MACD (12,26,9) Histogramm positiv. In der Vorwoche war es noch nicht grün.
    In der ersten grünen Woche "neu, vorläufig" (Wochenkerze noch offen),
    in der zweiten Woche "bestätigt" (Vorwoche grün abgeschlossen).
    Kurs unter TEMA 200 ist kein Ausschluss, wird aber als Hinweis angezeigt.
  Diagnose: Ticker im Feld "diagnose" beim manuellen Start (oder Umgebungs-
    variable DIAGNOSE=CMG,AMZN) zeigt jedes Kriterium mit Ergebnis im Protokoll.
  KGV <= 50, Strike >= 15, Prämie >= 0,10, Rendite 20-40 % p. a.
  Analysten positiv, Kursziel deutlich über Kurs
  Earnings mindestens 30 Tage entfernt und erst nach dem Verfall
  Put mit Laufzeit 7-21 Tage (ohne Wochenoptionen: nächste Monatsoption bis 28 Tage), Delta unter Grenze
  (Unterstützung und erwartete Bewegung werden angezeigt, sind aber keine Pflicht)
  Liquidität (Open Interest, Spread), Mindestrendite
  Streuung: höchstens 2 Treffer pro Branche

Optionskurse: Alpaca (wenn ALPACA_API_KEY und ALPACA_SECRET_KEY gesetzt sind, kostenloser
  "indicative"-Feed), fehlende Werte aus Eulerpool und Yahoo. Ist der Spread in den Daten
  unrealistisch breit, wird die Prämie aus der impliziten Volatilität geschätzt und markiert.

Über die Eulerpool-API (wenn EULERPOOL_API_KEY gesetzt ist):
  AAQS >= 6 und Kurs unter dem Eulerpool Fair Value
Daten: Yahoo Finance + Eulerpool. Keine Anlageberatung.
"""
import io
import math
import os
import sys
import datetime as dt

import numpy as np
import pandas as pd
import requests
import yfinance as yf

try:
    from eulerpool import Eulerpool
except ImportError:
    Eulerpool = None

CFG = dict(
    min_dte=7, max_dte=21,     # Laufzeit in Tagen wie im PowerX Optimizer
    min_days_to_earnings=30,
    max_rec_mean=2.5,          # Yahoo-Skala: 1 = Strong Buy ... 5 = Sell
    min_target_upside=0.10,    # Kursziel mind. 10 % über Kurs
    min_open_interest=50,
    max_spread_pct=0.35,       # Spread max. 35 % der Prämie (PXO-Beispiel NFLX: 28 %)
    max_dte_fallback=28,       # nur für Aktien ohne Wochenoptionen: nächste Monatsoption bis 28 Tage
    min_yield_pa=0.20,         # Untergrenze 20 % p. a. (Ziel 30-40 %), auf volles Kapital wie PXO
    max_yield_pa=0.40,
    min_premium=0.10,          # Prämie mind. 0,10 je Aktie
    min_strike=15,
    max_pe=50,                 # KGV höchstens 50
    risk_free=0.04,
    min_price=15,
    signal_weeks=2,            # 1. grüne Woche = "neu, vorläufig"; 2. Woche = "bestätigt" (Vorwoche grün abgeschlossen)
    stoch_rising=True,         # Stoch RSI K muss gegenüber der Vorwoche steigen
    min_market_cap=10e9,       # nur große, liquide Werte
    max_per_sector=2,
    top_n=15,
    min_aaqs=6,
    aaqs_borderline=5,         # AAQS 5: als Grenzfall getrennt anzeigen (im Terminal prüfen)
    em_required=False,         # True = Strike muss außerhalb der erwarteten Bewegung liegen
    estimate_premium=True,     # Spread in den Daten unrealistisch breit: Prämie aus IV schätzen (zwischen Geld und Brief)
    max_spread_est=1.5,        # bis zu dieser Spreadbreite (150 % der Prämie) wird geschätzt, darüber verworfen
    support_required=False,    # True = Strike muss unter der Unterstützung liegen (verträgt sich nicht mit
                               #        Einstieg direkt nach dem Dreh, deshalb nur angezeigt)
)
OUT_DIR = os.environ.get("OUT_DIR", "docs")
TODAY = dt.date.today()
DIAG = {x.strip().upper() for x in os.environ.get("DIAGNOSE", "").replace(";", ",").replace(" ", ",").split(",") if x.strip()}


def dlog(tk, msg):
    if tk in DIAG:
        print(f"  [Diagnose {tk}] {msg}")


# ---------- Universum & Markt ----------
def load_universe():
    if os.path.exists("watchlist.txt"):
        with open("watchlist.txt") as f:
            return [l.strip().upper() for l in f if l.strip() and not l.startswith("#")]
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = requests.get(url, headers={"User-Agent": "Mozilla/5.0 wheel-screener"}, timeout=30).text
    table = pd.read_html(io.StringIO(html))[0]
    return [s.replace(".", "-") for s in table["Symbol"]]


def market_regime():
    vix = float(yf.Ticker("^VIX").history(period="5d")["Close"].iloc[-1])
    spx = yf.Ticker("^GSPC").history(period="1y")["Close"]
    above = bool(spx.iloc[-1] > spx.rolling(200).mean().iloc[-1])
    if vix > 30 or (not above and vix > 22):
        return vix, above, 0.15, "Angespannt"
    if vix > 22 or not above:
        return vix, above, 0.20, "Vorsichtig"
    return vix, above, 0.30, "Ruhig"


# ---------- Indikatoren ----------
def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def tema(x, n):
    e1 = x.ewm(span=n, adjust=False).mean()
    e2 = e1.ewm(span=n, adjust=False).mean()
    e3 = e2.ewm(span=n, adjust=False).mean()
    return 3 * e1 - 3 * e2 + e3


def to_weekly(df):
    w = df[["Open", "High", "Low", "Close"]].dropna(how="all").resample("W-FRI").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last"})
    return w.dropna(subset=["Close"])


def weekly_signal(df):
    """Einstiegssignal im Wochenchart. Gibt ein Dict mit allen Werten zurück:
    green_weeks = seit wie vielen Wochen grün (0 = diese Woche nicht grün)."""
    w = to_weekly(df)
    c = w["Close"]
    t8, t21, t200 = tema(c, 8), tema(c, 21), tema(c, 200)
    r = rsi(c, 14)
    lo, hi = r.rolling(14).min(), r.rolling(14).max()
    k = ((r - lo) / (hi - lo) * 100).rolling(3).mean()
    d = k.rolling(3).mean()
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    sig = macd.ewm(span=9, adjust=False).mean()
    hist = macd - sig
    c_tema = t8 > t21
    c_stoch = (k > d) & ((k > k.shift(1)) if CFG["stoch_rising"] else True)
    c_macd = hist > 0
    green = (c_tema & c_stoch & c_macd).fillna(False).tolist()
    n = 0
    for g in reversed(green):
        if not g:
            break
        n += 1
    last = lambda s: float(s.iloc[-1]) if len(s) and not pd.isna(s.iloc[-1]) else float("nan")
    prev = lambda s: float(s.iloc[-2]) if len(s) > 1 and not pd.isna(s.iloc[-2]) else float("nan")
    return dict(green_weeks=n, weeks=len(w), close=last(c),
                tema8=last(t8), tema21=last(t21), tema200=last(t200),
                k=last(k), d=last(d), k_prev=prev(k), hist=last(hist), hist_prev=prev(hist),
                macd=last(macd), signal=last(sig),
                ok_tema=bool(c_tema.iloc[-1]), ok_stoch=bool(c_stoch.iloc[-1]) if hasattr(c_stoch, "iloc") else True,
                ok_macd=bool(c_macd.iloc[-1]),
                below_tema200=bool(last(c) < last(t200)),
                provisional=(n == 1 and TODAY.weekday() < 5))


def signal_text(s):
    j = lambda b: "✓" if b else "✗"
    return (f"Woche: Kurs {s['close']:.2f} | TEMA 8 {s['tema8']:.2f} / 21 {s['tema21']:.2f} {j(s['ok_tema'])} | "
            f"Stoch RSI K {s['k']:.1f} (Vorwoche {s['k_prev']:.1f}) / D {s['d']:.1f} {j(s['ok_stoch'])} | "
            f"MACD-Hist {s['hist']:.4f} (Vorwoche {s['hist_prev']:.4f}) {j(s['ok_macd'])} | "
            f"TEMA 200 {s['tema200']:.2f} ({'Kurs darunter' if s['below_tema200'] else 'Kurs darüber'}) | "
            f"grün seit {s['green_weeks']} Woche(n)")


def support_level(df, price):
    """Höchstes markantes Tief der letzten 6 Monate unterhalb des Kurses."""
    low = df["Low"].dropna().iloc[-120:]
    pivots = [low.iloc[i] for i in range(5, len(low) - 5) if low.iloc[i] == low.iloc[i - 5:i + 6].min()]
    below = [p for p in pivots if p < price * 0.98]
    return float(max(below)) if below else float(low.iloc[-60:].min())


def next_earnings(t):
    """Nächster Earnings-Termin, bewusst vorsichtig: frühester Wert aus
    Yahoo-Kalender, Yahoo-Earnings-Liste und Schätzung (letzter Termin + 85 Tage)."""
    cands = []
    try:
        cal = t.calendar
        if isinstance(cal, dict):
            ed = cal.get("Earnings Date")
        elif cal is not None and "Earnings Date" in getattr(cal, "index", []):
            ed = cal.loc["Earnings Date"].tolist()
        else:
            ed = None
        if ed is not None:
            for x in (ed if isinstance(ed, (list, tuple)) else [ed]):
                cands.append(pd.Timestamp(x).date())
    except Exception:
        pass
    try:
        eds = t.get_earnings_dates(limit=8)
        if eds is not None and len(eds):
            dates = sorted({pd.Timestamp(x).date() for x in eds.index})
            cands += [d for d in dates if d >= TODAY]
            past = [d for d in dates if d < TODAY]
            if past:
                est = max(past) + dt.timedelta(days=85)
                cands.append(est if est >= TODAY else TODAY)
    except Exception:
        pass
    future = [d for d in cands if d >= TODAY]
    return min(future) if future else None


def put_delta(S, K, T, iv, r):
    if iv <= 0 or T <= 0:
        return None
    d1 = (math.log(S / K) + (r + iv * iv / 2) * T) / (iv * math.sqrt(T))
    return abs(0.5 * (1 + math.erf(d1 / math.sqrt(2))) - 1)


def put_price(S, K, T, iv, r):
    """Theoretischer Put-Preis (Black-Scholes)."""
    if iv <= 0 or T <= 0:
        return None
    N = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
    d1 = (math.log(S / K) + (r + iv * iv / 2) * T) / (iv * math.sqrt(T))
    d2 = d1 - iv * math.sqrt(T)
    return K * math.exp(-r * T) * N(-d2) - S * N(-d1)


# ---------- Alpaca ----------
ALPACA_KEY, ALPACA_SECRET = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
_ALPACA_ERR = {"printed": False}


def _alpaca_get(url, params):
    h = {"APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET}
    out, tok = [], None
    for _ in range(10):
        q = dict(params)
        if tok:
            q["page_token"] = tok
        r = requests.get(url, headers=h, params=q, timeout=20)
        r.raise_for_status()
        j = r.json()
        out.append(j)
        tok = j.get("next_page_token")
        if not tok:
            break
    return out


def alpaca_puts(tk, e):
    """Put-Kette einer Laufzeit von Alpaca (Kurse, Delta, IV) plus Open Interest."""
    if not (ALPACA_KEY and ALPACA_SECRET):
        return None
    try:
        snaps = {}
        for j in _alpaca_get(f"https://data.alpaca.markets/v1beta1/options/snapshots/{tk}",
                             dict(feed="indicative", type="put", expiration_date=e, limit=1000)):
            snaps.update(j.get("snapshots") or {})
        oi = {}
        try:
            for j in _alpaca_get("https://paper-api.alpaca.markets/v2/options/contracts",
                                 dict(underlying_symbols=tk, expiration_date=e, type="put", limit=1000)):
                for c in j.get("option_contracts") or []:
                    oi[c.get("symbol")] = int(float(c.get("open_interest") or 0))
        except Exception as ex:
            dlog(tk, f"Alpaca Open Interest nicht abrufbar: {ex}")
        rows = []
        for sym, sn in snaps.items():
            q, g = sn.get("latestQuote") or {}, sn.get("greeks") or {}
            rows.append(dict(strike=int(sym[-8:]) / 1000, bid=float(q.get("bp") or 0), ask=float(q.get("ap") or 0),
                             openInterest=oi.get(sym, 0), impliedVolatility=float(sn.get("impliedVolatility") or 0),
                             delta=abs(float(g.get("delta") or 0))))
        return pd.DataFrame(rows).sort_values("strike").reset_index(drop=True) if rows else None
    except Exception as ex:
        if not _ALPACA_ERR["printed"]:
            print(f"    Alpaca nicht erreichbar: {ex}")
            _ALPACA_ERR["printed"] = True
        dlog(tk, f"Alpaca {e} nicht abrufbar: {ex}")
        return None


def fill_from(df, other, cols_if_missing_price=("bid", "ask"), fill_zero=("openInterest", "impliedVolatility", "delta")):
    """Ergänzt fehlende Werte in df je Strike aus other. Gibt Anzahl ergänzter Preise zurück."""
    if other is None or not len(other):
        return 0
    o = other.drop_duplicates("strike").set_index("strike")
    n = 0
    for i in df.index:
        K = df.at[i, "strike"]
        if K not in o.index:
            continue
        r = o.loc[K]
        if (df.at[i, "bid"] <= 0 or df.at[i, "ask"] <= 0) and float(r.get("bid") or 0) > 0 and float(r.get("ask") or 0) > 0:
            for c in cols_if_missing_price:
                df.at[i, c] = float(r.get(c) or 0)
            n += 1
        for c in fill_zero:
            if c in df.columns and c in r.index and (pd.isna(df.at[i, c]) or not df.at[i, c]) and not pd.isna(r.get(c)):
                df.at[i, c] = r.get(c)
    return n


# ---------- Eulerpool ----------
_EP = None
_EP_DEBUG = {"printed": 0}


def ep_client():
    global _EP
    if _EP is None and Eulerpool and os.environ.get("EULERPOOL_API_KEY"):
        _EP = Eulerpool(os.environ["EULERPOOL_API_KEY"], use_auth_header=True)
    return _EP


def find_number(obj, words):
    """Sucht in einer beliebigen JSON-Antwort die erste Zahl unter einem passenden Schlüssel."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if any(w in k.lower().replace("_", "") for w in words) and isinstance(v, (int, float)) and not isinstance(v, bool):
                return float(v)
        for v in obj.values():
            r = find_number(v, words)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj[:5]:
            r = find_number(v, words)
            if r is not None:
                return r
    return None


def ep_call(label, calls, words):
    for fn, ident in calls:
        if not ident:
            continue
        try:
            res = fn(ident)
        except Exception as ex:
            print(f"    Eulerpool {label} {ident}: {ex}", file=sys.stderr)
            continue
        if _EP_DEBUG["printed"] < 2:  # erste Antworten ins Log, zur Kontrolle
            print(f"    [Eulerpool {label} {ident}] {str(res)[:400]}")
            _EP_DEBUG["printed"] += 1
        val = find_number(res, words)
        if val is not None:
            return val
    return None


def eulerpool_check(tk, t):
    """Liefert (aaqs, fair_value) oder (None, None), wenn nicht verfügbar."""
    ep = ep_client()
    if ep is None:
        return None, None
    try:
        isin = t.isin if t.isin and t.isin != "-" else None
    except Exception:
        isin = None
    aaqs = ep_call("AAQS", [(ep.aaqs.by_isin, isin), (ep.equity.aaqs, tk)], ["aaqs", "score"])
    fair = ep_call("FairValue", [(ep.fair_value.by_isin, isin)], ["fairvalue", "fair"])
    return aaqs, fair



# ---------- Optionsketten: Eulerpool zuerst, Yahoo als Ersatz ----------
SOURCE_STATS = {"Alpaca": 0, "Eulerpool": 0, "Yahoo": 0, "Eulerpool + Yahoo-Preise": 0}
_EP_PARAM = {"name": None}   # welcher Parameter bei Eulerpool die Laufzeit wählt


def _ep_rows(res):
    if isinstance(res, dict):
        res = res.get("data") or res.get("options") or res.get("results") or []
    return res if isinstance(res, list) else []


def _ep_puts(rows, e):
    out = []
    for r in rows:
        if str(r.get("type", "")).lower() != "put" or str(r.get("expiration_date", ""))[:10] != e:
            continue
        g = r.get("greeks") or {}
        out.append(dict(strike=float(r.get("strike") or 0), bid=float(r.get("bid") or 0),
                        ask=float(r.get("ask") or 0), openInterest=r.get("open_interest") or 0,
                        impliedVolatility=float(r.get("impliedVol") or 0),
                        delta=abs(float(g.get("delta") or 0))))
    return pd.DataFrame(out)


class ChainSource:
    """Liefert Put-Ketten je Laufzeit für eine Aktie."""

    def __init__(self, tk, t):
        self.tk, self.t, self.ep_rows = tk, t, None
        ep = ep_client()
        if ep is not None:
            try:
                self.ep_rows = _ep_rows(ep.derivatives.options_greeks(tk))
            except Exception as ex:
                print(f"    Eulerpool Optionen {tk}: {ex}", file=sys.stderr)

    def _ep_for(self, e):
        if self.ep_rows is None:
            return None
        df = _ep_puts(self.ep_rows, e)
        if len(df):
            return df
        ep = ep_client()
        names = [_EP_PARAM["name"]] if _EP_PARAM["name"] else ["expiration", "expiration_date", "expiry"]
        for n in names:
            try:
                rows = _ep_rows(ep.derivatives.options_greeks(self.tk, **{n: e}))
            except Exception:
                continue
            df = _ep_puts(rows, e)
            if len(df):
                if _EP_PARAM["name"] is None:
                    print(f"    Eulerpool: Laufzeit wird über Parameter '{n}' gewählt")
                _EP_PARAM["name"] = n
                self.ep_rows += rows
                return df
        return None

    def _yahoo(self, e):
        try:
            return self.t.option_chain(e).puts
        except Exception as ex:
            print(f"    Yahoo Optionen {self.tk} {e}: {ex}", file=sys.stderr)
            dlog(self.tk, f"Yahoo-Optionskette {e} nicht abrufbar: {ex}")
            return None

    def puts(self, e):
        a = alpaca_puts(self.tk, e)
        if a is not None and len(a) and (a["bid"] > 0).any():
            miss = int(((a["bid"] <= 0) | (a["ask"] <= 0)).sum())
            dlog(self.tk, f"Alpaca {e}: {len(a)} Puts, davon {miss} ohne Preis")
            if miss:
                n = fill_from(a, self._ep_for(e))
                if miss - n > 0.2 * len(a):
                    n += fill_from(a, self._yahoo(e))
                dlog(self.tk, f"Alpaca {e}: {n} fehlende Preise aus Eulerpool/Yahoo ergänzt")
            SOURCE_STATS["Alpaca"] += 1
            return a
        df = self._ep_for(e)
        if df is not None and len(df) and (df["bid"] > 0).any():
            missing = (df["bid"] <= 0) | (df["ask"] <= 0)
            dlog(self.tk, f"Eulerpool {e}: {len(df)} Puts, davon {int(missing.sum())} ohne Preis")
            if missing.mean() > 0.2:
                # Eulerpool liefert für viele Strikes keinen Preis: aus Yahoo ergänzen
                y = self._yahoo(e)
                if y is not None and len(y):
                    y = y.set_index("strike")
                    for i in df.index:
                        K = df.at[i, "strike"]
                        if K in y.index:
                            yr = y.loc[K]
                            if isinstance(yr, pd.DataFrame):
                                yr = yr.iloc[0]
                            if missing[i]:
                                df.at[i, "bid"] = float(yr.get("bid") or 0)
                                df.at[i, "ask"] = float(yr.get("ask") or 0)
                            if not df.at[i, "openInterest"]:
                                oi = yr.get("openInterest")
                                df.at[i, "openInterest"] = 0 if pd.isna(oi) else int(oi)
                            if not df.at[i, "impliedVolatility"]:
                                df.at[i, "impliedVolatility"] = float(yr.get("impliedVolatility") or 0)
                    still = int(((df["bid"] <= 0) | (df["ask"] <= 0)).sum())
                    dlog(self.tk, f"Yahoo {e}: {len(y)} Puts geliefert, danach noch {still} ohne Preis")
                    SOURCE_STATS["Eulerpool + Yahoo-Preise"] += 1
                    return df
            SOURCE_STATS["Eulerpool"] += 1
            return df
        SOURCE_STATS["Yahoo"] += 1
        return self._yahoo(e)


# ---------- Optionen ----------
def best_put(t, price, support, max_delta, earn_date, src=None, tk=None):
    """Gibt (bester Put oder None, Diagnose) zurück."""
    best = None
    diag = {"Laufzeiten": 0, "kein Preis": 0, "Prämie unter 0,10": 0, "Spread zu breit": 0,
            "Open Interest zu niedrig": 0, "Delta zu hoch": 0,
            "nicht unter Unterstützung": 0, "Rendite unter Minimum": 0, "Rendite über Maximum": 0}
    near = None  # bester Put, der nur an Unterstützung oder Rendite scheitert
    all_exp = [(e, (dt.date.fromisoformat(e) - TODAY).days) for e in t.options]
    exps = [x for x in all_exp if CFG["min_dte"] <= x[1] <= CFG["max_dte"]]
    if not exps:  # keine Wochenoptionen: nächste Monatsoption bis max_dte_fallback
        exps = [x for x in all_exp if CFG["min_dte"] <= x[1] <= CFG["max_dte_fallback"]][:1]
    for e, dte in exps:
        exp = dt.date.fromisoformat(e)
        if earn_date and earn_date <= exp:
            continue
        diag["Laufzeiten"] += 1
        puts = src.puts(e) if src else t.option_chain(e).puts
        if puts is None or puts.empty:
            continue
        atm = puts.iloc[(puts["strike"] - price).abs().argsort()[:3]]
        iv_atm = float(atm["impliedVolatility"].median())
        if not iv_atm or iv_atm < 0.05:
            continue
        T = dte / 365
        em = price * iv_atm * math.sqrt(T)
        for _, o in puts.iterrows():
            K, bid, ask = float(o["strike"]), float(o["bid"] or 0), float(o["ask"] or 0)
            oi = 0 if pd.isna(o["openInterest"]) else int(o["openInterest"])
            if K >= price or K < CFG["min_strike"]:
                continue
            if tk in DIAG and K >= price * 0.90:   # Strikes bis 10 % unter Kurs einzeln zeigen
                _iv = float(o["impliedVolatility"]) if 0.05 < (o["impliedVolatility"] or 0) < 3 else iv_atm
                _d = put_delta(price, K, T, _iv, CFG["risk_free"]) or 0
                _m = (bid + ask) / 2
                _t = put_price(price, K, T, _iv, CFG["risk_free"]) or 0
                dlog(tk, f"  {e} Strike {K:g}: Geld {bid:.2f} / Brief {ask:.2f}, theor. {_t:.2f}, OI {oi}, "
                         f"Delta ca. {_d:.2f}, {(_m / K * 365 / dte * 100) if _m > 0 else 0:.0f} % p. a. (Mitte)")
            if bid <= 0 or ask <= 0:
                diag["kein Preis"] += 1; continue
            mid = (bid + ask) / 2
            est = False
            if (ask - bid) / mid > CFG["max_spread_pct"]:
                iv_k = float(o["impliedVolatility"]) if 0.05 < (o["impliedVolatility"] or 0) < 3 else iv_atm
                theo = put_price(price, K, T, iv_k, CFG["risk_free"])
                if CFG["estimate_premium"] and theo and (ask - bid) / mid <= CFG["max_spread_est"]:
                    mid, est = min(max(theo, bid), ask), True   # geschätzt, aber nie außerhalb Geld/Brief
                else:
                    diag["Spread zu breit"] += 1; continue
            if mid < CFG["min_premium"]:
                diag["Prämie unter 0,10"] += 1; continue
            if oi < CFG["min_open_interest"]:
                diag["Open Interest zu niedrig"] += 1; continue
            iv = float(o["impliedVolatility"]) if 0.05 < o["impliedVolatility"] < 3 else iv_atm
            ep_d = float(o["delta"]) if "delta" in o and not pd.isna(o["delta"]) else 0
            own_d = put_delta(price, K, T, iv, CFG["risk_free"]) or 0
            # vorsichtig: das höhere der beiden Deltas zählt (Eulerpool-Delta wirkte teils zu niedrig)
            d = max(ep_d, own_d) if 0 < ep_d < 1 else own_d
            if d is None or d > max_delta:
                diag["Delta zu hoch"] += 1; continue
            y = mid / K * 365 / dte
            reason = None
            if CFG["support_required"] and K >= support:
                reason = "nicht unter Unterstützung"
            elif CFG["em_required"] and K > price - em:
                reason = "innerhalb erwarteter Bewegung"
            elif y < CFG["min_yield_pa"]:
                reason = "Rendite unter Minimum"
            elif y > CFG["max_yield_pa"]:
                reason = "Rendite über Maximum"
            if reason:
                diag[reason] = diag.get(reason, 0) + 1
                if near is None or y > near["y"]:
                    near = dict(K=K, e=e, d=d, y=y, why=reason)
                continue
            cand = dict(expiry=e, dte=dte, strike=K, premium=round(mid, 2), delta=round(d, 3),
                        yield_pa=y, em=em, iv=iv_atm, oi=oi, est=est, bid=bid, ask=ask)
            # Sicherheit zuerst: im Renditekorridor den Put mit dem niedrigsten Delta nehmen
            if best is None or d < best["delta"]:
                best = cand
    diag["near"] = near
    return best, diag


# ---------- Ausgabe ----------
def fmt(x, d=2):
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def ep_txt(x, d=2):
    return "in Eulerpool prüfen" if x is None or pd.isna(x) else fmt(x, d)


def cards_html(df, border=False):
    cards = []
    for i, r in enumerate(df.itertuples(), 1):
        cards.append(f"""<article{' class="border"' if border else ''}>
<h2>{i}. {r.ticker} <small>{r.name}</small></h2>
<p class="put">Put {fmt(r.strike)} zum {dt.date.fromisoformat(r.expiry).strftime('%d.%m.%Y')} für ca. {fmt(r.premium)}</p>
{f'<p class="estw">Prämie geschätzt (Daten: Geld {fmt(r.bid)} / Brief {fmt(r.ask)}). Echten Preis im Broker prüfen.</p>' if getattr(r, 'est', False) else ''}
<dl>
<div><dt>Rendite p. a.</dt><dd>{fmt(r.yield_pa*100,1)} %</dd></div>
<div><dt>Delta</dt><dd>{fmt(r.delta)}</dd></div>
<div><dt>Kurs</dt><dd>{fmt(r.price)}</dd></div>
<div><dt>Unterstützung</dt><dd>{fmt(r.support)} {'(Strike darunter)' if r.strike < r.support else '(Strike darüber)'}</dd></div>
<div><dt>Abstand zum Strike</dt><dd>{fmt((r.price-r.strike)/r.price*100,1)} %</dd></div>
<div><dt>Erwartete Bewegung</dt><dd>±{fmt(r.em)} {'(Strike außerhalb)' if r.strike <= r.price - r.em else '(Strike innerhalb)'}</dd></div>
<div><dt>Earnings</dt><dd>{r.earnings.strftime('%d.%m.%Y')}</dd></div>
<div><dt>Kursziel Analysten</dt><dd>{fmt(r.target)} (+{fmt(r.upside*100,0)} %)</dd></div>
<div><dt>AAQS</dt><dd>{ep_txt(r.aaqs, 1)}</dd></div>
<div><dt>Fair Value</dt><dd>{ep_txt(r.fair_value)}{'' if r.fair_value is None or pd.isna(r.fair_value) else f" (Kurs {fmt((1-r.price/r.fair_value)*100,0)} % darunter)"}</dd></div>
</dl>
<p class="signal{' warn2' if r.below_tema200 else ''}">Wochensignal: {'NEU diese Woche, vorläufig (Wochenkerze schließt Freitag)' if r.provisional else ('bestätigt, letzte Woche grün geschlossen' if r.signal_weeks >= 2 else 'bestätigt')}{f" · Kurs unter TEMA 200 ({fmt(r.tema200)}): Gegentrend, höheres Risiko" if r.below_tema200 else ''}</p>
<p class="meta">{r.sector}, KGV {fmt(r.pe,1)}, Open Interest {r.oi}.</p>
</article>""")
    return "\n".join(cards)


def write_report(df, vix, above, max_delta, regime, border=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    border = border if border is not None else pd.DataFrame()
    pd.concat([df, border]).to_csv(os.path.join(OUT_DIR, "treffer.csv"), index=False)
    colors = {"Ruhig": "#2E7D4F", "Vorsichtig": "#B7791F", "Angespannt": "#B23A34"}
    body = cards_html(df) if len(df) else "<p class='empty'>Heute erfüllt keine Aktie alle Kriterien. Kein Trade ist auch ein Trade.</p>"
    if len(border):
        body += ("\n<h2 class='section'>Grenzfälle: AAQS im Terminal prüfen</h2>\n"
                 "<p class='warn'>Diese Aktien haben laut Eulerpool-API einen AAQS von 5. API und Terminal können "
                 "um einen Punkt abweichen. Nur handeln, wenn dein Eulerpool-Terminal mindestens 6 anzeigt.</p>\n"
                 + cards_html(border, border=True))
    html = f"""<!DOCTYPE html><html lang="de"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Wheel-Screener</title>
<style>
body{{margin:0;background:#EDF0EE;color:#1B2B2A;font:16px/1.5 system-ui,sans-serif}}
main{{max-width:720px;margin:0 auto;padding:16px}}
h1{{margin:0 0 4px;font-size:28px}} .sub{{color:#5D6B69;margin:0 0 14px}}
.market{{background:#fff;border-left:5px solid {colors[regime]};padding:10px 12px;border-radius:0 8px 8px 0;margin-bottom:16px}}
article{{background:#fff;border:1px solid #C9D1CE;border-radius:10px;padding:14px;margin-bottom:12px}}
h2{{margin:0;font-size:20px}} h2 small{{color:#5D6B69;font-weight:400;font-size:14px}}
.put{{font-weight:600;color:#2F5D62;margin:4px 0 10px}}
dl{{display:grid;grid-template-columns:1fr 1fr;gap:6px 12px;margin:0}} dt{{font-size:13px;color:#5D6B69}} dd{{margin:0;font-weight:600}}
.meta{{font-size:14px;color:#5D6B69;margin:10px 0 0}} .empty{{text-align:center;color:#5D6B69;padding:30px}}
h2.section{{margin:28px 0 6px;font-size:22px;color:#B7791F}}
.warn{{background:#FBF4E6;border-left:5px solid #B7791F;padding:10px 12px;border-radius:0 8px 8px 0;font-size:15px}}
article.border{{border:2px dashed #B7791F}}
.signal{{font-size:14px;margin:10px 0 0;padding:6px 10px;border-radius:6px;background:#E8F3EC;color:#2E5E40}}
.signal.warn2{{background:#FBF4E6;color:#7A5212}}
.estw{{font-size:14px;margin:-6px 0 10px;color:#7A5212}}
</style></head><body><main>
<h1>Wheel-Screener</h1>
<p class="sub">Stand {dt.datetime.now().strftime('%d.%m.%Y, %H:%M')} UTC · Kurse: Yahoo · Optionen: {', '.join(k for k, v in SOURCE_STATS.items() if v) or 'keine'} (leicht verzögert)</p>
<div class="market"><b>Markt: {regime}</b>. VIX {fmt(vix,1)}, S&amp;P 500 {'über' if above else 'unter'} der 200-Tage-Linie. Maximales Delta heute {fmt(max_delta)}.</div>
{body}
<p class="meta">Prämien vor dem Handel immer im Broker prüfen. Keine Anlageberatung.</p>
</main></body></html>"""
    with open(os.path.join(OUT_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)


# ---------- Ablauf ----------
def main():
    vix, above, max_delta, regime = market_regime()
    print(f"Markt: {regime}, VIX {vix:.1f}, max. Delta {max_delta}")
    tickers = load_universe()
    for tk in DIAG:
        if tk not in tickers:
            tickers.append(tk)
    print(f"Scanne {len(tickers)} Aktien ...")
    if DIAG:
        print(f"Diagnose für: {', '.join(sorted(DIAG))}")
    # 10 Jahre Tageskurse, damit die Wochen-TEMA 200 sauber berechnet wird
    hist = yf.download(tickers, period="10y", group_by="ticker", auto_adjust=True, threads=True, progress=False)

    stage1 = []
    n_green, n_below = 0, 0
    for tk in tickers:
        try:
            df = hist[tk].dropna(how="all") if len(tickers) > 1 else hist.dropna(how="all")
        except KeyError:
            dlog(tk, "keine Kursdaten von Yahoo")
            continue
        if len(df) < 210:
            dlog(tk, f"zu wenig Kursdaten ({len(df)} Tage)")
            continue
        price = float(df["Close"].iloc[-1])
        if price < CFG["min_price"]:
            dlog(tk, f"Kurs {price:.2f} unter Mindestkurs {CFG['min_price']} ✗")
            continue
        sig = weekly_signal(df)
        fresh = 1 <= sig["green_weeks"] <= CFG["signal_weeks"]
        dlog(tk, signal_text(sig))
        if fresh:
            n_green += 1
            n_below += sig["below_tema200"]
            dlog(tk, f"Einstiegssignal ✓ ({'neu, vorläufig, Wochenkerze noch offen' if sig['provisional'] else 'bestätigt'})")
            stage1.append((tk, df, price, sig, False))
        elif tk in DIAG:
            why = ("diese Woche nicht grün" if sig["green_weeks"] == 0
                   else f"schon seit {sig['green_weeks']} Wochen grün, kein frisches Signal")
            dlog(tk, f"Einstiegssignal ✗ ({why}), die weiteren Kriterien werden trotzdem geprüft")
            stage1.append((tk, df, price, sig, True))
    print(f"{n_green} Aktien mit frischem Wochensignal, davon {n_below} unter TEMA 200.")
    print("Prüfe Fundamentaldaten und Optionen ...")
    drop = {k: 0 for k in ["Marktkapitalisierung", "Analysten/Kursziel", "KGV über 50", "Earnings zu nah",
                           "Kein passender Put", "AAQS unter 5", "Nicht unterbewertet", "Fehler"]}

    rows = []
    for tk, df, price, sig, diag_only in stage1:
        try:
            # diag_only: Aktie ohne Signal, nur für die Diagnose geprüft (nicht in Zählung und Ergebnis)
            def fail(key, msg):
                if not diag_only:
                    drop[key] += 1
                dlog(tk, msg + " ✗")
            t = yf.Ticker(tk)
            info = t.info
            rec, tgt = info.get("recommendationMean"), info.get("targetMeanPrice")
            mcap = info.get("marketCap") or 0
            if mcap < CFG["min_market_cap"]:
                fail("Marktkapitalisierung", f"Marktkapitalisierung {mcap/1e9:.1f} Mrd. unter {CFG['min_market_cap']/1e9:.0f} Mrd."); continue
            dlog(tk, f"Marktkapitalisierung {mcap/1e9:.1f} Mrd. ✓")
            if rec is None or rec > CFG["max_rec_mean"] or not tgt or tgt < price * (1 + CFG["min_target_upside"]):
                fail("Analysten/Kursziel", f"Analysten {rec} (max. {CFG['max_rec_mean']}), Kursziel {tgt} "
                     f"({'–' if not tgt else f'{(tgt/price-1)*100:+.1f} %'}, mind. +{CFG['min_target_upside']*100:.0f} %)"); continue
            dlog(tk, f"Analysten {rec}, Kursziel {tgt} ({(tgt/price-1)*100:+.1f} %) ✓")
            pe = info.get("trailingPE")
            if pe is None or pe <= 0 or pe > CFG["max_pe"]:
                fail("KGV über 50", f"KGV {pe} (max. {CFG['max_pe']})"); continue
            dlog(tk, f"KGV {pe:.1f} ✓")
            earn = next_earnings(t)
            if earn is None or (earn - TODAY).days < CFG["min_days_to_earnings"]:
                fail("Earnings zu nah", f"Earnings {earn} (mind. {CFG['min_days_to_earnings']} Tage)"); continue
            dlog(tk, f"Earnings {earn} ({(earn-TODAY).days} Tage) ✓")
            sup = support_level(df, price)
            put, diag = best_put(t, price, sup, max_delta, earn, ChainSource(tk, t), tk)
            if tk in DIAG:
                nr = diag.get("near")
                dlog(tk, f"Unterstützung {sup:.2f} ({(sup/price-1)*100:.1f} %), max. Delta {max_delta}, "
                         f"Rendite {CFG['min_yield_pa']*100:.0f}-{CFG['max_yield_pa']*100:.0f} % p. a., Laufzeit {CFG['min_dte']}-{CFG['max_dte']} Tage")
                dlog(tk, "Puts: " + ", ".join(f"{k} {v}" for k, v in diag.items() if k != "near" and v))
                if nr:
                    dlog(tk, f"bester knapper Put: Strike {nr['K']:g} ({nr['e']}), Delta {nr['d']:.2f}, "
                             f"{nr['y']*100:.0f} % p. a., scheitert an: {nr['why']}")
            if not put:
                if diag_only:
                    dlog(tk, "kein passender Put ✗"); continue
                drop["Kein passender Put"] += 1
                nr = diag.pop("near")
                info_txt = ", ".join(f"{k} {v}" for k, v in diag.items() if v)
                near_txt = (f" | knapp: Strike {nr['K']:g} ({nr['e']}), Delta {nr['d']:.2f}, "
                            f"{nr['y']*100:.0f} % p. a., scheitert an: {nr['why']}") if nr else ""
                print(f"  {tk}: Kurs {price:.2f}, Unterstützung {sup:.2f} ({(sup/price-1)*100:.1f} %) | {info_txt or 'keine Laufzeit im Zeitfenster'}{near_txt}")
                continue
            dlog(tk, f"Put gefunden: Strike {put['strike']} ({put['expiry']}), Prämie {put['premium']}"
                     f"{' (geschätzt)' if put['est'] else ''}, Delta {put['delta']}, {put['yield_pa']*100:.0f} % p. a. ✓")
            aaqs, fair = eulerpool_check(tk, t)
            borderline = aaqs is not None and CFG["aaqs_borderline"] <= aaqs < CFG["min_aaqs"]
            if aaqs is not None and aaqs < CFG["aaqs_borderline"]:
                if not diag_only:
                    drop["AAQS unter 5"] += 1
                print(f"  {tk}: AAQS {aaqs} zu niedrig")
                continue
            dlog(tk, f"AAQS {aaqs} {'(Grenzfall)' if borderline else '✓'}")
            if fair is not None and price >= fair:
                if not diag_only:
                    drop["Nicht unterbewertet"] += 1
                print(f"  {tk}: nicht unterbewertet (Kurs {price:.2f}, Fair Value {fair:.2f})")
                continue
            dlog(tk, f"Fair Value {fair} ✓")
            if diag_only:
                dlog(tk, "alle übrigen Kriterien erfüllt, es fehlt nur das frische Wochensignal")
                continue
            rows.append(dict(ticker=tk, name=info.get("shortName", tk), sector=info.get("sector", "Unbekannt"),
                             price=price, support=sup, rec=rec, target=tgt, upside=tgt / price - 1,
                             earnings=earn, aaqs=aaqs, fair_value=fair, pe=pe,
                             signal_weeks=sig["green_weeks"], provisional=sig["provisional"],
                             below_tema200=sig["below_tema200"], tema200=sig["tema200"],
                             grenzfall=borderline, **put))
            print(f"  {'Grenzfall (AAQS 5)' if borderline else 'Treffer'}: {tk} Put {put['strike']} {put['expiry']}"
                  f"{' (unter TEMA 200)' if sig['below_tema200'] else ''}")
        except Exception as ex:
            drop["Fehler"] += 1
            print(f"  {tk}: übersprungen ({ex})", file=sys.stderr)
    print("Optionsketten: " + ", ".join(f"{k} {v}" for k, v in SOURCE_STATS.items()))
    print("Ausgeschieden nach Kriterium:")
    for k, v in drop.items():
        print(f"  {k}: {v}")

    res = pd.DataFrame(rows)
    border = pd.DataFrame()
    if not res.empty:
        res["abstand_em"] = (res["price"] - res["strike"]) / res["em"]
        # Sicherheit zuerst: niedriges Delta und großer Abstand zählen mehr als Rendite
        res["score"] = ((1 - res["delta"]) * 0.5
                        + res["abstand_em"].clip(upper=2) / 2 * 0.3
                        + res["yield_pa"].clip(upper=0.4) / 0.4 * 0.2)
        res = res.sort_values("score", ascending=False)
        border = res[res["grenzfall"]].head(5)
        res = res[~res["grenzfall"]]
        res = res.groupby("sector", sort=False).head(CFG["max_per_sector"]).head(CFG["top_n"])
    write_report(res, vix, above, max_delta, regime, border)
    print(f"Fertig: {len(res)} Trades und {len(border)} Grenzfälle in {OUT_DIR}/index.html")


if __name__ == "__main__":
    main()
