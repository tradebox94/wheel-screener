#!/usr/bin/env python3
"""
Wheel-Screener: sucht täglich Cash-Secured Puts, die deine Wheel-Kriterien erfüllen.

Automatisch geprüft:
  Marktumfeld (VIX, S&P 500 über 200-Tage-Linie) -> maximales Delta
  Trend dreht nach oben (über 200-Tage-Linie, RSI dreht aus Rücksetzer)
  PowerX-Signal frisch grün (RSI 7, Stochastik 14/3/3, MACD 12/26/9)
  KGV <= 50, Strike >= 15, Prämie >= 0,10, Rendite 30-40 % p. a.
  Analysten positiv, Kursziel deutlich über Kurs
  Earnings mindestens 30 Tage entfernt und erst nach dem Verfall
  Put mit Laufzeit 30-45 Tage, Delta unter Grenze,
  Strike unter Unterstützung und außerhalb der erwarteten Bewegung
  Liquidität (Open Interest, Spread), Mindestrendite
  Streuung: höchstens 2 Treffer pro Branche

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
    min_dte=30, max_dte=45,
    min_days_to_earnings=30,
    max_rec_mean=2.5,          # Yahoo-Skala: 1 = Strong Buy ... 5 = Sell
    min_target_upside=0.10,    # Kursziel mind. 10 % über Kurs
    min_open_interest=100,
    max_spread_pct=0.25,       # Spread max. 25 % der Prämie
    min_yield_pa=0.30,         # Rendite pro Jahr 30-40 % wie im PowerX Optimizer
    max_yield_pa=0.40,
    min_premium=0.10,          # Prämie mind. 0,10 je Aktie
    min_strike=15,
    max_pe=50,                 # KGV höchstens 50
    risk_free=0.04,
    min_price=15,
    powerx_fresh_days=10,      # PowerX muss in den letzten 10 Tagen auf grün gedreht sein
    min_market_cap=10e9,       # nur große, liquide Werte
    max_per_sector=2,
    top_n=15,
    min_aaqs=6,
)
OUT_DIR = os.environ.get("OUT_DIR", "docs")
TODAY = dt.date.today()


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


def trend_turn(df):
    """Aufwärtstrend intakt und Aktie dreht nach einem Rücksetzer wieder nach oben."""
    c = df["Close"].dropna()
    if len(c) < 210:
        return False
    sma200 = c.rolling(200).mean().iloc[-1]
    sma10 = c.rolling(10).mean().iloc[-1]
    r = rsi(c)
    r_min, r_now = r.iloc[-10:].min(), r.iloc[-1]
    return c.iloc[-1] > sma200 and c.iloc[-1] > sma10 and r_min < 45 and r_now > r_min + 5 and r_now < 65


def powerx_green_days(df):
    """PowerX-Signal: RSI(7) > 50, Slow Stochastik(14,3,3) > 50, MACD(12,26,9) über Signallinie.
    Gibt zurück, seit wie vielen Tagen das Signal grün ist (0 = heute nicht grün)."""
    c, h, l = df["Close"], df["High"], df["Low"]
    r7 = rsi(c, 7)
    fast_k = (c - l.rolling(14).min()) / (h.rolling(14).max() - l.rolling(14).min()) * 100
    slow_k = fast_k.rolling(3).mean()
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    hist = macd - macd.ewm(span=9, adjust=False).mean()
    green = ((r7 > 50) & (slow_k > 50) & (hist > 0)).fillna(False).tolist()
    n = 0
    for g in reversed(green):
        if not g:
            break
        n += 1
    return n


def support_level(df, price):
    """Höchstes markantes Tief der letzten 6 Monate unterhalb des Kurses."""
    low = df["Low"].dropna().iloc[-120:]
    pivots = [low.iloc[i] for i in range(5, len(low) - 5) if low.iloc[i] == low.iloc[i - 5:i + 6].min()]
    below = [p for p in pivots if p < price * 0.98]
    return float(max(below)) if below else float(low.iloc[-60:].min())


def next_earnings(t):
    try:
        cal = t.calendar
        if isinstance(cal, dict):
            ed = cal.get("Earnings Date")
        elif cal is not None and "Earnings Date" in getattr(cal, "index", []):
            ed = cal.loc["Earnings Date"].tolist()
        else:
            ed = None
        if ed is None:
            return None
        ed = ed if isinstance(ed, (list, tuple)) else [ed]
        future = [pd.Timestamp(x).date() for x in ed if pd.Timestamp(x).date() >= TODAY]
        return min(future) if future else None
    except Exception:
        return None


def put_delta(S, K, T, iv, r):
    if iv <= 0 or T <= 0:
        return None
    d1 = (math.log(S / K) + (r + iv * iv / 2) * T) / (iv * math.sqrt(T))
    return abs(0.5 * (1 + math.erf(d1 / math.sqrt(2))) - 1)


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


# ---------- Optionen ----------
def best_put(t, price, support, max_delta, earn_date):
    best = None
    for e in t.options:
        exp = dt.date.fromisoformat(e)
        dte = (exp - TODAY).days
        if not (CFG["min_dte"] <= dte <= CFG["max_dte"]):
            continue
        if earn_date and earn_date <= exp:
            continue
        puts = t.option_chain(e).puts
        if puts.empty:
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
            if K >= support or K > price - em or K < CFG["min_strike"] or bid <= 0 or ask <= 0:
                continue
            mid = (bid + ask) / 2
            if mid < CFG["min_premium"] or (ask - bid) / mid > CFG["max_spread_pct"] or oi < CFG["min_open_interest"]:
                continue
            iv = float(o["impliedVolatility"]) if o["impliedVolatility"] > 0.05 else iv_atm
            d = put_delta(price, K, T, iv, CFG["risk_free"])
            if d is None or d > max_delta:
                continue
            y = mid / K * 365 / dte
            if not (CFG["min_yield_pa"] <= y <= CFG["max_yield_pa"]):
                continue
            cand = dict(expiry=e, dte=dte, strike=K, premium=round(mid, 2), delta=round(d, 3),
                        yield_pa=y, em=em, iv=iv_atm, oi=oi)
            # Sicherheit zuerst: im Renditekorridor den Put mit dem niedrigsten Delta nehmen
            if best is None or d < best["delta"]:
                best = cand
    return best


# ---------- Ausgabe ----------
def fmt(x, d=2):
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def ep_txt(x, d=2):
    return "in Eulerpool prüfen" if x is None or pd.isna(x) else fmt(x, d)


def write_report(df, vix, above, max_delta, regime):
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUT_DIR, "treffer.csv"), index=False)
    colors = {"Ruhig": "#2E7D4F", "Vorsichtig": "#B7791F", "Angespannt": "#B23A34"}
    cards = []
    for i, r in enumerate(df.itertuples(), 1):
        cards.append(f"""<article>
<h2>{i}. {r.ticker} <small>{r.name}</small></h2>
<p class="put">Put {fmt(r.strike)} zum {dt.date.fromisoformat(r.expiry).strftime('%d.%m.%Y')} für ca. {fmt(r.premium)}</p>
<dl>
<div><dt>Rendite p. a.</dt><dd>{fmt(r.yield_pa*100,1)} %</dd></div>
<div><dt>Delta</dt><dd>{fmt(r.delta)}</dd></div>
<div><dt>Kurs</dt><dd>{fmt(r.price)}</dd></div>
<div><dt>Unterstützung</dt><dd>{fmt(r.support)}</dd></div>
<div><dt>Abstand zum Strike</dt><dd>{fmt((r.price-r.strike)/r.price*100,1)} %</dd></div>
<div><dt>Erwartete Bewegung</dt><dd>±{fmt(r.em)}</dd></div>
<div><dt>Earnings</dt><dd>{r.earnings.strftime('%d.%m.%Y')}</dd></div>
<div><dt>Kursziel Analysten</dt><dd>{fmt(r.target)} (+{fmt(r.upside*100,0)} %)</dd></div>
<div><dt>AAQS</dt><dd>{ep_txt(r.aaqs, 1)}</dd></div>
<div><dt>Fair Value</dt><dd>{ep_txt(r.fair_value)}{'' if r.fair_value is None or pd.isna(r.fair_value) else f" (Kurs {fmt((1-r.price/r.fair_value)*100,0)} % darunter)"}</dd></div>
</dl>
<p class="meta">{r.sector}, KGV {fmt(r.pe,1)}, PowerX grün seit {r.powerx_days} Tag{'en' if r.powerx_days>1 else ''}, Open Interest {r.oi}.</p>
</article>""")
    body = "\n".join(cards) if cards else "<p class='empty'>Heute erfüllt keine Aktie alle Kriterien. Kein Trade ist auch ein Trade.</p>"
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
</style></head><body><main>
<h1>Wheel-Screener</h1>
<p class="sub">Stand {dt.datetime.now().strftime('%d.%m.%Y, %H:%M')} UTC, Yahoo-Daten, leicht verzögert</p>
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
    print(f"Scanne {len(tickers)} Aktien ...")
    hist = yf.download(tickers, period="1y", group_by="ticker", auto_adjust=True, threads=True, progress=False)

    stage1 = []
    funnel = {"Trend dreht nach oben": 0, "PowerX frisch grün": 0}
    for tk in tickers:
        try:
            df = hist[tk].dropna(how="all") if len(tickers) > 1 else hist.dropna(how="all")
        except KeyError:
            continue
        if len(df) < 210:
            continue
        price = float(df["Close"].iloc[-1])
        if price < CFG["min_price"] or not trend_turn(df):
            continue
        funnel["Trend dreht nach oben"] += 1
        gd = powerx_green_days(df)
        if 1 <= gd <= CFG["powerx_fresh_days"]:
            funnel["PowerX frisch grün"] += 1
            stage1.append((tk, df, price, gd))
    print(f"{funnel['Trend dreht nach oben']} Aktien mit Trenddreh, davon {len(stage1)} mit frischem PowerX-Signal.")
    print("Prüfe Fundamentaldaten und Optionen ...")
    drop = {k: 0 for k in ["Marktkapitalisierung", "Analysten/Kursziel", "KGV über 50", "Earnings zu nah",
                           "Kein passender Put", "AAQS unter 6", "Nicht unterbewertet", "Fehler"]}

    rows = []
    for tk, df, price, gd in stage1:
        try:
            t = yf.Ticker(tk)
            info = t.info
            rec, tgt = info.get("recommendationMean"), info.get("targetMeanPrice")
            if (info.get("marketCap") or 0) < CFG["min_market_cap"]:
                drop["Marktkapitalisierung"] += 1; continue
            if rec is None or rec > CFG["max_rec_mean"] or not tgt or tgt < price * (1 + CFG["min_target_upside"]):
                drop["Analysten/Kursziel"] += 1; continue
            pe = info.get("trailingPE")
            if pe is None or pe <= 0 or pe > CFG["max_pe"]:
                drop["KGV über 50"] += 1; continue
            earn = next_earnings(t)
            if earn is None or (earn - TODAY).days < CFG["min_days_to_earnings"]:
                drop["Earnings zu nah"] += 1; continue
            sup = support_level(df, price)
            put = best_put(t, price, sup, max_delta, earn)
            if not put:
                drop["Kein passender Put"] += 1; continue
            aaqs, fair = eulerpool_check(tk, t)
            if aaqs is not None and aaqs < CFG["min_aaqs"]:
                drop["AAQS unter 6"] += 1
                print(f"  {tk}: AAQS {aaqs} zu niedrig")
                continue
            if fair is not None and price >= fair:
                drop["Nicht unterbewertet"] += 1
                print(f"  {tk}: nicht unterbewertet (Kurs {price:.2f}, Fair Value {fair:.2f})")
                continue
            rows.append(dict(ticker=tk, name=info.get("shortName", tk), sector=info.get("sector", "Unbekannt"),
                             price=price, support=sup, rec=rec, target=tgt, upside=tgt / price - 1,
                             earnings=earn, aaqs=aaqs, fair_value=fair, pe=pe, powerx_days=gd, **put))
            print(f"  Treffer: {tk} Put {put['strike']} {put['expiry']}")
        except Exception as ex:
            drop["Fehler"] += 1
            print(f"  {tk}: übersprungen ({ex})", file=sys.stderr)
    print("Ausgeschieden nach Kriterium:")
    for k, v in drop.items():
        print(f"  {k}: {v}")

    res = pd.DataFrame(rows)
    if not res.empty:
        res["abstand_em"] = (res["price"] - res["strike"]) / res["em"]
        # Sicherheit zuerst: niedriges Delta und großer Abstand zählen mehr als Rendite
        res["score"] = ((1 - res["delta"]) * 0.5
                        + res["abstand_em"].clip(upper=2) / 2 * 0.3
                        + res["yield_pa"].clip(upper=0.4) / 0.4 * 0.2)
        res = res.sort_values("score", ascending=False)
        res = res.groupby("sector", sort=False).head(CFG["max_per_sector"]).head(CFG["top_n"])
    write_report(res, vix, above, max_delta, regime)
    print(f"Fertig: {len(res)} Trades in {OUT_DIR}/index.html")


if __name__ == "__main__":
    main()
