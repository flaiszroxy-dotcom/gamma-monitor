from datetime import date, datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# ───────────────────────── Oldal beállítás ─────────────────────────
st.set_page_config(page_title="XAUUSD GEX Monitor", page_icon="🥇", layout="wide")

GREEN, RED, BLUE, YELLOW, GREY = "#2ECC71", "#FF5C5C", "#4DA3FF", "#FFD24D", "#8A94A6"

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 1400px;}
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #161B26, #1B2130);
        border: 1px solid #262D3D; border-radius: 14px; padding: 16px 18px;
    }
    div[data-testid="stMetricLabel"] p {color: #8A94A6; font-size: 0.85rem;}
    div[data-testid="stMetricValue"] {font-size: 1.6rem;}
    .regime {border-radius: 14px; padding: 16px 20px; margin: 14px 0 6px 0;
             border: 1px solid #262D3D; line-height: 1.5;}
    .regime.pos {background: rgba(46,204,113,.10); border-color: rgba(46,204,113,.45);}
    .regime.neg {background: rgba(255,92,92,.10); border-color: rgba(255,92,92,.45);}
    .regime b {font-size: 1.05rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ───────────────────────── Számítások ─────────────────────────
def bs_gamma(S, K, T, r, sigma):
    """Black-Scholes gamma (vektorizált, broadcast-képes)."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return np.exp(-0.5 * d1**2) / np.sqrt(2 * np.pi) / (S * sigma * np.sqrt(T))


@st.cache_data(ttl=30, show_spinner=False)
def get_spots():
    gld = float(yf.Ticker("GLD").history(period="5d")["Close"].dropna().iloc[-1])
    xau = float(yf.Ticker("GC=F").history(period="5d")["Close"].dropna().iloc[-1])
    return gld, xau


@st.cache_data(ttl=300, show_spinner=False)
def get_expirations():
    return list(yf.Ticker("GLD").options)


@st.cache_data(ttl=60, show_spinner=False)
def get_chain(expiry):
    ch = yf.Ticker("GLD").option_chain(expiry)
    return ch.calls.copy(), ch.puts.copy()


def build_options(expiries):
    today = date.today()
    frames = []
    for e in expiries:
        calls, puts = get_chain(e)
        days = max((datetime.strptime(e, "%Y-%m-%d").date() - today).days, 0.5)
        for df, sign, typ in ((calls, 1, "Call"), (puts, -1, "Put")):
            d = df[["strike", "openInterest", "impliedVolatility"]].copy()
            d["sign"], d["type"], d["T"], d["expiry"] = sign, typ, days / 365.0, e
            frames.append(d)
    o = pd.concat(frames, ignore_index=True)
    o["openInterest"] = o["openInterest"].fillna(0)
    return o[(o["impliedVolatility"] > 0.01) & (o["openInterest"] > 0)].reset_index(drop=True)


def find_flip(levels, total, spot):
    s = np.sign(total)
    idx = np.where(s[:-1] * s[1:] < 0)[0]
    if len(idx) == 0:
        return None
    cands = [
        levels[i] - total[i] * (levels[i + 1] - levels[i]) / (total[i + 1] - total[i])
        for i in idx
    ]
    return min(cands, key=lambda x: abs(x - spot))


def style(fig, height=520):
    fig.update_layout(
        template="plotly_dark", height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=30, b=10),
        hoverlabel=dict(bgcolor="#1B2130", font_size=13),
        legend=dict(orientation="h", y=1.08, x=0),
        xaxis=dict(title="XAUUSD árszint (USD)", gridcolor="#1E2535", tickformat=",.0f"),
        yaxis=dict(gridcolor="#1E2535", zeroline=True, zerolinecolor="#4A5568"),
    )
    return fig


def vline(fig, x, color, text, pos, dash="dash", width=1.6):
    fig.add_vline(x=x, line_color=color, line_dash=dash, line_width=width,
                  annotation_text=text, annotation_position=pos,
                  annotation_font_color=color, annotation_font_size=12)


# ───────────────────────── Oldalsáv ─────────────────────────
st.sidebar.title("⚙️ Beállítások")

try:
    expirations = get_expirations()
except Exception as ex:
    st.error(f"Nem sikerült lekérni a lejáratokat: {ex}")
    st.stop()

if not expirations:
    st.warning("Nincsenek elérhető GLD opciós adatok.")
    st.stop()

today_str = date.today().strftime("%Y-%m-%d")
default_exp = next((e for e in expirations if e > today_str), expirations[0])

sel_expiries = st.sidebar.multiselect(
    "Lejáratok (több is választható, összegezve)", expirations, default=[default_exp]
)
range_pct = st.sidebar.slider("Megjelenített tartomány a spot körül (±%)", 2, 10, 5)
r_pct = st.sidebar.number_input("Kockázatmentes kamat (%)", 0.0, 10.0, 5.0, 0.25)
auto = st.sidebar.toggle("Automatikus frissítés", value=True)
every = st.sidebar.slider("Frissítés gyakorisága (mp)", 15, 120, 30, 5)
st.sidebar.caption("Adatforrás: Yahoo Finance (GLD opciók, GC=F). Az adatok késleltetettek lehetnek.")


# ───────────────────────── Dashboard ─────────────────────────
@st.fragment(run_every=every if auto else None)
def dashboard():
    st.title("🥇 XAUUSD Gamma Exposure Monitor")

    if not sel_expiries:
        st.info("Válassz legalább egy lejáratot az oldalsávban.")
        return

    try:
        s_gld, s_xau = get_spots()
        opts = build_options(sel_expiries)
    except Exception as ex:
        st.error(f"Hiba az adatok lekérésekor: {ex}")
        return

    if opts.empty:
        st.warning("Nincs használható opciós adat a kiválasztott lejáratokhoz.")
        return

    conv = s_xau / s_gld
    r = r_pct / 100.0

    # GEX opciónként: gamma * OI * 100 * S² * 1%  (dollár / 1%-os árfolyammozgás)
    K, T, IV = opts["strike"].values, opts["T"].values, opts["impliedVolatility"].values
    OI, SGN = opts["openInterest"].values, opts["sign"].values
    opts["gex"] = bs_gamma(s_gld, K, T, r, IV) * OI * 100 * s_gld**2 * 0.01 * SGN

    prof = opts.pivot_table(index="strike", columns="type", values="gex", aggfunc="sum").fillna(0)
    for c in ("Call", "Put"):
        if c not in prof:
            prof[c] = 0.0
    prof["Net"] = prof["Call"] + prof["Put"]
    prof["XAU"] = prof.index * conv
    prof = prof.reset_index()
    prof = prof[(prof["XAU"] >= s_xau * (1 - range_pct / 100)) & (prof["XAU"] <= s_xau * (1 + range_pct / 100))]

    if prof.empty:
        st.warning("A kiválasztott tartományban nincs adat.")
        return

    call_wall = prof.loc[prof["Call"].idxmax(), "XAU"]
    put_wall = prof.loc[prof["Put"].idxmin(), "XAU"]
    net_total = prof["Net"].sum()

    # Gamma flip: teljes GEX görbe különböző spot szinteken
    grid = np.linspace(s_gld * (1 - range_pct / 100), s_gld * (1 + range_pct / 100), 161)[:, None]
    mat = bs_gamma(grid, K[None, :], T[None, :], r, IV[None, :]) * (OI * 100 * SGN)[None, :]
    curve = (mat * grid**2 * 0.01).sum(axis=1)
    curve_xau = grid[:, 0] * conv
    flip = find_flip(curve_xau, curve, s_xau)

    # ── Fejléc kártyák ──
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("XAUUSD spot", f"${s_xau:,.2f}")
    c2.metric("Net GEX (tartományban)", f"{net_total / 1e6:+,.1f} M$ / 1%")
    c3.metric("Gamma flip szint", f"${flip:,.0f}" if flip else "–",
              f"{(flip - s_xau):+,.0f} $ a spottól" if flip else None, delta_color="off")
    c4.metric("Call fal  /  Put fal", f"${call_wall:,.0f}  /  ${put_wall:,.0f}")

    # ── Emberi nyelvű értelmezés ──
    positive = net_total > 0
    if positive:
        txt = ("<b>🟢 Pozitív gamma környezet</b><br>A market makerek az árral szemben kereskednek "
               "(esésnél vesznek, emelkedésnél adnak), ezért ez általában <b>fékezi a mozgásokat</b>: "
               "nyugodtabb, vissza-visszatérő piac.")
    else:
        txt = ("<b>🔴 Negatív gamma környezet</b><br>A market makerek az ár irányában fedeznek "
               "(esésnél adnak, emelkedésnél vesznek), ezért ez <b>felerősítheti a kilengéseket</b>: "
               "gyorsabb, volatilisebb mozgások lehetségesek.")
    if flip:
        txt += (f"<br>A spot a gamma flip szint <b>{'felett' if s_xau > flip else 'alatt'}</b> van "
                f"(${flip:,.0f}) – ennek átlépése jelezheti a környezet váltását.")
    st.markdown(f'<div class="regime {"pos" if positive else "neg"}">{txt}</div>', unsafe_allow_html=True)

    exp_label = ", ".join(sel_expiries)
    st.caption(f"Lejárat(ok): {exp_label}  •  Frissítve: {datetime.now():%Y-%m-%d %H:%M:%S}  •  "
               f"GLD → XAUUSD szorzó: {conv:.3f}")

    # ── Grafikonok ──
    spacing = np.median(np.diff(np.sort(prof["XAU"].values))) if len(prof) > 1 else 5
    bar_w = spacing * 0.8

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Net GEX", "⚖️ Call / Put bontás", "📈 GEX a spot függvényében", "🧾 Adatok"])

    with tab1:
        fig = go.Figure()
        fig.add_bar(x=prof["XAU"], y=prof["Net"] / 1e6, width=bar_w,
                    marker_color=np.where(prof["Net"] > 0, GREEN, RED), opacity=0.9,
                    customdata=prof["strike"],
                    hovertemplate="XAUUSD ≈ $%{x:,.0f}<br>GLD strike: $%{customdata:.0f}<br>"
                                  "Net GEX: %{y:+,.2f} M$<extra></extra>")
        vline(fig, s_xau, "#FFFFFF", f"Spot ${s_xau:,.0f}", "top", dash="solid", width=2.2)
        vline(fig, call_wall, BLUE, f"Call fal ${call_wall:,.0f}", "top right")
        vline(fig, put_wall, YELLOW, f"Put fal ${put_wall:,.0f}", "top left")
        if flip:
            vline(fig, flip, GREY, f"Flip ${flip:,.0f}", "bottom", dash="dot")
        fig.update_yaxes(title="Net GEX (millió USD / 1%)")
        st.plotly_chart(style(fig), use_container_width=True)

    with tab2:
        fig = go.Figure()
        fig.add_bar(x=prof["XAU"], y=prof["Call"] / 1e6, name="Call GEX", marker_color=GREEN, width=bar_w,
                    hovertemplate="$%{x:,.0f}<br>Call: %{y:,.2f} M$<extra></extra>")
        fig.add_bar(x=prof["XAU"], y=prof["Put"] / 1e6, name="Put GEX", marker_color=RED, width=bar_w,
                    hovertemplate="$%{x:,.0f}<br>Put: %{y:,.2f} M$<extra></extra>")
        fig.update_layout(barmode="relative")
        vline(fig, s_xau, "#FFFFFF", f"Spot ${s_xau:,.0f}", "top", dash="solid", width=2.2)
        fig.update_yaxes(title="GEX (millió USD / 1%)")
        st.plotly_chart(style(fig), use_container_width=True)

    with tab3:
        fig = go.Figure()
        fig.add_scatter(x=curve_xau, y=curve / 1e6, mode="lines", line=dict(color=BLUE, width=3),
                        fill="tozeroy", fillcolor="rgba(77,163,255,0.12)",
                        hovertemplate="Spot ≈ $%{x:,.0f}<br>Összes GEX: %{y:+,.2f} M$<extra></extra>")
        vline(fig, s_xau, "#FFFFFF", f"Spot ${s_xau:,.0f}", "top", dash="solid", width=2.2)
        if flip:
            vline(fig, flip, YELLOW, f"Gamma flip ${flip:,.0f}", "bottom")
        fig.update_yaxes(title="Összes GEX (millió USD / 1%)")
        st.plotly_chart(style(fig), use_container_width=True)
        st.caption("Azt mutatja, mekkora lenne az összesített gamma, ha az ár a vízszintes tengelyen látható szinten állna. "
                   "Ahol a görbe átlépi a nullát, ott vált a piac pozitív/negatív gamma környezetbe.")

    with tab4:
        show = prof[["XAU", "strike", "Call", "Put", "Net"]].copy()
        show.columns = ["XAUUSD szint", "GLD strike", "Call GEX", "Put GEX", "Net GEX"]
        for c in ("Call GEX", "Put GEX", "Net GEX"):
            show[c] = (show[c] / 1e6).round(3)
        st.dataframe(show.sort_values("XAUUSD szint", ascending=False), use_container_width=True, hide_index=True)
        st.caption("GEX értékek millió USD / 1%-os árfolyammozgás egységben.")


dashboard()
