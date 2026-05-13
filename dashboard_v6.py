"""
ESI — Earnings Signal Intelligence v6
Fixed: no overlapping tiles, score drivers below chart (full width),
bigger signal matrix cards, leaderboard filter dropdown.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="ESI · Earnings Signal Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

def css(s): st.markdown(f"<style>{s}</style>", unsafe_allow_html=True)
def h(s):   st.markdown(s, unsafe_allow_html=True)

css("@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');")
css(""":root{
  --bg:#141414; --card:#1C1C1C; --card2:#222;
  --b:rgba(255,255,255,0.06); --bhi:rgba(255,255,255,0.10);
  --t1:#F2F2F2; --t2:#888; --t3:#4A4A4A;
  --red:#E05C5C; --amb:#C9913A; --grn:#4CAF7D; --blu:#5B8FD4;
  --f:'Manrope',-apple-system,sans-serif;
  --m:'JetBrains Mono',monospace;
}""")
css("""
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
.stApp{background:var(--bg)!important;font-family:var(--f)}
.stApp>header{display:none!important}
.block-container{padding:0!important;max-width:100%!important}
div[data-testid="stVerticalBlock"]{gap:0!important}
div[data-testid="stHorizontalBlock"]{gap:0!important}
::-webkit-scrollbar{width:3px}
::-webkit-scrollbar-thumb{background:#2a2a2a;border-radius:3px}
""")
css("""
section[data-testid="stSidebar"]{background:var(--card)!important;border-right:1px solid var(--b)!important}
section[data-testid="stSidebar"]>div{padding:0!important}
""")
css("""
.stSelectbox label{font-family:var(--f)!important;font-size:10px!important;font-weight:600!important;
  color:var(--t3)!important;text-transform:uppercase!important;letter-spacing:0.06em!important}
.stSelectbox>div>div{background:var(--card2)!important;border:1px solid var(--bhi)!important;
  border-radius:10px!important;color:var(--t1)!important;font-family:var(--f)!important;
  font-size:13px!important;font-weight:500!important}
""")
css("""
.stExpander{background:var(--card)!important;border:1px solid var(--b)!important;
  border-radius:12px!important;overflow:hidden!important}
.stExpander summary{font-family:var(--f)!important;font-size:13px!important;
  font-weight:600!important;color:var(--t2)!important;padding:14px 20px!important}
""")
css("""
.stDataFrame{border-radius:12px!important;overflow:hidden!important}
.stDataFrame thead tr th{background:var(--card2)!important;color:var(--t3)!important;
  font-family:var(--m)!important;font-size:10px!important;letter-spacing:0.08em!important;
  text-transform:uppercase!important;border-bottom:1px solid var(--bhi)!important;border-right:none!important}
.stDataFrame tbody tr td{background:var(--card)!important;color:var(--t1)!important;
  font-family:var(--m)!important;font-size:12px!important;border-bottom:1px solid var(--b)!important;border-right:none!important}
.stDataFrame tbody tr:nth-child(even) td{background:rgba(255,255,255,0.015)!important}
.stDataFrame tbody tr:hover td{background:var(--card2)!important}
""")
css("""
.kpi-card{background:#1C1C1C;border:1px solid rgba(255,255,255,0.06);border-radius:14px;
  padding:24px 26px;height:130px;display:flex;flex-direction:column;justify-content:space-between}
.kpi-label{font-family:'Manrope',sans-serif;font-size:11px;font-weight:500;color:#4A4A4A}
.kpi-value{font-family:'Manrope',sans-serif;font-size:42px;font-weight:800;letter-spacing:-0.04em;line-height:1}
.kpi-sub{font-family:'JetBrains Mono',monospace;font-size:10px;color:#4A4A4A}
""")
css("""
.fc-card{background:#1C1C1C;border:1px solid rgba(255,255,255,0.06);border-radius:12px;padding:20px 22px;height:100%}
.fc-label{font-family:'Manrope',sans-serif;font-size:10px;font-weight:600;color:#4A4A4A;
  text-transform:uppercase;letter-spacing:0.05em;margin-bottom:10px}
.fc-value{font-family:'Manrope',sans-serif;font-size:22px;font-weight:800;letter-spacing:-0.02em;line-height:1}
.fc-desc{font-family:'JetBrains Mono',monospace;font-size:9px;color:#4A4A4A;margin-top:6px;line-height:1.5}
.fc-bar-bg{height:2px;background:#2a2a2a;border-radius:2px;margin-top:12px;overflow:hidden}
.fc-bar{height:100%;border-radius:2px;opacity:0.85}
""")
css("""
.shap-row{display:flex;align-items:center;gap:12px;padding:10px 0;
  border-bottom:1px solid rgba(255,255,255,0.04)}
.shap-row:last-child{border-bottom:none}
.shap-idx{font-family:'JetBrains Mono',monospace;font-size:9px;color:#4A4A4A;width:18px;flex-shrink:0}
.shap-name{font-family:'Manrope',sans-serif;font-size:12px;color:#888;flex:1;font-weight:500}
.shap-track{width:120px;height:2px;background:#2a2a2a;border-radius:2px;overflow:hidden;flex-shrink:0}
.shap-fill{height:100%;border-radius:2px}
.shap-val{font-family:'JetBrains Mono',monospace;font-size:11px;width:58px;text-align:right;font-weight:500;flex-shrink:0}
""")
css("""
.lb-row{display:flex;align-items:center;gap:14px;padding:13px 0;
  border-bottom:1px solid rgba(255,255,255,0.04)}
.lb-row:last-child{border-bottom:none}
.lb-rank{font-family:'JetBrains Mono',monospace;font-size:11px;font-weight:600;
  width:22px;text-align:center;flex-shrink:0}
.lb-ticker{font-family:'Manrope',sans-serif;font-size:14px;font-weight:700;
  color:#F2F2F2;width:56px;flex-shrink:0}
.lb-period{font-family:'JetBrains Mono',monospace;font-size:11px;color:#4A4A4A;flex:1}
.lb-heat{flex:2;height:3px;background:#222;border-radius:2px;overflow:hidden}
.lb-heat-fill{height:100%;border-radius:2px;opacity:0.6}
.lb-score{font-family:'Manrope',sans-serif;font-size:16px;font-weight:800;
  width:38px;text-align:right;flex-shrink:0;letter-spacing:-0.02em}
.lb-badge{font-family:'Manrope',sans-serif;font-size:10px;font-weight:600;
  padding:3px 10px;border-radius:100px;width:62px;text-align:center;flex-shrink:0}
""")

# ── Constants ─────────────────────────────────────────────────────────────────
FCOLS = ["uncertainty_ratio","hedging_ratio","negation_ratio","first_person_ratio",
         "positive_ratio","negative_ratio","uncertainty_delta","hedging_delta",
         "sentiment_delta","mean_qa_similarity","evasion_rate","cfo_evasion_rate",
         "ceo_evasion_rate","prepared_sentiment","qa_sentiment","sentiment_gap"]
FLABELS = {
    "uncertainty_ratio":"Uncertainty Language","hedging_ratio":"Hedging Frequency",
    "negation_ratio":"Negation Density","first_person_ratio":"First-Person Avoidance",
    "positive_ratio":"Positive Signal","negative_ratio":"Negative Signal",
    "uncertainty_delta":"Uncertainty Drift","hedging_delta":"Hedging Drift",
    "sentiment_delta":"Sentiment Drift","mean_qa_similarity":"Q&A Consistency",
    "evasion_rate":"Analyst Evasion Rate","cfo_evasion_rate":"CFO Evasion Rate",
    "ceo_evasion_rate":"CEO Evasion Rate","prepared_sentiment":"Prepared Sentiment",
    "qa_sentiment":"Q&A Sentiment","sentiment_gap":"Sentiment Divergence",
}
FDESC = {
    "uncertainty_ratio":"Uncertain words per 100 (may, believe, approximately)",
    "hedging_ratio":"Softening language that reduces statement commitment",
    "negation_ratio":"Denial and negation density — signals concealment risk",
    "first_person_ratio":"Personal accountability language per sentence",
    "positive_ratio":"Positive financial language in prepared remarks",
    "negative_ratio":"Negative financial language and loss terminology",
    "uncertainty_delta":"Deviation from 8-quarter company baseline",
    "hedging_delta":"Change in hedging vs historical average",
    "sentiment_delta":"Directional sentiment shift vs baseline",
    "mean_qa_similarity":"Semantic similarity: analyst questions vs executive answers",
    "evasion_rate":"Fraction of analyst questions deflected",
    "cfo_evasion_rate":"CFO-specific question deflection in Q&A",
    "ceo_evasion_rate":"CEO-specific question deflection in Q&A",
    "prepared_sentiment":"FinBERT score on prepared remarks section",
    "qa_sentiment":"FinBERT score on Q&A responses",
    "sentiment_gap":"Management optimism vs financial reality gap",
}
RDIR = {"uncertainty_ratio":+1,"hedging_ratio":+1,"negation_ratio":+1,"first_person_ratio":-1,
        "positive_ratio":-1,"negative_ratio":+1,"uncertainty_delta":+1,"hedging_delta":+1,
        "sentiment_delta":-1,"mean_qa_similarity":-1,"evasion_rate":+1,"cfo_evasion_rate":+1,
        "ceo_evasion_rate":+1,"prepared_sentiment":-1,"qa_sentiment":-1,"sentiment_gap":+1}

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        import xgboost as xgb
        m = xgb.XGBClassifier(); m.load_model("models/deception_model.json"); return m
    except: return None

@st.cache_data(ttl=300)
def load_data():
    try:
        from database import SessionLocal, FeatureVector, Transcript
        db = SessionLocal()
        rows = db.query(FeatureVector).all()
        data = []
        for r in rows:
            t = db.query(Transcript).filter_by(id=r.transcript_id).first()
            row = {c: getattr(r,c) for c in FCOLS}
            row.update({"ticker":r.ticker,"fiscal_year":r.fiscal_year,
                        "quarter":r.quarter,"period":f"Q{r.quarter} {r.fiscal_year}",
                        "label":r.label,"call_date":t.call_date if t else None})
            data.append(row)
        db.close()
        return pd.DataFrame(data)
    except: return pd.DataFrame()

def get_score(row_dict, model):
    if model is None: return 50.0,"MEDIUM",[]
    try:
        import shap
        X = np.array([[float(row_dict.get(c,0) or 0) for c in FCOLS]])
        prob = float(model.predict_proba(X)[0][1])
        s  = round(prob*100,1)
        lv = "HIGH" if s>65 else "MEDIUM" if s>40 else "LOW"
        exp= shap.TreeExplainer(model)
        sv = exp.shap_values(X)[0]
        drv= sorted([{"f":f,"label":FLABELS[f],"v":float(sh)}
                     for f,sh in zip(FCOLS,sv)],key=lambda x:abs(x["v"]),reverse=True)
        return s, lv, drv[:8]
    except: return 50.0,"MEDIUM",[]

def lc(lv): return {"HIGH":"#E05C5C","MEDIUM":"#C9913A","LOW":"#4CAF7D"}.get(lv,"#888")
def la(lv): return {"HIGH":"rgba(224,92,92,0.10)","MEDIUM":"rgba(201,145,58,0.10)",
                    "LOW":"rgba(76,175,125,0.10)"}.get(lv,"rgba(91,143,212,0.10)")
def divider(): h('<hr style="border:none;border-top:1px solid rgba(255,255,255,0.06);margin:0">')
def section_hdr(title, caption=""):
    cap = f'<span style="font-family:JetBrains Mono,monospace;font-size:10px;color:#4A4A4A">{caption}</span>' if caption else ""
    h(f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:20px">'
      f'<span style="font-family:Manrope,sans-serif;font-size:14px;font-weight:700;color:#F2F2F2;letter-spacing:-0.01em">{title}</span>'
      f'{cap}</div>')

# ── Load ──────────────────────────────────────────────────────────────────────
model  = load_model()
df_all = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    h('<div style="padding:24px 20px 16px;border-bottom:1px solid rgba(255,255,255,0.06)">'
      '<div style="display:flex;align-items:center;gap:10px">'
      '<div style="width:30px;height:30px;background:#222;border-radius:8px;border:1px solid rgba(255,255,255,0.08);'
      'display:flex;align-items:center;justify-content:center;flex-shrink:0">'
      '<svg width="14" height="14" viewBox="0 0 16 16" fill="none">'
      '<path d="M2 12L5 7.5L8 10L11 5.5L14 8" stroke="#F2F2F2" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>'
      '<circle cx="14" cy="8" r="1.4" fill="#F2F2F2"/></svg></div>'
      '<div><div style="font-family:Manrope,sans-serif;font-size:13px;font-weight:700;color:#F2F2F2">ESI Platform</div>'
      '<div style="font-family:JetBrains Mono,monospace;font-size:9px;color:#4A4A4A;margin-top:1px">Earnings Signal Intelligence</div>'
      '</div></div></div>')

    tickers = sorted(df_all["ticker"].unique().tolist()) if not df_all.empty else ["—"]
    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
    selected = st.selectbox("Company", tickers, key="co")

    feat_opts = [FLABELS[c] for c in FCOLS]
    feat_rev  = {v:k for k,v in FLABELS.items()}
    st.markdown('<div style="height:4px"></div>', unsafe_allow_html=True)
    sel_feat_label = st.selectbox("Signal View", feat_opts, key="fv")
    sel_feat = feat_rev[sel_feat_label]

    h(f'<div style="margin:6px 20px 14px;padding:11px 13px;background:#1C1C1C;border-radius:9px;border:1px solid rgba(255,255,255,0.05)">'
      f'<div style="font-family:JetBrains Mono,monospace;font-size:9px;color:#4A4A4A;line-height:1.6">{FDESC.get(sel_feat,"")}</div></div>')

    h('<hr style="border:none;border-top:1px solid rgba(255,255,255,0.05);margin:0 20px">')
    h('<div style="padding:14px 20px 12px">'
      '<div style="font-family:Manrope,sans-serif;font-size:10px;font-weight:600;color:#4A4A4A;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:10px">Risk Scale</div>'
      '<div style="display:flex;align-items:center;gap:8px;margin-bottom:7px"><div style="width:5px;height:5px;border-radius:50%;background:#E05C5C;flex-shrink:0"></div><span style="font-family:Manrope,sans-serif;font-size:12px;color:#666;flex:1">High Risk</span><span style="font-family:JetBrains Mono,monospace;font-size:10px;color:#4A4A4A">&gt; 65</span></div>'
      '<div style="display:flex;align-items:center;gap:8px;margin-bottom:7px"><div style="width:5px;height:5px;border-radius:50%;background:#C9913A;flex-shrink:0"></div><span style="font-family:Manrope,sans-serif;font-size:12px;color:#666;flex:1">Caution</span><span style="font-family:JetBrains Mono,monospace;font-size:10px;color:#4A4A4A">40–65</span></div>'
      '<div style="display:flex;align-items:center;gap:8px"><div style="width:5px;height:5px;border-radius:50%;background:#4CAF7D;flex-shrink:0"></div><span style="font-family:Manrope,sans-serif;font-size:12px;color:#666;flex:1">Low Risk</span><span style="font-family:JetBrains Mono,monospace;font-size:10px;color:#4A4A4A">&lt; 40</span></div>'
      '</div>')

    h('<hr style="border:none;border-top:1px solid rgba(255,255,255,0.05);margin:0 20px">')
    n_co = len(df_all["ticker"].unique()) if not df_all.empty else 0
    n_q  = len(df_all) if not df_all.empty else 0
    mc   = "#4CAF7D" if model else "#E05C5C"
    h(f'<div style="padding:14px 20px 20px">'
      f'<div style="display:flex;justify-content:space-between;padding:4px 0"><span style="font-family:Manrope,sans-serif;font-size:12px;color:#4A4A4A">Model</span><span style="font-family:Manrope,sans-serif;font-size:12px;color:{mc};font-weight:600">{"Active" if model else "Offline"}</span></div>'
      f'<div style="display:flex;justify-content:space-between;padding:4px 0"><span style="font-family:Manrope,sans-serif;font-size:12px;color:#4A4A4A">Coverage</span><span style="font-family:Manrope,sans-serif;font-size:12px;color:#666;font-weight:500">{n_co} companies</span></div>'
      f'<div style="display:flex;justify-content:space-between;padding:4px 0"><span style="font-family:Manrope,sans-serif;font-size:12px;color:#4A4A4A">Data points</span><span style="font-family:Manrope,sans-serif;font-size:12px;color:#666;font-weight:500">{n_q} quarters</span></div>'
      f'</div>')

# ── Guard ─────────────────────────────────────────────────────────────────────
if df_all.empty:
    h('<div style="display:flex;align-items:center;justify-content:center;height:80vh;flex-direction:column;gap:10px">'
      '<div style="font-family:Manrope,sans-serif;font-size:18px;color:#666;font-weight:600">No data available</div>'
      '<div style="font-family:JetBrains Mono,monospace;font-size:11px;color:#4A4A4A">python main.py --all --tickers MSFT GOOGL</div></div>')
    st.stop()

# ── Ticker data ───────────────────────────────────────────────────────────────
tdf = df_all[df_all["ticker"]==selected].copy().sort_values(["fiscal_year","quarter"])
all_periods = tdf["period"].tolist()
scores_list,levels_list = [],[]
for _,row in tdf.iterrows():
    s,lv,_ = get_score(row.to_dict(),model)
    scores_list.append(s); levels_list.append(lv)
tdf["risk_score"] = scores_list
tdf["risk_level"]  = levels_list

latest   = tdf.iloc[-1] if not tdf.empty else None
prev     = tdf.iloc[-2] if len(tdf)>=2 else None
lat_s,lat_lv,lat_drv = get_score(latest.to_dict(),model) if latest is not None else (50.0,"MEDIUM",[])
prev_s   = get_score(prev.to_dict(),model)[0] if prev is not None else None
hist_avg = tdf["risk_score"].mean()
peak_s   = tdf["risk_score"].max()
peak_p   = tdf.loc[tdf["risk_score"].idxmax(),"period"] if not tdf.empty else "—"
flagged  = (tdf["risk_score"]>65).sum()
score_delta = round(lat_s-prev_s,1) if prev_s is not None else None

# ── Page header ───────────────────────────────────────────────────────────────
c_left, c_right = st.columns([2,1])
with c_left:
    h(f'<div style="padding:32px 40px 24px">'
      f'<div style="font-family:JetBrains Mono,monospace;font-size:10px;color:#4A4A4A;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:10px">Deception Risk Profile</div>'
      f'<div style="font-family:Manrope,sans-serif;font-size:36px;font-weight:800;color:#F2F2F2;letter-spacing:-0.03em;line-height:1">{selected}</div>'
      f'<div style="font-family:Manrope,sans-serif;font-size:12px;color:#4A4A4A;margin-top:8px">{len(tdf)} quarters &nbsp;·&nbsp; Loughran-McDonald &nbsp;·&nbsp; FinBERT &nbsp;·&nbsp; sentence-BERT</div>'
      f'</div>')
with c_right:
    h(f'<div style="padding:32px 40px 24px;display:flex;justify-content:flex-end;align-items:center">'
      f'<div style="padding:11px 20px;border-radius:100px;background:{la(lat_lv)};border:1px solid {lc(lat_lv)}33;display:inline-flex;align-items:center;gap:8px">'
      f'<div style="width:6px;height:6px;border-radius:50%;background:{lc(lat_lv)}"></div>'
      f'<div style="font-family:Manrope,sans-serif;font-size:12px;font-weight:700;color:{lc(lat_lv)}">{lat_lv} RISK</div>'
      f'<div style="width:1px;height:12px;background:{lc(lat_lv)}44"></div>'
      f'<div style="font-family:JetBrains Mono,monospace;font-size:12px;color:{lc(lat_lv)};font-weight:500">{lat_s:.0f}/100</div>'
      f'</div></div>')
divider()

# ── KPI strip — proper spacing with padding wrapper ───────────────────────────
h('<div style="padding:24px 40px">')
k1,k2,k3,k4 = st.columns(4, gap="medium")

sd_col = "#4CAF7D" if score_delta and score_delta<0 else "#E05C5C"
sd_ar  = ("↓" if score_delta<0 else "↑") if score_delta else ""
sd_str = f' <span style="font-size:14px;color:{sd_col}">{sd_ar}{abs(score_delta) if score_delta else ""}</span>' if score_delta else ""

with k1:
    h(f'<div class="kpi-card"><div class="kpi-label">Current Risk Score</div>'
      f'<div class="kpi-value" style="color:{lc(lat_lv)}">{lat_s:.0f}{sd_str}</div>'
      f'<div class="kpi-sub">{lat_lv} &nbsp;·&nbsp; {latest["period"] if latest is not None else "—"}</div></div>')
with k2:
    h(f'<div class="kpi-card"><div class="kpi-label">Historical Average</div>'
      f'<div class="kpi-value" style="color:#F2F2F2">{hist_avg:.0f}</div>'
      f'<div class="kpi-sub">across {len(tdf)} quarters</div></div>')
with k3:
    fc3 = "#E05C5C" if flagged>2 else "#C9913A" if flagged>0 else "#4CAF7D"
    h(f'<div class="kpi-card"><div class="kpi-label">High-Risk Quarters</div>'
      f'<div class="kpi-value" style="color:{fc3}">{flagged}</div>'
      f'<div class="kpi-sub">of {len(tdf)} total &nbsp;·&nbsp; score &gt; 65</div></div>')
with k4:
    fc4 = "#E05C5C" if peak_s>65 else "#C9913A"
    h(f'<div class="kpi-card"><div class="kpi-label">Peak Score</div>'
      f'<div class="kpi-value" style="color:{fc4}">{peak_s:.0f}</div>'
      f'<div class="kpi-sub">{peak_p}</div></div>')
h('</div>')
divider()

# ── Risk trajectory (full width) ──────────────────────────────────────────────
h('<div style="padding:28px 40px 0">')
section_hdr("Risk Trajectory", "Y-axis auto-zoomed · dot color = risk level")

if len(tdf)>=2:
    periods = tdf["period"].tolist()
    rscores = tdf["risk_score"].tolist()
    rlevels = tdf["risk_level"].tolist()
    mn,mx   = min(rscores),max(rscores)
    pad     = max((mx-mn)*0.4,6)
    ylo,yhi = max(0,mn-pad),min(100,mx+pad)
    dot_colors = [lc(lv) for lv in rlevels]
    fig = go.Figure()
    if ylo<65<yhi:
        fig.add_hline(y=65,line_color="rgba(224,92,92,0.15)",line_width=1,line_dash="dot",
                      annotation_text="65",annotation_font=dict(size=9,color="rgba(224,92,92,0.4)"),annotation_position="right")
    if ylo<40<yhi:
        fig.add_hline(y=40,line_color="rgba(201,145,58,0.15)",line_width=1,line_dash="dot",
                      annotation_text="40",annotation_font=dict(size=9,color="rgba(201,145,58,0.4)"),annotation_position="right")
    fig.add_trace(go.Scatter(x=periods,y=rscores,fill='tozeroy',
        fillcolor='rgba(91,143,212,0.03)',line=dict(width=0),showlegend=False,hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=periods,y=rscores,mode='lines+markers',
        line=dict(color='rgba(242,242,242,0.5)',width=1.5,shape='spline',smoothing=0.5),
        marker=dict(color=dot_colors,size=9,line=dict(color='#141414',width=2)),
        customdata=rlevels,
        hovertemplate='<b>%{x}</b><br>Score: <b>%{y:.1f}</b><br>%{customdata}<extra></extra>'))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0,r=44,t=4,b=0),height=200,
        xaxis=dict(showgrid=False,zeroline=False,tickfont=dict(family='JetBrains Mono',size=10,color='#4A4A4A')),
        yaxis=dict(showgrid=True,gridcolor='rgba(255,255,255,0.04)',zeroline=False,range=[ylo,yhi],
                   tickfont=dict(family='JetBrains Mono',size=10,color='#4A4A4A')),
        hoverlabel=dict(bgcolor='#222',font=dict(family='JetBrains Mono',size=12,color='#F2F2F2'),bordercolor='#333'))
    st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
else:
    h('<div style="padding:28px;text-align:center;background:#1C1C1C;border-radius:12px;border:1px solid rgba(255,255,255,0.05)">'
      '<span style="font-family:JetBrains Mono,monospace;font-size:11px;color:#4A4A4A">Insufficient data — minimum 2 quarters required</span></div>')

h('<div style="height:8px"></div></div>')
divider()

# ── Signal view (left 60%) + Score Drivers (right 40%) ───────────────────────
h('<div style="padding:28px 40px 0">')
col_chart, col_shap = st.columns([6,4], gap="large")

with col_chart:
    h(f'<div style="margin-bottom:16px">'
    f'<div style="font-family:Manrope,sans-serif;font-size:14px;font-weight:700;color:#F2F2F2;letter-spacing:-0.01em">{sel_feat_label}</div>'
    f'<div style="font-family:JetBrains Mono,monospace;font-size:10px;color:#4A4A4A;margin-top:4px">Red = above avg in concerning direction</div>'
    f'</div>')
    feat_vals = tdf[sel_feat].fillna(0).tolist()
    fperiods  = tdf["period"].tolist()
    if not feat_vals or all(v==0 for v in feat_vals):
        h('<div style="padding:28px;text-align:center;background:#1C1C1C;border-radius:12px;border:1px solid rgba(255,255,255,0.05)">'
          '<span style="font-family:JetBrains Mono,monospace;font-size:10px;color:#4A4A4A">Feature unavailable — incomplete quarter coverage</span></div>')
    else:
        direction = RDIR.get(sel_feat,+1)
        mean_val  = np.mean(feat_vals)
        bar_colors = []
        for v in feat_vals:
            above = v > mean_val
            if direction==+1: c="rgba(224,92,92,0.70)" if above else "rgba(76,175,125,0.45)"
            else:              c="rgba(76,175,125,0.55)" if above else "rgba(224,92,92,0.40)"
            bar_colors.append(c)
        fig2 = go.Figure()
        fig2.add_hline(y=mean_val,line_color="rgba(91,143,212,0.25)",line_width=1,line_dash="dot",
                       annotation_text="avg",annotation_font=dict(size=9,color="rgba(91,143,212,0.5)"),annotation_position="right")
        fig2.add_trace(go.Bar(x=fperiods,y=feat_vals,marker=dict(color=bar_colors,line=dict(width=0)),
            hovertemplate=f'<b>%{{x}}</b><br>{sel_feat_label}: <b>%{{y:.4f}}</b><extra></extra>'))
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0,r=44,t=4,b=0),height=220,bargap=0.28,
            xaxis=dict(showgrid=False,zeroline=False,tickfont=dict(family='JetBrains Mono',size=10,color='#4A4A4A')),
            yaxis=dict(showgrid=True,gridcolor='rgba(255,255,255,0.04)',zeroline=False,
                       tickfont=dict(family='JetBrains Mono',size=10,color='#4A4A4A')),
            hoverlabel=dict(bgcolor='#222',font=dict(family='JetBrains Mono',size=12,color='#F2F2F2'),bordercolor='#333'))
        st.plotly_chart(fig2,use_container_width=True,config={"displayModeBar":False})

with col_shap:
    h('<div style="margin-bottom:16px">'
      '<div style="font-family:Manrope,sans-serif;font-size:14px;font-weight:700;color:#F2F2F2;letter-spacing:-0.01em">Score Drivers</div>'
      '<div style="font-family:JetBrains Mono,monospace;font-size:10px;color:#4A4A4A;margin-top:4px">SHAP attribution</div>'
      '</div>')
    if all_periods:
        shap_q   = st.selectbox("Quarter", options=list(reversed(all_periods)), index=0, key="shap_q")
        shap_row = tdf[tdf["period"]==shap_q].iloc[0] if shap_q in tdf["period"].values else latest
        shap_s,shap_lv,shap_drv = get_score(shap_row.to_dict(),model) if shap_row is not None else (50.0,"MEDIUM",[])

        h(f'<div style="display:inline-flex;align-items:center;gap:8px;margin-bottom:14px">'
          f'<div style="padding:3px 10px;border-radius:100px;background:{la(shap_lv)};'
          f'font-family:Manrope,sans-serif;font-size:10px;font-weight:700;color:{lc(shap_lv)}">{shap_lv}</div>'
          f'<div style="font-family:JetBrains Mono,monospace;font-size:13px;color:{lc(shap_lv)};font-weight:500">{shap_s:.1f}/100</div>'
          f'</div>')

        if shap_drv:
            max_sv = max(abs(d["v"]) for d in shap_drv) or 1
            for i,d in enumerate(shap_drv):
                pct = abs(d["v"])/max_sv*100
                col = "#E05C5C" if d["v"]>0 else "#4CAF7D"
                ar  = "+" if d["v"]>0 else "−"
                h(f'<div class="shap-row">'
                  f'<span class="shap-idx">{i+1:02d}</span>'
                  f'<span class="shap-name">{d["label"]}</span>'
                  f'<div class="shap-track"><div class="shap-fill" style="width:{pct:.0f}%;background:{col}"></div></div>'
                  f'<span class="shap-val" style="color:{col}">{ar}{abs(d["v"]):.3f}</span>'
                  f'</div>')
        else:
            h('<div style="font-family:JetBrains Mono,monospace;font-size:11px;color:#4A4A4A;padding:16px 0">Train model to enable SHAP.</div>')

h('<div style="height:8px"></div></div>')
divider()

# ── Full signal matrix — quarter selector + 4 per row ─────────────────────────
h('<div style="padding:28px 40px">')

mx_hdr, mx_sel = st.columns([3, 1], gap="small")
with mx_hdr:
    h('<div style="margin-bottom:20px">'
      '<div style="font-family:Manrope,sans-serif;font-size:14px;font-weight:700;color:#F2F2F2;letter-spacing:-0.01em">Full Signal Matrix</div>'
      '<div style="font-family:JetBrains Mono,monospace;font-size:10px;color:#4A4A4A;margin-top:4px">16 features · red = concerning · green = healthy · delta vs prev quarter</div>'
      '</div>')
with mx_sel:
    mx_quarter = st.selectbox("Quarter", options=list(reversed(all_periods)), index=0, key="mx_q")

mx_row  = tdf[tdf["period"]==mx_quarter].iloc[0] if mx_quarter in tdf["period"].values else latest
# Get previous quarter for delta
mx_idx  = tdf[tdf["period"]==mx_quarter].index[0] if mx_quarter in tdf["period"].values else None
mx_prev_row = None
if mx_idx is not None:
    pos = tdf.index.get_loc(mx_idx)
    if pos > 0:
        mx_prev_row = tdf.iloc[pos - 1]

if mx_row is not None:
    all_vals  = [float(mx_row.get(c,0) or 0) for c in FCOLS]
    prev_vals = [float(mx_prev_row.get(c,0) or 0) for c in FCOLS] if mx_prev_row is not None else [None]*len(FCOLS)
    max_val   = max(abs(v) for v in all_vals) or 1

    for row_i in range(4):
        cols = st.columns(4, gap="small")
        for col_i in range(4):
            idx = row_i*4 + col_i
            if idx >= len(FCOLS): break
            col_name = FCOLS[idx]
            val      = all_vals[idx]
            pv       = prev_vals[idx]
            label    = FLABELS[col_name]
            desc     = FDESC[col_name]
            pct      = abs(val)/max_val*100
            d        = RDIR.get(col_name,+1)
            high     = (d==+1 and pct>60) or (d==-1 and pct<20)
            med      = (d==+1 and pct>30) or (d==-1 and pct<50)
            bc       = "#E05C5C" if high else "#C9913A" if med else "#4CAF7D"

            delta_str = ""
            if pv is not None:
                diff = val - pv
                if abs(diff) > 0.001:
                    dc = "#E05C5C" if (diff>0)==(d==+1) else "#4CAF7D"
                    ar = "+" if diff>0 else "−"
                    delta_str = (f'<span style="font-family:JetBrains Mono,monospace;'
                                 f'font-size:9px;color:{dc};margin-left:6px">{ar}{abs(diff):.3f}</span>')

            with cols[col_i]:
                h(f'<div class="fc-card">'
                  f'<div class="fc-label">{label}</div>'
                  f'<div style="display:flex;align-items:baseline">'
                  f'<div class="fc-value" style="color:{bc}">{val:.4f}</div>{delta_str}</div>'
                  f'<div class="fc-desc">{desc}</div>'
                  f'<div class="fc-bar-bg"><div class="fc-bar" style="width:{pct:.1f}%;background:{bc}"></div></div>'
                  f'</div>')
        if row_i < 3:
            h('<div style="height:10px"></div>')

h('</div>')
divider()

# ── Leaderboard with YEAR filter ──────────────────────────────────────────────
h('<div style="padding:28px 40px">')

# Build full leaderboard — all quarters for all companies, not just latest
all_years = sorted(df_all["fiscal_year"].unique().tolist(), reverse=True)
year_options = ["All Years (latest quarter)"] + [str(y) for y in all_years]

lb_hdr_col, lb_filter_col = st.columns([3,1], gap="small")
with lb_hdr_col:
    h('<div style="margin-bottom:16px">'
      '<div style="font-family:Manrope,sans-serif;font-size:14px;font-weight:700;color:#F2F2F2;letter-spacing:-0.01em">Risk Leaderboard</div>'
      '<div style="font-family:JetBrains Mono,monospace;font-size:10px;color:#4A4A4A;margin-top:4px">Ranked by deception score · select year to compare all companies in that year</div>'
      '</div>')
with lb_filter_col:
    lb_year = st.selectbox("Year", year_options, key="lb_year")

# Build board based on year selection
board = []
if lb_year == "All Years (latest quarter)":
    # Show most recent quarter per company
    for ticker, grp in df_all.groupby("ticker"):
        grp = grp.sort_values(["fiscal_year","quarter"])
        row = grp.iloc[-1]
        s,lv,_ = get_score(row.to_dict(), model)
        board.append({"ticker":ticker,"period":row["period"],"score":s,"level":lv,"year":row["fiscal_year"]})
else:
    # Show all quarters for selected year, all companies
    year_int = int(lb_year)
    year_df  = df_all[df_all["fiscal_year"]==year_int]
    for _, row in year_df.iterrows():
        s,lv,_ = get_score(row.to_dict(), model)
        board.append({"ticker":row["ticker"],"period":row["period"],"score":s,"level":lv,"year":row["fiscal_year"]})

board.sort(key=lambda x:x["score"], reverse=True)

if not board:
    h('<div style="padding:20px;text-align:center;background:#1C1C1C;border-radius:12px;border:1px solid rgba(255,255,255,0.05)">'
      '<span style="font-family:JetBrains Mono,monospace;font-size:11px;color:#4A4A4A">No data for selected year</span></div>')
else:
    for i,e in enumerate(board):
        rank  = i+1
        col   = lc(e["level"])
        rkcol = "#E3B341" if rank==1 else "#999" if rank==2 else "#7A6040" if rank==3 else "#4A4A4A"
        is_sel= e["ticker"]==selected
        bg    = "background:rgba(255,255,255,0.03);" if is_sel else ""
        bl    = f"border-left:2px solid {col};" if is_sel else "border-left:2px solid transparent;"
        h(f'<div class="lb-row" style="{bg}{bl}">'
          f'<span class="lb-rank" style="color:{rkcol}">{rank:02d}</span>'
          f'<span class="lb-ticker" style="color:{"#F2F2F2" if is_sel else "#CCC"}">{e["ticker"]}</span>'
          f'<span class="lb-period">{e["period"]}</span>'
          f'<div class="lb-heat"><div class="lb-heat-fill" style="width:{e["score"]}%;background:{col}"></div></div>'
          f'<span class="lb-score" style="color:{col}">{e["score"]:.0f}</span>'
          f'<span class="lb-badge" style="background:{la(e["level"])};color:{col}">{e["level"]}</span>'
          f'</div>')

h('</div>')
divider()

# ── Full history ───────────────────────────────────────────────────────────────
h('<div style="padding:8px 40px 0">')
with st.expander("Full Quarter History — All 16 Features"):
    disp = tdf[["period","risk_score","risk_level"]+FCOLS+["label"]].copy()
    disp.columns = ["Period","Risk Score","Level"]+[FLABELS[c] for c in FCOLS]+["Label"]
    disp = disp.sort_values("Period",ascending=False)
    st.dataframe(disp.round(4),hide_index=True,use_container_width=True)
h('</div>')
divider()

# ── Footer ─────────────────────────────────────────────────────────────────────
h('<div style="padding:16px 40px;display:flex;justify-content:space-between;align-items:center">'
  '<span style="font-family:JetBrains Mono,monospace;font-size:10px;color:#4A4A4A">ESI Platform v6 · Earnings Signal Intelligence · Research use only</span>'
  '<span style="font-family:JetBrains Mono,monospace;font-size:10px;color:#4A4A4A">Loughran-McDonald · FinBERT · sentence-BERT · XGBoost + SHAP</span>'
  '</div>')
