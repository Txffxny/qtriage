"""qtriage dashboard.

Design principle: ranking and certification are separate capabilities.

Every host-window has a conformal p-value, so the ranked queue always exists.
Benjamini-Hochberg does not produce that ranking - it draws a line on it, with
a false-discovery bound attached. Most tools show a ranked list and imply a
guarantee they do not have. This one draws the line explicitly and states what
is below it.
"""
import json
import pathlib

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="qtriage", layout="wide")

HERE = pathlib.Path(__file__).resolve().parent
ALERTS = HERE / "alerts.parquet"
META = HERE / "alerts_meta.json"


@st.cache_data
def load():
    if not ALERTS.exists():
        return None, None
    with open(META) as fh:
        meta = json.load(fh)
    return pd.read_parquet(ALERTS), meta


df, meta = load()

if df is None:
    st.error("alerts.parquet not found next to app.py.")
    st.stop()

# ---------------------------------------------------------------- sidebar
st.sidebar.title("qtriage")
st.sidebar.caption("Statistically certified alert triage")

day = st.sidebar.selectbox("Day", sorted(df["day"].unique()))
q = st.sidebar.select_slider("Target FDR (q)", options=[0.01, 0.05, 0.10, 0.20, 0.30],
                             value=0.05)
qcol = "certified_q" + str(q).replace("0.", "")

st.sidebar.markdown("---")
st.sidebar.caption(
    "Baseline: " + meta["calibration_file"].replace(".csv", "")
    + f"\n\n{meta['n_cal']:,} calibration windows from {meta['n_cal_hosts']} hosts"
    + f"\n\nWindow: {meta['window']} · unit: {meta['unit']}"
    + f"\n\np-value floor: {meta['floor']:.2e}"
)

d = df[df["day"] == day].copy()
d = d.sort_values("p_value").reset_index(drop=True)

certified = d[d[qcol]]
n_cert = len(certified)
tp = int(certified["is_attack"].sum())
fp = n_cert - tp

# ---------------------------------------------------------------- header
st.title(f"{day} · alert triage")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Host-windows observed", f"{len(d):,}")
c2.metric(f"Certified at q={q}", f"{n_cert:,}",
          delta=f"-{100*(1-n_cert/max(len(d),1)):.1f}% volume")
c3.metric("True / false", f"{tp} / {fp}")
c4.metric("Realised FDP", f"{fp/n_cert:.3f}" if n_cert else "n/a",
          delta=f"nominal {q}", delta_color="off")

if n_cert and fp / n_cert > q * 2:
    st.warning(
        f"**Realised FDP is {fp/n_cert:.2f} against a nominal {q}.** "
        "BH bounds the mean false discovery proportion across *all* batches, "
        "including those that raise nothing — and roughly 90% of batches stay "
        "silent, contributing a definitional zero. Conditional on an alert "
        "being raised, the error rate is much higher. Both numbers are correct; "
        "this one is what you experience."
    )

# ---------------------------------------------------------------- funnel
st.subheader("Funnel")
f1, f2, f3 = st.columns(3)
n_flows = int(d["n_flows"].sum())
f1.metric("Raw flows", f"{n_flows:,}")
f2.metric("Host-windows", f"{len(d):,}", delta=f"{n_flows/max(len(d),1):.0f} flows each",
          delta_color="off")
f3.metric("Certified alerts", f"{n_cert:,}",
          delta=f"1 per {n_flows/max(n_cert,1):,.0f} flows", delta_color="off")

# ---------------------------------------------------------------- queue
st.subheader("Ranked queue")
st.caption(
    "Ordered by evidence strength. The line marks where the false-discovery "
    "guarantee stops — everything below it is ranked but uncertified."
)

show = d.head(60).copy()
show["status"] = np.where(show[qcol], "CERTIFIED", "uncertified")
show["outcome"] = np.where(show["is_attack"], "attack", "benign")
show["p"] = show["p_value"].map(lambda v: f"{v:.2e}")
show["q_val"] = show["q_value"].map(lambda v: "—" if pd.isna(v) else f"{v:.3f}")

cols = ["status", "host", "window", "p", "q_val", "outcome", "n_flows",
        "dst_port_entropy", "top_port_share", "rst_rate", "profiled_host"]


def highlight(row):
    if row["status"] == "CERTIFIED":
        colour = "#1b4332" if row["outcome"] == "attack" else "#5c1a1a"
    else:
        colour = ""
    return [f"background-color: {colour}"] * len(row)


st.dataframe(
    show[cols].style.apply(highlight, axis=1).format(
        {"dst_port_entropy": "{:.2f}", "top_port_share": "{:.2f}",
         "rst_rate": "{:.2f}", "n_flows": "{:,.0f}"}
    ),
    width='stretch', height=420,
)
st.caption(
    "Green: certified and genuinely malicious. Red: certified but benign — a "
    "false discovery. Unshaded rows carry no guarantee either way."
)

# ---------------------------------------------------------------- evidence
st.subheader("Evidence")
e1, e2 = st.columns(2)

with e1:
    st.caption("Alert composition at the selected q")
    comp = pd.DataFrame({
        "count": [tp, fp, len(d) - n_cert],
    }, index=["certified · attack", "certified · benign", "uncertified"])
    st.bar_chart(comp)

with e2:
    st.caption("Volume reduction across q")
    rows = []
    for qq in meta["q_levels"]:
        c = "certified_q" + str(qq).replace("0.", "")
        sel = d[d[c]]
        rows.append({"q": qq, "alerts": len(sel),
                     "true": int(sel["is_attack"].sum()),
                     "false": len(sel) - int(sel["is_attack"].sum())})
    st.dataframe(pd.DataFrame(rows).set_index("q"), width='stretch')

# ---------------------------------------------------------------- caveats
with st.expander("What this dashboard does not claim"):
    st.markdown(
        """
- **The guarantee is per batch, not per queue.** BH controls the mean false
  discovery proportion within each time window. Applying it to the whole day's
  queue would require a calibration set roughly `m/q` in size — about **41
  benign days at q = 0.05**, against the one day available here.
- **Uncertified is not cleared.** Rows below the line are ranked by evidence and
  may well be malicious. They simply carry no bound.
- **Attacker hosts are absent from the baseline** in this dataset, so malicious
  windows differ in provenance as well as behaviour. Detection performance here
  is optimistic relative to a network where attackers use existing machines.
- **Evidence rank is not priority.** A weak signal on a domain controller
  outranks a strong signal on a meeting-room printer. Priority requires asset
  criticality, which is a human input, not a statistic.
        """
    )
