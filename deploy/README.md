---
title: qtriage
emoji: 🔬
colorFrom: green
colorTo: red
sdk: streamlit
sdk_version: 1.40.0
app_file: app.py
pinned: false
license: mit
---

# qtriage — statistically certified alert triage

Conformal p-values plus Benjamini–Hochberg correction applied to network
intrusion detection, on the corrected CICIDS2017 dataset.

**The finding this dashboard exists to show:** BH's false discovery guarantee
holds exactly as promised, but bounds the mean FDP across *all* time batches —
including the ~90% that raise no alert at all and contribute a definitional
zero. Conditional on the system actually alerting, 30% of alerts are false at a
nominal q of 0.05. Both numbers are correct. Only one is what an analyst
experiences.

Ranking and certification are separate capabilities. Every host-window has a
p-value, so the ranked queue always exists; BH draws a line on it with a bound
attached. This dashboard shows the line explicitly rather than implying a
guarantee over the whole list.

Full method, results and limitations: see the project repository.
