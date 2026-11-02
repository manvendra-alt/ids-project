import sys, time
sys.path.append("src")
import pandas as pd
import torch
import streamlit as st
from data_prep import prepare_data
from models import IDSNet
from attacks import pgd

st.set_page_config(page_title="ML-Based NIDS", layout="wide")

CONTROLLABLE = ["duration", "src_bytes", "count", "srv_count", "dst_host_count", "dst_host_srv_count"]

@st.cache_resource
def load_all():
    d = prepare_data("data/KDDTrain.txt", "data/KDDTest.txt")
    n = d["X_test"].shape[1]
    models = {}
    for name, path in [("Baseline", "saved_models/nn_baseline.pt"),
                       ("Hardened (adversarial training)", "saved_models/nn_hardened.pt")]:
        m = IDSNet(n)
        m.load_state_dict(torch.load(path))
        m.eval()
        models[name] = m
    return d, models

d, models = load_all()
X_test = torch.tensor(d["X_test"])
y_test = torch.tensor(d["y_test"], dtype=torch.long)
mask = torch.tensor([1.0 if n in CONTROLLABLE else 0.0 for n in d["feature_names"]])

st.title("ML-Based Network Intrusion Detection: Attack and Defense Demo")
st.caption("Replay of recorded NSL-KDD test traffic (not live capture).")

with st.sidebar:
    model_name = st.selectbox("Detector", list(models))
    attack_on = st.checkbox("Attacker alters malicious traffic", value=False)
    attack_kind = st.radio("Attack type", ["Unconstrained PGD", "Realistic PGD (controllable features only)"])
    eps = st.slider("Attacker strength (eps)", 0.0, 1.0, 0.1, 0.05)
    n_flows = st.slider("Flows to replay", 50, 500, 200, 50)
    delay = st.slider("Delay per flow (seconds)", 0.0, 0.2, 0.02, 0.01)
    start = st.button("Start replay")

model = models[model_name]

if start:
    idx = torch.randperm(len(X_test))[:n_flows]
    X, y = X_test[idx].clone(), y_test[idx]
    if attack_on and eps > 0:
        m = y == 1
        use_mask = mask if attack_kind.startswith("Realistic") else None
        X[m] = pgd(model, X[m], y[m], eps, mask=use_mask)
    with torch.no_grad():
        probs = torch.softmax(model(X), dim=1)[:, 1]
    preds = (probs > 0.5).long()

    cols = st.columns(4)
    ph = [c.empty() for c in cols]
    chart = st.empty()
    table = st.empty()
    seen = alerts = caught = missed = 0
    history, curve = [], []

    for i in range(n_flows):
        t, p = int(y[i]), int(preds[i])
        seen += 1
        alerts += p
        caught += int(t == 1 and p == 1)
        missed += int(t == 1 and p == 0)
        history.append({"flow": i + 1, "true label": "ATTACK" if t else "benign",
                        "model says": "ALERT" if p else "benign",
                        "attack score": round(float(probs[i]), 3)})
        curve.append(missed)
        ph[0].metric("Flows seen", seen)
        ph[1].metric("Alerts raised", alerts)
        ph[2].metric("Attacks caught", caught)
        ph[3].metric("Attacks MISSED", missed)
        table.dataframe(pd.DataFrame(history[-8:]))
        if i % 5 == 0 or i == n_flows - 1:
            chart.line_chart(pd.DataFrame({"missed attacks (cumulative)": curve}))
        time.sleep(delay)

st.subheader("Robustness results: evasion rate of detected attacks (test set)")
res = pd.DataFrame({
    "eps": [0.05, 0.1, 0.2, 0.5, 1.0],
    "Baseline, PGD": [40.2, 98.7, 99.9, 100.0, 100.0],
    "Hardened, PGD": [7.2, 13.2, 22.9, 98.0, 100.0],
    "Baseline, realistic PGD": [1.4, 3.0, 7.4, 17.8, 26.8],
    "Hardened, realistic PGD": [1.2, 1.8, 3.3, 8.3, 16.5],
}).set_index("eps")
st.line_chart(res)
st.dataframe(res)