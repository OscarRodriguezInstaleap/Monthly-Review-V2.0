import pandas as pd

def churn_alerts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return pd.DataFrame()
    mrr = df[df["metric"]=="MRR"].copy()
    tx  = df[df["metric"]=="Transactions"].copy()
    cihs = df[df["metric"]=="CIHS"].copy()

    alerts = []
    for cli, g in mrr.groupby("cliente"):
        hist = g.groupby("period", as_index=False)["value"].sum().sort_values("period")
        if len(hist) < 3: continue
        last = hist.iloc[-1]["value"]; prev = hist.iloc[-2]["value"]
        mom_drop = (prev-last)/prev*100 if prev>0 else 0

        cihs_last = cihs[cihs["cliente"]==cli].sort_values("period").tail(1)["value"].mean() if not cihs.empty else None
        tx_hist = tx[tx["cliente"]==cli].groupby("period", as_index=False)["value"].sum().sort_values("period")
        tx_drop = None
        if len(tx_hist)>=2:
            last_tx = tx_hist.iloc[-1]["value"]; prev_tx = tx_hist.iloc[-2]["value"]
            tx_drop = (prev_tx-last_tx)/prev_tx*100 if prev_tx>0 else None

        riesgo = 0
        if mom_drop>=30: riesgo += 1
        if tx_drop is not None and tx_drop>=30: riesgo += 1
        if cihs_last is not None and cihs_last < 0.3: riesgo += 1

        if riesgo>0:
            alerts.append({
                "cliente": cli,
                "ultimo_mrr": round(last,2),
                "caida_mrr_mom_%": round(mom_drop,1),
                "cihs_ultimo": None if pd.isna(cihs_last) else round(float(cihs_last),2),
                "caida_tx_mom_%": None if tx_drop is None else round(tx_drop,1),
                "riesgo": riesgo
            })
    return pd.DataFrame(alerts)
