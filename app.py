import os, pandas as pd, numpy as np, plotly.express as px, plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output, State, callback_context
import dash_bootstrap_components as dbc

from data_utils import build_unified_long, apply_filters, compute_kpis
from forecasting import fit_prophet, extrema_mask
from alerts import churn_alerts

# ---------- Config ----------
SHEET_ID = os.environ.get("SHEET_ID", "1ACX9OWNB0vHs8EpxeHxgByuPjDP9VC0E3k9b61V-i1I")  # <- tu Sheet ID
SHEETS = ("MRR","CIHS","Transactions")

def load_data():
    df = build_unified_long(SHEET_ID, SHEETS)
    return df

df = load_data()

external_stylesheets = [dbc.themes.FLATLY]
app = Dash(__name__, external_stylesheets=external_stylesheets, title="Instaleap BI — Dash")
server = app.server

# ------------- Helpers de UI -------------
def kpi_cards(mrr_df: pd.DataFrame):
    k = compute_kpis(mrr_df)
    return dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([html.Small("MRR Total"), html.H3(f"${k['total_last']:,.0f}"), html.Small(k["last_period"])])), md=3),
        dbc.Col(dbc.Card(dbc.CardBody([html.Small("MoM"), html.H3(f"{k['mom']:,.1f}%" if k["mom"] is not None else "—")])), md=2),
        dbc.Col(dbc.Card(dbc.CardBody([html.Small("YoY"), html.H3(f"{k['yoy']:,.1f}%" if k["yoy"] is not None else "—")])), md=2),
        dbc.Col(dbc.Card(dbc.CardBody([html.Small("ARR"), html.H3(f"${k['arr']:,.0f}")])), md=2),
    ], className="g-3")

def line_with_forecast(hist: pd.DataFrame, title: str, horizon=6, label_mode="extrema"):
    fc = fit_prophet(hist[["date","value"]].sort_values("date"), horizon=horizon)
    fig = go.Figure()
    if not hist.empty:
        fig.add_trace(go.Scatter(x=hist["date"], y=hist["value"], mode="lines", name="Histórico"))
    if not fc.empty:
        fig.add_trace(go.Scatter(x=list(fc["date"])+list(fc["date"][::-1]),
                                 y=list(fc["yhat_upper"])+list(fc["yhat_lower"][::-1]),
                                 fill="toself", name="Intervalo", opacity=0.2, line=dict(width=0)))
        mode = "lines"
        text = None
        if label_mode in ("extrema","all"):
            y = fc["yhat"].to_numpy()
            if label_mode=="all":
                text = [f"{v:,.0f}" for v in y]
            else:
                is_max, is_min = extrema_mask(y, window=2)
                text = [f"{y[i]:,.0f}" if (is_max[i] or is_min[i]) else None for i in range(len(y))]
            mode = "lines+markers+text"
        fig.add_trace(go.Scatter(x=fc["date"], y=fc["yhat"], mode=mode, name="Pronóstico", text=text, textposition="top center"))
    fig.update_layout(title=title, height=420, margin=dict(l=10,r=10,t=40,b=10), xaxis_title="Fecha", yaxis_title="")
    return fig, fc

def countries_summary(mrr_df: pd.DataFrame) -> pd.DataFrame:
    last_p = mrr_df["period"].max()
    g_last = mrr_df[mrr_df["period"]==last_p]
    agg = g_last.groupby("pais", as_index=False)["value"].sum().rename(columns={"value":"MRR"})
    agg["ARR"] = agg["MRR"]*12
    # MoM/YoY
    out_rows = []
    for pais in agg["pais"]:
        hist = mrr_df[mrr_df["pais"]==pais].groupby("period", as_index=False)["value"].sum().sort_values("period")
        if hist.empty: continue
        last = hist.iloc[-1]["value"]
        prev = hist.iloc[-2]["value"] if len(hist)>=2 else None
        yoyp = str((pd.Period(hist.iloc[-1]["period"])-12))
        yoyv = float(hist.loc[hist["period"]==yoyp,"value"].sum()) if yoyp in hist["period"].values else None
        mom = ((last-prev)/prev*100) if prev and prev>0 else None
        yoy = ((last-yoyv)/yoyv*100) if yoyv and yoyv>0 else None
        clients = int((mrr_df[(mrr_df["pais"]==pais) & (mrr_df["period"]==last_p)]
                      .groupby("cliente")["value"].sum()>0).sum())
        out_rows.append({"pais": pais, "MRR": last, "ARR": last*12, "MoM_%": None if mom is None else round(mom,1),
                         "YoY_%": None if yoy is None else round(yoy,1), "Clientes": clients})
    return pd.DataFrame(out_rows).sort_values("MRR", ascending=False)

# ------------- Layout -------------
def options(values):
    return [{"label": str(v), "value": v} for v in values]

zones = sorted([z for z in df["zona"].dropna().unique().tolist()])
types = sorted([t for t in df["type"].dropna().unique().tolist()])
clients = sorted([c for c in df["cliente"].dropna().unique().tolist()])

app.layout = dbc.Container([
    html.Br(),
    html.H2("Instaleap — Revenue & Health (Dash)"),
    html.Div("MRR · CIHS · Transacciones · Cohortes · País · Alertas"),
    html.Hr(),

    dbc.Row([
        dbc.Col(dcc.Dropdown(id="zona-dd", options=options(zones), value=zones, multi=True, placeholder="Zona"), md=3),
        dbc.Col(dcc.Dropdown(id="type-dd", options=options(types), value=types, multi=True, placeholder="Tipo"), md=3),
        dbc.Col(dcc.Dropdown(id="client-dd", options=options(clients), value=[], multi=True, placeholder="Clientes (opcional)"), md=6),
    ], className="mb-2"),
    dbc.Button("Actualizar datos", id="refresh-btn", color="secondary", size="sm"),
    dcc.Store(id="data-store"),

    html.Br(),
    html.Div(id="kpis-row"),

    html.Br(),
    dbc.Tabs(id="tabs", active_tab="tab-mrr", children=[
        dbc.Tab(label="MRR & Forecast", tab_id="tab-mrr"),
        dbc.Tab(label="CIHS", tab_id="tab-cihs"),
        dbc.Tab(label="Transacciones", tab_id="tab-tx"),
        dbc.Tab(label="Cohortes", tab_id="tab-cohort"),
        dbc.Tab(label="Mapa por país", tab_id="tab-map"),
        dbc.Tab(label="Alertas de churn", tab_id="tab-alerts"),
    ]),
    html.Div(id="tab-content"),

    # Descargas
    dcc.Download(id="dl-mrr-forecast"),
    dcc.Download(id="dl-cihs-forecast"),
    dcc.Download(id="dl-cohort"),
    dcc.Download(id="dl-country"),
])

# ------------- Callbacks -------------

# Inicializa data-store
@app.callback(Output("data-store","data"), Input("refresh-btn","n_clicks"), prevent_initial_call=False)
def refresh_data(_):
    data = load_data()
    return data.to_dict("records")

# Opciones reactivas de clientes
@app.callback(
    Output("client-dd","options"),
    Input("zona-dd","value"), Input("type-dd","value"), Input("data-store","data")
)
def update_clients(z_sel, t_sel, data):
    d = pd.DataFrame(data or [])
    if d.empty: return []
    filt = apply_filters(d, z_sel, t_sel, None)
    return options(sorted([c for c in filt["cliente"].dropna().unique().tolist()]))

# KPIs
@app.callback(
    Output("kpis-row","children"),
    Input("zona-dd","value"), Input("type-dd","value"), Input("client-dd","value"), Input("data-store","data")
)
def update_kpis(z_sel, t_sel, c_sel, data):
    d = pd.DataFrame(data or [])
    if d.empty: return html.Div("Cargando datos...")
    f = apply_filters(d, z_sel, t_sel, c_sel)
    m = f[f["metric"]=="MRR"]
    if m.empty: return html.Div("Sin MRR con los filtros actuales.")
    return kpi_cards(m)

# Tabs content
@app.callback(
    Output("tab-content","children"),
    Input("tabs","active_tab"),
    Input("zona-dd","value"), Input("type-dd","value"), Input("client-dd","value"),
    Input("data-store","data")
)
def render_tab(tab, z_sel, t_sel, c_sel, data):
    d = pd.DataFrame(data or [])
    if d.empty: return html.Div("Cargando...")
    f = apply_filters(d, z_sel, t_sel, c_sel)

    if tab=="tab-mrr":
        m = f[f["metric"]=="MRR"].groupby("date", as_index=False)["value"].sum().sort_values("date")
        fig, fc = line_with_forecast(m, "MRR — Histórico + Pronóstico", horizon=6, label_mode="extrema")
        merged = m.rename(columns={"value":"historico"}).merge(fc, on="date", how="outer").sort_values("date")
        return dbc.Row([
            dbc.Col(dcc.Graph(figure=fig), md=8),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Datos de pronóstico"),
                html.Div("CSV con histórico + yhat + intervalos"),
                dbc.Button("Descargar CSV", id="btn-dl-mrr", color="primary")
            ])), md=4)
        ])

    if tab=="tab-cihs":
        c = f[f["metric"]=="CIHS"].groupby("date", as_index=False)["value"].sum().sort_values("date")
        fig, fc = line_with_forecast(c, "CIHS (suma total) — Histórico + Pronóstico", horizon=6, label_mode="extrema")
        return dbc.Row([
            dbc.Col(dcc.Graph(figure=fig), md=8),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Datos de pronóstico CIHS"),
                dbc.Button("Descargar CSV", id="btn-dl-cihs", color="primary")
            ])), md=4)
        ])

    if tab=="tab-tx":
        t = f[f["metric"]=="Transactions"].groupby("date", as_index=False)["value"].sum().sort_values("date")
        fig, fc = line_with_forecast(t, "Transacciones — Histórico + Pronóstico", horizon=6, label_mode="none")
        return dcc.Graph(figure=fig)

    if tab=="tab-cohort":
        m = f[f["metric"]=="MRR"].copy()
        if m.empty: return html.Div("Sin MRR")
        sel = m["period"].max()
        yoy = str((pd.Period(sel)-12))
        cohort = m[(m["period"]==sel) & (m["value"]>0)]["cliente"].unique().tolist()
        now_df = m[(m["period"]==sel) & (m["cliente"].isin(cohort))].groupby("cliente", as_index=False)["value"].sum().rename(columns={"value":"now"})
        yoy_df = m[(m["period"]==yoy) & (m["cliente"].isin(cohort))].groupby("cliente", as_index=False)["value"].sum().rename(columns={"value":"yoy"})
        det = now_df.merge(yoy_df, on="cliente", how="outer").fillna(0.0)
        det["diff"] = det["now"] - det["yoy"]
        det["diff_%"] = det.apply(lambda r: (r["diff"]/r["yoy"]*100) if r["yoy"]>0 else None, axis=1)
        topN = 12
        top_clients = det.sort_values("now", ascending=False).head(topN)["cliente"].tolist()
        det["bucket"] = det["cliente"].where(det["cliente"].isin(top_clients), other="Otros")
        stack = det.groupby("bucket", as_index=False)[["now","yoy"]].sum()

        fig = go.Figure()
        for _, row in stack.iterrows():
            fig.add_trace(go.Bar(x=["Año anterior","Mes seleccionado"], y=[row["yoy"], row["now"]], name=str(row["bucket"])))
        total_yoy = stack["yoy"].sum(); total_now = stack["now"].sum()
        fig.add_trace(go.Scatter(x=["Año anterior"], y=[total_yoy], mode="text", text=[f"${total_yoy:,.0f}"], textposition="top center", showlegend=False))
        fig.add_trace(go.Scatter(x=["Mes seleccionado"], y=[total_now], mode="text", text=[f"${total_now:,.0f}"], textposition="top center", showlegend=False))
        fig.update_layout(barmode="stack", title=f"Cohorte {sel}: activos vs {yoy}", height=420, margin=dict(l=10,r=10,t=40,b=10))

        summary = dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([html.Small(f"Total {sel}"), html.H4(f"${total_now:,.0f}")]))),
            dbc.Col(dbc.Card(dbc.CardBody([html.Small(f"Total {yoy}"), html.H4(f"${total_yoy:,.0f}")]))),
            dbc.Col(dbc.Card(dbc.CardBody([html.Small("Diferencia"), html.H4(f"${(total_now-total_yoy):,.0f}")]))),
            dbc.Col(dbc.Button("Descargar detalle", id="btn-dl-cohort", color="primary"))
        ], className="g-3")

        table = dbc.Table.from_dataframe(det.sort_values("diff", ascending=False), striped=True, hover=True, bordered=False)

        return html.Div([summary, html.Br(), dcc.Graph(figure=fig), html.Br(), table])

    if tab=="tab-map":
        m = f[f["metric"]=="MRR"].copy()
        if m.empty: return html.Div("Sin MRR")
        last = m["period"].max()
        by = m[m["period"]==last].groupby("pais", as_index=False)["value"].sum().rename(columns={"value":"MRR"})
        by["ARR"] = by["MRR"]*12
        fig = px.choropleth(by, locations="pais", locationmode="country names", color="MRR",
                            hover_data={"ARR":":,.0f"}, color_continuous_scale="Blues", scope="world")
        fig.update_geos(showcountries=True, countrycolor="#555", showland=True, landcolor="#E5ECF6")
        summary = countries_summary(m)
        return html.Div([
            dcc.Graph(id="map-graph", figure=fig),
            html.Br(),
            dbc.Card(dbc.CardBody([
                html.H6("Resumen por país"),
                dbc.Button("Descargar CSV", id="btn-dl-country", color="primary", size="sm"),
                html.Br(), html.Br(),
                dbc.Table.from_dataframe(summary, striped=True, hover=True, bordered=False),
            ])),
            html.Br(),
            html.Div(id="country-detail")
        ])

    if tab=="tab-alerts":
        a = churn_alerts(f)
        if a.empty: return html.Div("Sin alertas con las reglas actuales.")
        return dbc.Table.from_dataframe(a.sort_values(["riesgo","ultimo_mrr"], ascending=[True, False]), striped=True, hover=True, bordered=False)

    return html.Div("Tab no disponible.")

# Descargas
@app.callback(
    Output("dl-mrr-forecast","data"),
    Input("btn-dl-mrr","n_clicks"),
    State("zona-dd","value"), State("type-dd","value"), State("client-dd","value"), State("data-store","data"),
    prevent_initial_call=True
)
def dl_mrr(n, z, t, c, data):
    d = pd.DataFrame(data or []); f = apply_filters(d, z, t, c)
    m = f[f["metric"]=="MRR"].groupby("date", as_index=False)["value"].sum().sort_values("date")
    fc = fit_prophet(m, horizon=6)
    merged = m.rename(columns={"value":"historico"}).merge(fc, on="date", how="outer").sort_values("date")
    merged["period"] = merged["date"].dt.to_period("M").astype(str)
    return dcc.send_data_frame(merged.to_csv, "mrr_consolidado_forecast.csv", index=False)

@app.callback(
    Output("dl-cihs-forecast","data"),
    Input("btn-dl-cihs","n_clicks"),
    State("zona-dd","value"), State("type-dd","value"), State("client-dd","value"), State("data-store","data"),
    prevent_initial_call=True
)
def dl_cihs(n, z, t, c, data):
    d = pd.DataFrame(data or []); f = apply_filters(d, z, t, c)
    cihs = f[f["metric"]=="CIHS"].groupby("date", as_index=False)["value"].sum().sort_values("date")
    fc = fit_prophet(cihs, horizon=6)
    merged = cihs.rename(columns={"value":"historico"}).merge(fc, on="date", how="outer").sort_values("date")
    merged["period"] = merged["date"].dt.to_period("M").astype(str)
    return dcc.send_data_frame(merged.to_csv, "cihs_consolidado_forecast.csv", index=False)

@app.callback(
    Output("dl-cohort","data"),
    Input("btn-dl-cohort","n_clicks"),
    State("zona-dd","value"), State("type-dd","value"), State("client-dd","value"), State("data-store","data"),
    prevent_initial_call=True
)
def dl_cohort(n, z, t, c, data):
    d = pd.DataFrame(data or []); f = apply_filters(d, z, t, c)
    m = f[f["metric"]=="MRR"].copy()
    sel = m["period"].max(); yoy = str((pd.Period(sel)-12))
    cohort = m[(m["period"]==sel) & (m["value"]>0)]["cliente"].unique().tolist()
    now_df = m[(m["period"]==sel) & (m["cliente"].isin(cohort))].groupby("cliente", as_index=False)["value"].sum().rename(columns={"value":"now"})
    yoy_df = m[(m["period"]==yoy) & (m["cliente"].isin(cohort))].groupby("cliente", as_index=False)["value"].sum().rename(columns={"value":"yoy"})
    det = now_df.merge(yoy_df, on="cliente", how="outer").fillna(0.0)
    det["diff"] = det["now"] - det["yoy"]
    det["diff_%"] = det.apply(lambda r: (r["diff"]/r["yoy"]*100) if r["yoy"]>0 else None, axis=1)
    return dcc.send_data_frame(det.to_csv, f"cohorte_{sel}_detalle.csv", index=False)

@app.callback(
    Output("dl-country","data"),
    Input("btn-dl-country","n_clicks"),
    State("zona-dd","value"), State("type-dd","value"), State("client-dd","value"), State("data-store","data"),
    prevent_initial_call=True
)
def dl_country(n, z, t, c, data):
    d = pd.DataFrame(data or []); f = apply_filters(d, z, t, c)
    m = f[f["metric"]=="MRR"].copy()
    return dcc.send_data_frame(countries_summary(m).to_csv, "resumen_paises.csv", index=False)

if __name__ == "__main__":
    app.run_server(debug=True)
