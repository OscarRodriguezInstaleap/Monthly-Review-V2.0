import re
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime

SPANISH_MONTH_MAP = {
    "enero":"january","febrero":"february","marzo":"march","abril":"april",
    "mayo":"may","junio":"june","julio":"july","agosto":"august",
    "setiembre":"september","septiembre":"september",
    "octubre":"october","noviembre":"november","diciembre":"december",
    "ene":"jan","abr":"apr","ago":"aug","sept":"sep","dic":"dec",
}

def es_to_en(s: str) -> str:
    t = str(s).strip().lower()
    for es, en in SPANISH_MONTH_MAP.items():
        t = re.sub(rf"\b{re.escape(es)}\b", en, t, flags=re.IGNORECASE)
    return t

def read_gsheet_csv(sheet_id: str, sheet_name: str) -> pd.DataFrame:
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    return pd.read_csv(url)

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    low = {c: (str(c).strip().lower() if not isinstance(c, (pd.Timestamp, datetime)) else c) for c in df.columns}
    df.rename(columns=low, inplace=True)
    mapping = {
        "type client":"type","tipo cliente":"type","tipo de cliente":"type",
        "razon social":"razon_social","razón social":"razon_social","country":"pais","país":"pais",
    }
    for k, v in mapping.items():
        if k in df.columns:
            df.rename(columns={k: v}, inplace=True)
    return df

def detect_meta_and_time(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    def is_time_header(col):
        if isinstance(col, (pd.Timestamp, datetime)): return True
        s = es_to_en(str(col))
        for fmt in ("%b-%y","%b-%Y","%B-%y","%B-%Y","%Y-%m","%Y/%m","%m-%Y","%m/%Y"):
            try: pd.to_datetime(s, format=fmt); return True
            except: pass
        try:
            pd.to_datetime(s)
            if str(col).lower() not in {"type","zona","nuevo","cliente","razon social","razón social","razon_social","pais"}:
                return True
        except: return False
        return False

    meta, ts = [], []
    for c in df.columns:
        (ts if is_time_header(c) else meta).append(c)
    return meta, ts

def to_long(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    df = standardize_columns(df)
    meta, ts = detect_meta_and_time(df)
    if not ts:
        return pd.DataFrame(columns=meta+["date","period","value","metric"])

    long = df.melt(id_vars=meta, value_vars=ts, var_name="month", value_name="value_raw")
    long["value"] = pd.to_numeric(
        long["value_raw"].astype(str).str.replace(",","").str.replace(" ",""),
        errors="coerce"
    ).fillna(0.0)

    def parse_date(x):
        s = es_to_en(str(x))
        for fmt in ("%b-%y","%b-%Y","%B-%y","%B-%Y","%Y-%m","%Y/%m","%m-%Y","%m/%Y"):
            try: return pd.to_datetime(s, format=fmt)
            except: pass
        try: return pd.to_datetime(s)
        except: return pd.NaT

    long["date"] = long["month"].map(parse_date)
    long["period"] = long["date"].dt.to_period("M").astype(str)
    long["metric"] = metric

    for col in ["type","zona","nuevo","cliente","razon_social","pais"]:
        if col not in long.columns: long[col] = None
    return long

# data_utils.py — reemplaza SOLO esta función

def build_unified_long(sheet_id: str, sheets=("MRR","CIHS","Transactions")) -> pd.DataFrame:
    """
    Lee hojas desde Google Sheets en formato CSV (gviz) y arma un dataframe long unificado.
    Tolera alias de nombres de pestañas y falla por hoja (no por todo).
    """
    # Aliases tolerados por métrica
    CANDIDATES = {
        "MRR": ["MRR", "Revenue", "Ingresos", "MRR$"],
        "CIHS": ["CIHS", "Health", "Adopcion", "Adopción"],
        "Transactions": ["Transactions", "Transacciones", "Orders", "TX"],
    }

    parts = []
    used_tabs = []

    def _try_add(tab_name: str, metric_name: str):
        try:
            df = read_gsheet_csv(sheet_id, tab_name)
            parts.append(to_long(df, metric_name))
            used_tabs.append((metric_name, tab_name))
            return True
        except Exception:
            return False

    # 1) Primero intenta con los nombres pasados explícitos (param sheets)
    for s in sheets:
        key = s.strip().lower()
        if key in ("mrr", "revenue"):
            if not _try_add(s, "MRR"):
                # intenta alias
                for alt in CANDIDATES["MRR"]:
                    if alt == s: continue
                    if _try_add(alt, "MRR"): break
        elif key in ("cihs", "health"):
            if not _try_add(s, "CIHS"):
                for alt in CANDIDATES["CIHS"]:
                    if alt == s: continue
                    if _try_add(alt, "CIHS"): break
        else:
            # Transactions (o cualquier otro nombre)
            if not _try_add(s, "Transactions"):
                for alt in CANDIDATES["Transactions"]:
                    if alt == s: continue
                    if _try_add(alt, "Transactions"): break

    # 2) Si por algún motivo no se agregó nada, intenta al menos los defaults
    if not parts:
        for metric, candidates in CANDIDATES.items():
            for tab in candidates:
                if _try_add(tab, metric): break

    if not parts:
        # Nada de nada: devuelve un dataframe vacío con columnas esperadas
        return pd.DataFrame(columns=["type","zona","nuevo","cliente","razon_social","pais","date","period","value","metric"])

    # Log útil en Render (verás en “Logs” qué pestañas se usaron realmente)
    print("✅ Hojas detectadas:", used_tabs)

    all_df = pd.concat(parts, ignore_index=True)

    # Normaliza meta y completa últimos no-nulos por cliente (para zona/type/pais, etc.)
    all_df = standardize_columns(all_df)
    meta_cols = ["type","zona","nuevo","razon_social","pais"]
    tmp = all_df.sort_values("date")
    last_meta = tmp.groupby("cliente", as_index=False)[meta_cols].agg(lambda s: s.dropna().iloc[-1] if s.dropna().size else None)
    out = all_df.merge(last_meta, on="cliente", how="left", suffixes=("","_bf"))
    for c in meta_cols:
        out[c] = out[c].fillna(out[f"{c}_bf"])
        out.drop(columns=[f"{c}_bf"], inplace=True)
    return out

def compute_kpis(mrr_df: pd.DataFrame):
    """Devuelve dict con periodos y métricas MRR total, MoM, YoY, ARR, clientes activos."""
    agg = mrr_df.groupby("period", as_index=False)["value"].sum().sort_values("period")
    last_p = agg["period"].max()
    periods = agg["period"].tolist()
    idx = periods.index(last_p) if last_p in periods else -1
    prev_p = periods[idx-1] if idx-1>=0 else None
    yoy_p = str((pd.Period(last_p)-12)) if last_p else None

    total_last = float(agg.loc[agg["period"]==last_p, "value"].sum()) if last_p else 0.0
    total_prev = float(agg.loc[agg["period"]==prev_p, "value"].sum()) if prev_p else 0.0
    total_yoy = float(agg.loc[agg["period"]==yoy_p, "value"].sum()) if yoy_p in periods else 0.0

    mom = ((total_last-total_prev)/total_prev*100) if total_prev>0 else None
    yoy = ((total_last-total_yoy)/total_yoy*100) if total_yoy>0 else None
    arr = total_last*12.0
    return {
        "last_period": last_p, "total_last": total_last,
        "mom": mom, "yoy": yoy, "arr": arr
    }

def apply_filters(df: pd.DataFrame, zones, types, clients):
    out = df.copy()
    if zones: out = out[out["zona"].isin(zones)]
    if types: out = out[out["type"].isin(types)]
    if clients: out = out[out["cliente"].isin(clients)]
    return out
