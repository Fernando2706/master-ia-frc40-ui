from __future__ import annotations

import numpy as np
import pandas as pd


def standardize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "fecha": pd.to_datetime(df["Fecha"], errors="coerce"),
            "caudal_tratado": pd.to_numeric(df["Caudal tratado"], errors="coerce"),
            "dqo_entrada": pd.to_numeric(df["DQO Entrante"], errors="coerce"),
            "dqo_salida": pd.to_numeric(df["DQO Saliente"], errors="coerce"),
            "dqo_eliminada": pd.to_numeric(df["Caudal por DQO"], errors="coerce"),
            "policloruro_aluminio": pd.to_numeric(df["Policloruro de aluminio"], errors="coerce"),
            "coagulante_organico": pd.to_numeric(
                df["ADITIVOS/TRATAMIEN COAGULANTE ORG. POLYD"], errors="coerce"
            ),
            "floculante_cationico": pd.to_numeric(
                df["FLOCULANTE CATIONICO -POLIELECTROLITO"], errors="coerce"
            ),
        }
    )
    out = out[out["fecha"].notna()].sort_values("fecha").reset_index(drop=True)
    safe_flow = out["caudal_tratado"].where(out["caudal_tratado"] > 0)
    safe_effluent = out["dqo_salida"].where(out["dqo_salida"] > 0)
    out["carga_dqo_entrada"] = out["caudal_tratado"] * out["dqo_entrada"] / 1000
    out["dosis_policloruro_m3"] = out["policloruro_aluminio"] / safe_flow
    out["dosis_coagulante_m3"] = out["coagulante_organico"] / safe_flow
    out["dosis_floculante_m3"] = out["floculante_cationico"] / safe_flow
    out["reduccion_dqo"] = (out["dqo_entrada"] - out["dqo_salida"]).clip(lower=0)
    out["ratio_entrada_salida"] = out["dqo_entrada"] / safe_effluent
    out["mes"] = out["fecha"].dt.month
    out["dia_ano"] = out["fecha"].dt.dayofyear
    numeric_cols = out.columns.drop("fecha")
    out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    return out


def build_prediction_row(fecha, caudal: float, dqo_entrada: float, dqo_salida: float) -> dict:
    dqo_eliminada = caudal * max(dqo_entrada - dqo_salida, 0) / 1000
    ratio = dqo_entrada / dqo_salida if dqo_salida > 0 else 0
    return {
        "fecha": fecha,
        "caudal_tratado": caudal,
        "dqo_entrada": dqo_entrada,
        "dqo_salida": dqo_salida,
        "dqo_eliminada": dqo_eliminada,
        "reduccion_dqo": max(dqo_entrada - dqo_salida, 0),
        "ratio_entrada_salida": ratio,
        "mes": fecha.month,
        "dia_ano": fecha.dayofyear,
    }

