from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CHEMICAL_COLUMNS_RAW


def days_until_next_month(date_value: pd.Timestamp) -> dt.timedelta:
    if pd.isna(date_value):
        return pd.NaT
    if date_value.month < 12:
        next_month = date_value.replace(month=date_value.month + 1)
    else:
        next_month = date_value.replace(year=date_value.year + 1, month=1)
    return next_month - date_value


def get_multi_col(df: pd.DataFrame, level_0: str, level_1_contains: str) -> tuple[str, str]:
    for col in df.columns:
        if str(col[0]).strip() == level_0 and level_1_contains.lower() in str(col[1]).strip().lower():
            return col
    raise KeyError(f"No se encuentra columna {level_0} / {level_1_contains}")


def convert_excels(datos_path: Path, quimicos_path: Path, output_dir: Path) -> pd.DataFrame:
    depuradora = pd.read_excel(
        datos_path,
        sheet_name="Datos",
        index_col=None,
        header=[0, 1],
        na_values=["NA"],
    )
    frc40 = depuradora.take([0, 7, 8, 9, 10, 11], axis=1).copy()
    frc40 = frc40.rename(columns={"Caudal tratado": "Caudal tratado"})

    date_col = frc40.columns[0]
    homo_dqo_col = get_multi_col(frc40, "HOMO", "DQO")
    frc40_dqo_col = get_multi_col(frc40, "FRC40", "DQO")
    scada_col = get_multi_col(frc40, "FRC40", "Lectura caudal scada")
    hour_col = get_multi_col(frc40, "FRC40", "Hora")
    treated_flow_col = get_multi_col(frc40, "FRC40", "Caudal tratado")

    for col in [homo_dqo_col, frc40_dqo_col]:
        frc40[col] = frc40[col].replace(["-", "--", "ND", "V"], np.nan)

    frc40 = frc40.bfill()
    if not frc40.empty:
        frc40 = frc40.drop([0]).reset_index(drop=True)

    frc40 = frc40.iloc[:503].copy()
    if len(frc40) >= 503:
        for row_idx in [500, 501, 502]:
            frc40.loc[row_idx, homo_dqo_col] = frc40.loc[499, homo_dqo_col]
            frc40.loc[row_idx, frc40_dqo_col] = frc40.loc[499, frc40_dqo_col]

    if len(frc40) > 142:
        frc40.loc[:142, homo_dqo_col] = pd.to_numeric(frc40.loc[:142, homo_dqo_col], errors="coerce") * 1000

    frc40[scada_col] = frc40[scada_col].map(lambda x: np.nan if isinstance(x, str) else x)
    frc40[treated_flow_col] = frc40[treated_flow_col].map(
        lambda x: np.nan if pd.notna(x) and (x < -500000 or x > 500000) else x
    )
    frc40 = frc40.bfill()

    caudal_por_dqo = (
        pd.to_numeric(frc40[treated_flow_col], errors="coerce")
        * (
            pd.to_numeric(frc40[homo_dqo_col], errors="coerce")
            - pd.to_numeric(frc40[frc40_dqo_col], errors="coerce")
        )
    ) / 1000

    quimicos = pd.read_excel(
        quimicos_path,
        sheet_name="Hoja1",
        index_col=None,
        header=2,
        na_values=["NA"],
    )
    quimicos = quimicos.take([0, 4, 5, 7], axis=1).copy()
    quimicos = quimicos.replace("-", 0)
    if len(quimicos) > 1 and "Policloruro de aluminio" in quimicos.columns:
        quimicos.loc[1, "Policloruro de aluminio"] = 0

    date_chem_col = quimicos.columns[0]
    quimicos["dias"] = pd.to_datetime(quimicos[date_chem_col], errors="coerce").map(days_until_next_month)
    quimicos = quimicos.rename(columns={date_chem_col: "Fecha"})
    quimicos = quimicos.replace(0, np.nan).bfill()

    final = pd.DataFrame(
        {
            "Fecha": pd.to_datetime(frc40[date_col], errors="coerce"),
            "DQO Entrante": pd.to_numeric(frc40[homo_dqo_col], errors="coerce"),
            "Lectura caudal SCADA": pd.to_numeric(frc40[scada_col], errors="coerce"),
            "Hora": frc40[hour_col],
            "Caudal tratado": pd.to_numeric(frc40[treated_flow_col], errors="coerce"),
            "DQO Saliente": pd.to_numeric(frc40[frc40_dqo_col], errors="coerce"),
            "Caudal por DQO": caudal_por_dqo,
        }
    )
    final = final.merge(quimicos, on="Fecha", how="left").ffill()

    for chem_col in CHEMICAL_COLUMNS_RAW:
        if chem_col not in final.columns:
            raise KeyError(f"Falta columna de quimico: {chem_col}")
        final[chem_col] = final.apply(
            lambda row: row[chem_col] / row["dias"].days if pd.notna(row["dias"]) else np.nan,
            axis=1,
        )

    final["mes"] = final["Fecha"].dt.month
    final["ano"] = final["Fecha"].dt.year
    monthly = (
        final.groupby(["mes", "ano"])
        .agg(
            Fecha=("Fecha", "first"),
            caudal_dqo_suma=("Caudal por DQO", "sum"),
            caudal_dqo_promedio=("Caudal por DQO", "mean"),
        )
        .reset_index(drop=True)
    )
    monthly["mes"] = monthly["Fecha"].dt.month
    monthly["ano"] = monthly["Fecha"].dt.year
    final = final.drop(columns=["dias"]).merge(monthly, on=["mes", "ano"], how="left")
    final = final.rename(columns={"Fecha_x": "Fecha"}).drop(columns=["Fecha_y", "mes", "ano"])
    final["desviacion_caudal_dqo"] = final["Caudal por DQO"] / final["caudal_dqo_promedio"].replace(0, np.nan)
    final["desviacion_caudal_dqo"] = final["desviacion_caudal_dqo"].replace([np.inf, -np.inf], np.nan).fillna(0)

    for chem_col in CHEMICAL_COLUMNS_RAW:
        final[chem_col] = final[chem_col] * final["desviacion_caudal_dqo"]

    output_dir.mkdir(parents=True, exist_ok=True)
    final.to_csv(output_dir / "frc40_full_app.csv", index=False, encoding="utf-8-sig")
    return final

