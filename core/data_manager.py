import os
import pandas as pd
from core.constants import NUM_COLS, DERIVED_COLS

REQUIRED_COLS = NUM_COLS.copy()
REQUIRED_COLS.append("wl_home")
REQUIRED_COLS.append("season_type")

def load_file(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="cp1254")
        
    control(df)
    
    df["wl_home"] = df["wl_home"].astype("category")
    df["season_type"] = df["season_type"].astype("category")
    
    return df

def control(dataset: pd.DataFrame):
    missing_cols = []
    for c in REQUIRED_COLS:
        if c not in dataset.columns:
            missing_cols.append(c)
    if missing_cols:
        raise ValueError(f"Bu dosya doğru formatta değil. Eksik sütunlar var: {missing_cols}")
    
def add_target(dataset: pd.DataFrame) -> pd.DataFrame:
    dataset["home_win"] = (dataset["wl_home"] == "W").astype(int)
    return dataset
        
def summarize(df: pd.DataFrame) -> dict:
    sozluk = {}
    sozluk["Satır sayısı"] = df.shape[0]
    sozluk["Sütun sayısı"] = df.shape[1]
    sozluk["Galibiyet oranı"] = df["home_win"].mean() * 100
    sozluk["Playoff maç sayısı"] = int((df["season_type"] == "Playoffs").sum())
    sozluk["Toplam eksik değer sayısı"] = int(df.isna().sum().sum())
    sozluk["Sayısal özet tablosu"] = df[NUM_COLS].describe().T.round(3)
    sozluk["Kazanan ev sahibi takım istatisikleri"] = df.groupby('wl_home')[NUM_COLS].mean().T
    sozluk["Tip ve eksik değer tablosu"] = pd.DataFrame({
        "Veri tipi": df.dtypes,
        "Eksik değer sayısı": df.isna().sum()
    })
    return sozluk

def add_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in DERIVED_COLS:
        if c.startswith("reb"):
            continue
        df[c] = df[f"{c[:-5]}_home"] - df[f"{c[:-5]}_away"]
        
    df["reb_home"] = df["oreb_home"] + df["dreb_home"]
    df["reb_away"] = df["oreb_away"] + df["dreb_away"]
    df["reb_diff"] = df["reb_home"] - df["reb_away"]
    
    return df
    