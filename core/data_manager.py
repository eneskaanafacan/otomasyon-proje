import os
import pandas as pd

def veri_yukle(dosya_yolu: str):
    try:
        df = pd.read_csv(dosya_yolu)
        print("Veri başarıyla yüklendi:\n", df.shape)
        return df
    except Exception as e:
        print(f"Veri yüklenirken hata oluştu: {e}")
        return None

data_set_path = "../nba_games_clean.csv"

veri = veri_yukle(data_set_path)