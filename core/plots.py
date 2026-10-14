import pandas as pd
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from core.constants import WL_COLORS, LABELS
import seaborn as sns
import numpy as np

def plot_result_counts(df: pd.DataFrame, ax: Axes):
    ozet_tablo = pd.crosstab(df["season_type"], df["wl_home"])
    
    ozet_tablo.plot(
        kind='bar',
        ax=ax,
        color=[WL_COLORS["L"], WL_COLORS["W"]],
        rot=0
    )
    
    regular_season = df[df["season_type"] == "Regular Season"]
    toat_game_for_rs = len(regular_season)
    number_of_win_for_rs = (regular_season["wl_home"] == "W").sum() / toat_game_for_rs * 100
    number_of_lose_for_rs = (regular_season["wl_home"] == "L").sum() / toat_game_for_rs * 100

    playoffs = df[df["season_type"] == "Playoffs"]
    toat_game_for_po = len(playoffs)
    number_of_win_for_po = (playoffs["wl_home"] == "W").sum() / toat_game_for_po * 100
    number_of_lose_for_po = (playoffs["wl_home"] == "L").sum() / toat_game_for_po * 100
    
    ax.bar_label(ax.containers[0], labels=[f"%{number_of_lose_for_po:.1f}", f"%{number_of_lose_for_rs:.1f}"], padding=3)
    ax.bar_label(ax.containers[1], labels=[f"%{number_of_win_for_po:.1f}", f"%{number_of_win_for_rs:.1f}"], padding=3)
    
    ax.set_title(("Sezon Türüne Göre Sonuçlar"))
    ax.set_xlabel(("Sezon Türü"))
    ax.set_ylabel(("Maç sayısı"))
    ax.legend(["Mağlubiyet (L)", "Galibiyet (W)"])
    
def plot_hist_by_result(df: pd.DataFrame, col: str, ax: Axes):
    sns.histplot(
        data=df,
        x=col,
        hue="wl_home",
        stat="density",
        common_norm=False,
        element="step",
        bins=30,
        palette=WL_COLORS,
        ax=ax
    )
    ax.set_xticks(np.arange(0.25, 0.75, 0.05))
    etiket = LABELS.get(col, col)
    ax.set_title(f"Sonuca Göre Dağılım: {etiket}")
    ax.set_xlabel(etiket)
    ax.set_ylabel("Yoğunluk (Density)")
    
def plot_box_by_result(df: pd.DataFrame, col: str, ax: Axes):
    sns.boxplot(
        data=df,
        x="wl_home",
        y=col,
        order=["W", "L"],
        palette=WL_COLORS,
        ax=ax
    )
    ax.set_yticks(np.arange(0.25, 0.75, 0.05))
    etiket = LABELS.get(col, col)
    ax.set_title(f"Sonuca Göre Kutu Grafiği: {etiket}")
    ax.set_xlabel("Maç Sonucu")
    ax.set_ylabel(etiket)
    ax.set_xticklabels(["Galibiyet (W)", "Mağlubiyet (L)"])
    
if __name__ == "__main__":
    from core.data_manager import load_file

    df = load_file("data/nba_games_clean.csv")

    fig, ax = plt.subplots(figsize=(8, 6))
    plot_box_by_result(df, "fg_pct_home", ax)

    plt.tight_layout()
    plt.savefig("outputs/test_box_fg_pct.png")
    print("Grafik kaydedildi: outputs/test_box_fg_pct.png")
