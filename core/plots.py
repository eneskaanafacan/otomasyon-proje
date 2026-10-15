import pandas as pd
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from core.constants import WL_COLORS, LABELS, HOME_AWAY_TEAM_COLORS, NUM_COLS
import seaborn as sns
import numpy as np
from core.data_manager import add_target

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
    
def plot_home_vs_away(df: pd.DataFrame, stat: str, ax: Axes):
    home = f"{stat}_home"
    away = f"{stat}_away"
        
    sns.histplot(
        data=df[home],
        stat="density",
        element="step",
        fill=True,
        alpha=0.3,
        color=HOME_AWAY_TEAM_COLORS["H"],
        label="Ev Sahibi",
        ax=ax
    )
    
    sns.histplot(
        data=df[away],
        stat="density",
        element="step",
        fill=True,
        alpha=0.3,
        color=HOME_AWAY_TEAM_COLORS["A"],
        label="Deplasman",
        ax=ax
    )
    
    ax.legend()
    etiket = LABELS.get(stat, stat)
    ax.set_title(f"Ev Sahibi vs Deplasman {stat}")
    ax.set_xlabel(stat)
    ax.set_ylabel("Yoğunluk(Density)")
    
def plot_scatter(df: pd.DataFrame, x: str, y: str, ax: Axes, n: int = 3000):
    sample_df = df.sample(min(n, len(df)), random_state=42)
    
    sns.scatterplot(
        data=sample_df,
        x=x,
        y=y,
        hue="wl_home",
        palette=WL_COLORS,
        alpha=0.4,
        s=10,
        ax=ax
    )      
    
    etiket_x = LABELS.get(x, x)
    etiket_y = LABELS.get(y, y)
    ax.set_title(f"{etiket_x} vs {etiket_y}")
    ax.set_xlabel(etiket_x)
    ax.set_ylabel(etiket_y)
    
    ax.set_yticks(np.arange(0.25, 0.75, 0.05))
    ax.set_xticks(np.arange(0.25, 0.75, 0.05))
    ax.plot([0.20, 0.70],[0.20, 0.70])    
    
def plot_corr(df: pd.DataFrame, ax: Axes, annot: bool=False):
    df_corr = df.copy()
    df_corr = add_target(df_corr)
    
    cols = []
    for c in NUM_COLS:
        if c in df_corr.columns:
            cols.append(c)
    cols.append("home_win")
    
    corr_matrix = df_corr[cols].corr()
    
    sns.heatmap(
        data=corr_matrix,
        cmap="coolwarm",
        center=0,
        vmax=1,
        vmin=-1,
        annot=annot,
        annot_kws={"size":5} if annot else None,
        ax=ax
    )
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=8)

    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

    ax.set_title("Isı Haritası")
    
    
if __name__ == "__main__":
    from core.data_manager import load_file

    df = load_file("data/nba_games_clean.csv")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Isı haritasını test et
    plot_corr(df, ax, annot=False)

    plt.tight_layout()
    plt.savefig("outputs/test_heatmap.png")
    print("Grafik kaydedildi: outputs/test_heatmap.png")