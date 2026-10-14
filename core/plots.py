import pandas as pd
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from core.constants import WL_COLORS, LABELS

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
    
if __name__ == "__main__":
    from core.data_manager import load_file
    
    # 1. Gerçek veriyi yükle
    df = load_file("data/nba_games_clean.csv")
    
    # 2. Test için bir figür ve boş bir eksen (ax) üret
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 3. Yazdığın fonksiyonu çağır (tuvali fonksiyona paslıyoruz)
    plot_result_counts(df, ax)
    
    # 4. İster ekranda gör, ister PNG olarak diske kaydet
    plt.tight_layout()
    plt.savefig("outputs/test_result_counts.png")  # outputs altına kaydeder
    plt.show()  # Pencere açıp ekranda gösterir

