STAT_PAIRS = [
    "fg_pct",
    "fg3_pct",
    "ft_pct",
    "oreb",
    "dreb",
    "ast",
    "stl",
    "blk",
    "tov",
    "pf"
]

HOME_COLS = []
AWAY_COLS = []

for stat in STAT_PAIRS:
    HOME_COLS.append(f"{stat}_home")
    AWAY_COLS.append(f"{stat}_away")
    
NUM_COLS = HOME_COLS + AWAY_COLS

LABELS = {
    "fg_pct": "Saha içi şut %",
    "fg3_pct": "3 sayı %",
    "ft_pct": "Serbest atış %",
    "oreb": "Hücum ribaundu",
    "dreb": "Savunma ribaundu",
    "ast": "Asist",
    "stl": "Top çalma",
    "blk": "Blok",
    "tov": "Top kaybı",
    "pf": "Faul"
}

DERIVED_COLS = [
    "fg_pct_diff",
    "fg3_pct_diff",
    "ft_pct_diff",
    "ast_diff",
    "stl_diff",
    "blk_diff",
    "tov_diff",
    "pf_diff",
    "reb_home",
    "reb_away",
    "reb_diff",
]

WL_COLORS = {
    "W": "#2ca02c",  # Galibiyet (Yeşil)
    "L": "#d62728",  # Mağlubiyet (Kırmızı)
}

HOME_AWAY_TEAM_COLORS = {
    "H": "#1f77b4",  # Galibiyet (Yeşil)
    "A": "#ff7f0e",  # Mağlubiyet (Kırmızı)
}