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

