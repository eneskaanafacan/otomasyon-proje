import pandas as pd
from core.data_manager import add_target, add_feature
from core.constants import NUM_COLS, HOME_COLS, AWAY_COLS, DERIVED_COLS

def prepare_xy(
    df: pd.DataFrame,
    feature_group: str="Tümü",
    use_derived: bool=False,
    use_playoff: bool=False 
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    df_work = df.copy()
    df_work = add_target(df_work)
    y = df_work["home_win"]
    
    selected_cols = []
    
    if feature_group == "Tümü":
        selected_cols.extend(NUM_COLS)
    elif feature_group == "Sadece ev sahibi":
        selected_cols.extend(HOME_COLS)
    elif feature_group == "Sadece deplasman":
        selected_cols.extend(AWAY_COLS)
    else:
        raise ValueError(f"Geçersiz özellik grubu: {feature_group}")
        
    if use_derived:
        df_work = add_feature(df_work)  
        selected_cols.extend(DERIVED_COLS)
        
    if use_playoff:
        df_work["playoff"] = (df_work["season_type"] == "Playoffs").astype(int)
        selected_cols.append("playoff")
    
    X = df_work[selected_cols].copy()
    
    if "home_win" in X.columns or "wl_home" in X.columns:
        raise ValueError("HATA: Hedef değişkenler X içerisine sızmış!")
    
    feature_names = list(X.columns)
    
    return X, y, feature_names

