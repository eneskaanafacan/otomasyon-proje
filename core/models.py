import pandas as pd
from core.data_manager import add_target, add_feature
from core.constants import NUM_COLS, HOME_COLS, AWAY_COLS, DERIVED_COLS
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier

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

def split_data(
    X:  pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    
    X_train, X_test, y_train , y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def get_baseline_score(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> float:
    
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(X_train, y_train)
    return dummy_clf.score(X_test, y_test)

if __name__ == "__main__":
    from core.data_manager import load_file
    
    dataset = load_file("data/nba_games_clean.csv")
    
    X, y, feature_names = prepare_xy(dataset)
    X_train, X_test, y_train, y_test = split_data(X,y)
    print(f"X_train boyutu: {X_train.shape}, y_train boyutu: {y_train.shape}")
    print(f"X_test boyutu: {X_test.shape}, y_test boyutu: {y_test.shape}")

    baseline_acc = get_baseline_score(X_train, y_train, X_test, y_test)
    print(f"Baseline Accuracy (Dummy): {baseline_acc:.4f}")