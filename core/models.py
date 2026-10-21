import pandas as pd
from core.data_manager import add_target, add_feature
from core.constants import NUM_COLS, HOME_COLS, AWAY_COLS, DERIVED_COLS
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier

from core.model_factories import (
    create_logistic_regression,
    create_knn,
    create_decision_tree,
    create_random_forest,
    create_gradient_boosting
)

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

MODELS = {
    "Logistic Regression": {
        "factory": create_logistic_regression,
        "params": [("C", 0.01, 10.0, 1.0, float)]
    },
    "KNN": {
        "factory": create_knn,
        "params": [("n_neighbors", 1, 50, 25, int)]
    },
    "Decision Tree": {
        "factory": create_decision_tree,
        "params": [("max_depth", 1, 20, 5, int)]
    },
    "Random Forest": {
        "factory": create_random_forest,
        "params": [
            ("n_estimators", 10, 300, 200, int),
            ("max_depth", 2, 20, 10, int)
        ]
    },
    "Gradient Boosting": {
        "factory": create_gradient_boosting,
        "params": [
            ("n_estimators", 10, 300, 100, int),
            ("learning_rate", 0.01, 0.5, 0.1, float)
        ]
    }
}

