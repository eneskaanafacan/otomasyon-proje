import pandas as pd
from core.data_manager import add_target, add_feature
from core.constants import NUM_COLS, HOME_COLS, AWAY_COLS, DERIVED_COLS
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

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

def train_and_evaluate(
    model_name: str,
    params: dict,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
):
    pipeline = MODELS[model_name]["factory"](**params)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_train_pred = pipeline.predict(X_train)
    
    metrics = {
        "train_accuracy": (accuracy_score(y_train, y_train_pred)),
        "accuracy": (accuracy_score(y_test, y_pred)),
        "precision": (precision_score(y_test, y_pred)),
        "recall": (recall_score(y_test, y_pred)),
        "f1": (f1_score(y_test, y_pred)),
        "roc_auc": (roc_auc_score(y_test, y_proba)),
    }
    
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        "pipeline": pipeline,
        "metrics": metrics,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "cm": cm,
        "feature_names": list(X_train.columns)
    }

def compute_cross_val(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: int = 5,
    random_state: int = 42,
) -> tuple[float, float]:
    cv = StratifiedKFold(
        n_splits=cv_splits, shuffle=True, random_state=random_state
    )
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")

    mean_score = float(scores.mean())
    std_score = float(scores.std())

    return mean_score, std_score


def get_importance(result: dict, model_name: str) -> pd.DataFrame | None:
    clf = result["pipeline"].named_steps["clf"]
    features = result["feature_names"]

    if model_name == "Logistic Regression":
        importances = clf.coef_[0]
    elif model_name in ["Decision Tree", "Random Forest", "Gradient Boosting"]:
        importances = clf.feature_importances_
    elif model_name == "KNN":
        print(f"Bilgi: {model_name} modeli için yerleşik özellik önemi desteklenmiyor.")
        return None
    else:
        raise ValueError(f"Geçersiz model adı: {model_name}")

    df_importance = pd.DataFrame({
        "feature": features,
        "importance": importances
    })

    # Mutlak büyüklüğe göre sırala 
    df_importance["abs_importance"] = df_importance["importance"].abs()
    df_importance = df_importance.sort_values(by="abs_importance", ascending=False).drop(columns=["abs_importance"])

    return df_importance.reset_index(drop=True)


def predict_one(
    pipeline: Pipeline,
    feature_names: list[str],
    values: dict,
    default_means: dict | pd.Series | None = None,
) -> tuple[float, int]:

    row_data = {}
    for feat in feature_names:
        if feat in values:
            row_data[feat] = values[feat]
        elif default_means is not None and feat in default_means:
            row_data[feat] = default_means[feat]
        else:
            row_data[feat] = 0.0

    df_single = pd.DataFrame([row_data])[feature_names]

    proba = float(pipeline.predict_proba(df_single)[0, 1])
    pred = int(pipeline.predict(df_single)[0])

    return proba, pred
 
