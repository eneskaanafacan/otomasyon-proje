from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def create_logistic_regression(C=1.0, random_state=42):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=C, max_iter=2000, random_state=random_state))
    ])

def create_knn(n_neighbors=25):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=n_neighbors))
    ])

def create_decision_tree(max_depth=5, random_state=42):
    return Pipeline([
        ("clf", DecisionTreeClassifier(max_depth=max_depth, random_state=random_state))
    ])

def create_random_forest(n_estimators=200, max_depth=10, random_state=42):
    return Pipeline([
        ("clf", RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1, random_state=random_state))
    ])

def create_gradient_boosting(n_estimators=100, learning_rate=0.1, random_state=42):
    return Pipeline([
        ("clf", GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state))
    ])