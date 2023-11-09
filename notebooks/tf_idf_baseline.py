# %% [markdown]
"""Script for preprocessing text before modelling"""

# %% [markdown]
"""
1. Import libraries
"""

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from src.config import ClassifierResearchConfig

# %% [markdown]
"""
2. Load config and define constants
"""

config = ClassifierResearchConfig()

USE_GRID_SEARCH = False

# %% [markdown]
"""
3. Load cleaned data and split into train, valid, test chunks
"""

# %%
data = pd.read_feather(config.CLEAN_DATA_PATH + "cleaned_title.feather")

# %%
X, y = data["cleaned_title"], data["topic"]

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42, shuffle=True
)

X_valid, X_test, y_valid, y_test = train_test_split(
    X_test, y_test, stratify=y_test, test_size=0.2, random_state=42, shuffle=True
)

# %% [markdown]
"""
3. Define model pipe and use grid search in order to find optimap hyperparams
"""

# %%
pipe = Pipeline(
    steps=[
        ("vectorizer", TfidfVectorizer()),
        (
            "clf",
            OneVsRestClassifier(
                estimator=LogisticRegression(max_iter=1000, class_weight="balanced")
            ),
        ),
    ]
)

# {"clf__estimator__C": 0.1, "vectorizer__max_features": 10000}
#

# %%
if USE_GRID_SEARCH:
    f1_macro = make_scorer(f1_score, average="macro")
    param_grid = {
        "vectorizer__ngram_range": [(1, 1)],
        "clf__estimator__C": [0.01, 0.05, 0.1, 0.2, 0.3, 0.6],
        "vectorizer__max_features": [2000, 8000, 10000],
    }

    gsv = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring=f1_macro,
        cv=StratifiedKFold(5),
        return_train_score=True,
        verbose=3,
    )

    gsv.fit(X_train, y_train)

    results = pd.DataFrame(gsv.cv_results_)

    print(
        results[
            [
                "params",
                "mean_train_score",
                "std_train_score",
                "mean_test_score",
                "std_test_score",
                "rank_test_score",
            ]
        ].query("abs(mean_test_score - mean_train_score) < 0.025")
    )

# %%
pipe.set_params(**{"clf__estimator__C": 0.1, "vectorizer__max_features": 10000})
# %%
pipe.fit(X_train, y_train)
# %%
f1_score(y_valid, pipe.predict(X_valid.values), average="macro")

# %%
f1_score(y_test, pipe.predict(X_test.values), average="macro")

# %%
print(classification_report(y_valid, pipe.predict(X_valid.values)))

# %%
print(classification_report(y_test, pipe.predict(X_test.values)))

# %%
joblib.dump(pipe, config.ARTIFACT_PATH + "tf_idf_base.pkl")
