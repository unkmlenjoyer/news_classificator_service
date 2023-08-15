# %%
import re
import string
from typing import List
from unittest import result

import nltk

nltk.download("stopwords")
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from src.utils import TextPreprocess

# %%
data = pd.read_json("../data/News_Category_Dataset_v3.json", lines=True)

# %%
formatter = TextPreprocess()

# %%
data["cleaned"] = data["headline"].apply(lambda x: formatter.process_text(x))
# %%
data = data[data.category.isin(data.category.value_counts().nlargest(10).index.values)]

# %%
categories = data["category"].unique()
category2idx = {ctg: i for i, ctg in enumerate(categories)}
idx2category = {i: ctg for i, ctg in enumerate(categories)}

data["category"] = data["category"].map(category2idx)

X, y = data["cleaned"], data["category"]

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_train, y_train, stratify=y_train, test_size=0.2, random_state=42, shuffle=True
)

# %%
pipe = Pipeline(
    steps=[
        ("vectorizer", TfidfVectorizer()),
        ("clf", OneVsRestClassifier(estimator=LogisticRegression(max_iter=1000))),
    ]
)

# %%
f1_macro = make_scorer(f1_score, average="macro")
param_grid = {
    "clf__estimator__C": [0.2, 0.3, 0.6],
    "vectorizer__max_features": [2000, 8000, 10000],
}
# %%
gsv = GridSearchCV(
    pipe,
    param_grid=param_grid,
    scoring=f1_macro,
    cv=StratifiedKFold(5),
    return_train_score=True,
    verbose=3,
)

# %%
gsv.fit(X_train, y_train)

# %%
results = pd.DataFrame(gsv.cv_results_)

# %%
results[["params", "mean_test_score", "mean_train_score"]].query(
    "abs(mean_test_score - mean_train_score) < 0.03"
)
# %%
results.iloc[3].params
# %%
pipe.set_params(**{"clf__estimator__C": 0.3, "vectorizer__max_features": 2000})
# %%
pipe.fit(X_train, y_train)
# %%
f1_score(y_valid, pipe.predict(X_valid.values), average="macro")

# %%
f1_score(y_test, pipe.predict(X_test.values), average="macro")
# %%
from sklearn.metrics import classification_report

# %%
print(classification_report(y_valid, pipe.predict(X_valid.values)))
# %%
import joblib

joblib.dump(pipe, "tfidf_logreg.pkl")
# %%
