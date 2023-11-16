# %% [markdown]
"""Script for preprocessing text before modelling"""

# %% [markdown]
"""
1. Import libraries
"""

import catboost
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from src.config import ClassifierResearchConfig
from src.utils import TextPreprocess

# %% [markdown]
"""
2. Load config and define constants
"""

config = ClassifierResearchConfig()
TRAIN_TASK = False

# %% [markdown]
"""
3. Load cleaned data and split into train, valid, test chunks
"""

# %%
data = pd.read_feather(config.CLEAN_DATA_PATH + "cleaned_title.feather")

# %%
data = data[data.date > "2014/01/01"]

# %%
X, y = data["cleaned_title"], data["topic"]

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42, shuffle=True
)

X_valid, X_test, y_valid, y_test = train_test_split(
    X_test, y_test, stratify=y_test, test_size=0.2, random_state=42, shuffle=True
)

if TRAIN_TASK:
    vectorizer = TfidfVectorizer(max_features=10000)

    train_x_tr = vectorizer.fit_transform(X_train)
    val_x_tr = vectorizer.transform(X_valid)

    train_pool = catboost.Pool(train_x_tr, y_train)
    valid_pool = catboost.Pool(val_x_tr, y_valid)

    clf = catboost.CatBoostClassifier(
        task_type="CPU",
        iterations=400,
        loss_function="MultiClass",
        eval_metric="TotalF1:average=Macro",
        auto_class_weights="SqrtBalanced",
        random_seed=config.RANDOM_SEED,
        early_stopping_rounds=5,
        use_best_model=True,
    )

    clf.fit(train_pool, eval_set=valid_pool, plot=True)

else:
    vectorizer = joblib.load(config.ARTIFACT_PATH + "tfidf_vect_10k.pkl")
    clf = catboost.CatBoostClassifier().load_model(config.ARTIFACT_PATH + "ctb_clf_v1")

# %%
processor = TextPreprocess()

# %%
res = processor.process_text(
    "США ввели санкции против Европы за ввод санкций простив США"
)

# %%
clf.predict(
    vectorizer.transform([res]),
    prediction_type="Probability",
).ravel()
# %%
idx2topic = joblib.load("../artifacts/idx2topic.pkl")

# %%
test_data = vectorizer.transform(X_test)

y_test_pred = clf.predict(test_data)

f1_score(y_test, y_test_pred, average="macro")
