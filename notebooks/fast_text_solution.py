# %% [markdown]
"""Script for preprocessing text before modelling"""

# %% [markdown]
"""
1. Import libraries
"""

import fasttext
import joblib
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from src.config import ClassifierResearchConfig

# %% [markdown]
"""
2. Load config and define constants
"""

config = ClassifierResearchConfig()

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
with open("train.txt", "w") as f:
    for title, topic in zip(X_train.values, y_train.values):
        f.writelines(f"__label__{topic} {title}\n")

with open("valid.txt", "w") as f:
    for title, topic in zip(X_valid.values, y_valid.values):
        f.writelines(f"__label__{topic} {title}\n")

# %%
model = fasttext.train_supervised("train.txt", epoch=15, lr=0.05)
# %%
model.test("valid.txt")
# %%
model.predict("Новейшее оружие совершило прорыв", k=10)
# %%
y_pred_valid = (
    X_valid.apply(model.predict).apply(lambda x: x[0][0].split("__")[-1]).astype(int)
)

# %%
print(classification_report(y_valid, y_pred_valid))

# %%
y_pred_test = (
    X_test.apply(model.predict).apply(lambda x: x[0][0].split("__")[-1]).astype(int)
)
# %%
print(classification_report(y_test, y_pred_test))
# %%
joblib.load("../artifacts/idx2topic.pkl")
