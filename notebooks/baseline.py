# %%
import re
import string
from typing import List

import nltk

nltk.download("stopwords")
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WordPunctTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

# %%
data = pd.read_json("data/News_Category_Dataset_v3.json", lines=True)


# %%
class TextPreprocess:
    def __init__(self):
        self.tokenizer = WordPunctTokenizer()
        self.lemmatizer = WordNetLemmatizer()
        self.reg_exp = re.compile("[^A-Za-z]")
        self.stop_words = stopwords.words("english")

    def get_pruned_text(self, text: str):
        return self.reg_exp.sub(" ", text)

    def get_replaced_form(self, text, target: str, symb: str) -> str:
        return text.replace(target, symb)

    def get_strip_form(self, text):
        return text.strip()

    def get_lower_form(self, text: str):
        return text.lower()

    def get_tokenized_form(self, text: str):
        return self.tokenizer.tokenize(text)

    def get_normal_form(self, text: List[str]):
        return [self.lemmatizer.lemmatize(word) for word in text]

    def filter_words(self, text: List[str]) -> List[str]:
        return [word for word in text if word not in self.stop_words]

    def process_text(self, text: str) -> str:
        pruned = self.get_pruned_text(text)
        striped = self.get_strip_form(pruned)
        replaced = self.get_replaced_form(striped, "  ", " ")
        lowered = self.get_lower_form(replaced)
        tokenized = self.get_tokenized_form(lowered)
        normalized = self.get_normal_form(tokenized)
        filtered = self.filter_words(normalized)

        return " ".join(filtered)


# %%
formatter = TextPreprocess()

# %%
data["cleaned"] = data["headline"].apply(lambda x: formatter.process_text(x))
# %%
data = data[data.category.isin(data.category.value_counts().nlargest(10).index.values)]
X, y = data["cleaned"], data["category"]

# %%
X_train, X_text, y_train, y_test = train_test_split(
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
    "clf__estimator__C": [0.1, 0.5, 1.0],
    "vectorizer__max_features": [20000],
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
