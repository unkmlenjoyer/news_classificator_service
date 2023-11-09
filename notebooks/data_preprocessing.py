# %% [markdown]
"""Script for preprocessing text before modelling"""

# %% [markdown]
"""
1. Import libraries
"""

import joblib
import pandas as pd
import seaborn as sns
from src.config import ClassifierResearchConfig
from src.utils import TextPreprocess
from tqdm.notebook import tqdm

# %% [markdown]
"""
2. Load configs and set options
"""
# in order to .progress_apply
tqdm.pandas()

config = ClassifierResearchConfig()

# text pipeline runner
formatter = TextPreprocess()

# pandas settings
pd.set_option("display.max_rows", config.PD_MAX_ROWS)
pd.set_option("display.max_columns", config.PD_MAX_COLS)

# seaborn settings
sns.set({"figure.figsize": config.SNS_FIG_SIZE})
sns.set_style(config.SNS_STYLE)

# %% [markdown]
"""
3. Load raw data and base clean
"""

data = pd.read_csv(config.RAW_DATA_PATH)
data = data[~data.topic.isna()]

# %% [markdown]
"""
4. Check topic distribution normalized
"""

data.topic.value_counts(normalize=True)

# %%
sns.countplot(data=data, x="topic")

# %%
# There are low frequences topics, let's combine them by remap categories
topic_mapper = {
    "Библиотека": "Дом",
    "Россия": "Россия",
    "Мир": "Мир",
    "Экономика": "Экономика и бизнес",
    "Интернет и СМИ": "Интернет и СМИ",
    "Спорт": "Спорт",
    "Культура": "Культура",
    "Из жизни": "Из жизни",
    "Наука и техника": "Наука и техника",
    "Бывший СССР": "Бывший СССР",
    "Дом": "Дом",
    "Сочи": "Спорт",
    "ЧМ-2014": "Спорт",
    "Путешествия": "Путешествия",
    "Силовые структуры": "Силовые структуры",
    "Ценности": "Из жизни",
    "Легпром": "Другое",
    "Бизнес": "Экономика и бизнес",
    "МедНовости": "Другое",
    "Оружие": "Силовые структуры",
    "69-я параллель": "Другое",
    "Культпросвет ": "Культура",
    "Крым": "Россия",
}

data["topic"] = data["topic"].map(topic_mapper)

# %%
sns.countplot(data=data, x="topic")

# %% [markdown]
"""
5. Encode categories and save idx to topic mapper in order to decode labels
"""

# %%
topics = data.topic.unique()
topic2idx = {topic: i for i, topic in enumerate(topics)}
idx2topic = {i: topic for i, topic in enumerate(topics)}

# %%
data["topic"] = data["topic"].map(topic2idx).astype(int)

# %%
joblib.dump(idx2topic, config.ARTIFACT_PATH + "idx2topic.pkl")

# %% [markdown]
"""
6. Clean and normalize text
"""

# %%
data["cleaned_title"] = data["title"].progress_apply(
    lambda x: formatter.process_text(x)
)

# %% [markdown]
"""
6. Save cleaned text
"""

data.reset_index(drop=True).to_feather(config.CLEAN_DATA_PATH + "cleaned_title.feather")
