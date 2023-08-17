import re
from typing import List

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WordPunctTokenizer


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
