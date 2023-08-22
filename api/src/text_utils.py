"""Useful utils to process / format text"""
import re
from typing import List

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WordPunctTokenizer


class TextPreprocess:
    """Class for text preprocessing

    To run full text pipeline use `process_text` method

    Attributes
    ----------
    tokenizer : WordPunctTokenizer
        Text tokenizer (splitter)

    lemmatizer : WordNetLemmatizer
        Text lemmatizer (normalizer)

    reg_exp : re.Pattern
        Regular expression to clean text

    stop_words : List[str]


    """

    def __init__(self):
        """Initialize text's preprocessor"""
        self.tokenizer: WordPunctTokenizer = WordPunctTokenizer()
        self.lemmatizer: WordNetLemmatizer = WordNetLemmatizer()
        self.reg_exp: re.Pattern = re.compile("[^A-Za-z]")
        self.stop_words: List[str] = stopwords.words("english")

    @staticmethod
    def is_empty(text: str) -> bool:
        """Method to chekc is string is empty

        Parameters
        ----------
        text : str
            Text to process

        Returns
        -------
        bool
            Is the text empty
        """
        return text == ""

    def get_pruned_text(self, text: str) -> str:
        """Method to get pruned (cleaned from other symbols) text

        Parameters
        ----------
        text : str
            Text to prune

        Returns
        -------
        str
            Pruned text
        """
        return self.reg_exp.sub(" ", text)

    def get_replaced_form(self, text: str, target: str, symb: str) -> str:
        """Method to replace symbols in text

        Parameters
        ----------
        text : str
            Text to process
        target : str
            Symbol to change
        symb : str
            Symbol to place

        Returns
        -------
        str
            New text
        """
        return text.replace(target, symb)

    def get_strip_form(self, text: str) -> str:
        """Method to get stripped text

        Parameters
        ----------
        text : str
            Text to process

        Returns
        -------
        str
            Stripped text
        """
        return text.strip()

    def get_lower_form(self, text: str) -> str:
        """Method to get lowered form of text

        Parameters
        ----------
        text : str
            Text to process

        Returns
        -------
        str
            Lowered text
        """
        return text.lower()

    def get_tokenized_form(self, text: str) -> List[str]:
        """Method to get tokenized form of text

        Parameters
        ----------
        text : str
            Text to process

        Returns
        -------
        List[str]
            Tokens of text
        """
        return self.tokenizer.tokenize(text)

    def get_normal_form(self, text: List[str]) -> List[str]:
        """Method to normalize text's tokens

        Parameters
        ----------
        text : List[str]
            Tokens to normalize (lemmatize)

        Returns
        -------
        List[str]
            Normalized tokens
        """
        return [self.lemmatizer.lemmatize(word) for word in text]

    def filter_words(self, text: List[str]) -> List[str]:
        """Method to filter text from stop words

        Parameters
        ----------
        text : List[str]
            Tokens to filter

        Returns
        -------
        List[str]
            Filtered tokens
        """
        return [word for word in text if word not in self.stop_words]

    def process_text(self, text: str) -> str:
        """Method to process full pipeline

        Parameters
        ----------
        text : str
            Text to process

        Returns
        -------
        str
            Cleaned text
        """
        pruned = self.get_pruned_text(text)
        striped = self.get_strip_form(pruned)
        replaced = self.get_replaced_form(striped, "  ", " ")
        lowered = self.get_lower_form(replaced)
        tokenized = self.get_tokenized_form(lowered)
        normalized = self.get_normal_form(tokenized)
        filtered = self.filter_words(normalized)

        return " ".join(filtered)
