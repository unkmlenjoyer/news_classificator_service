"""Custom exceptions"""

from typing import Any, Dict, Union

from fastapi import HTTPException


class EmptyModelInputException(HTTPException):
    """Customized exception. Raises when input to model is empty.

    Example: ''.
    """

    def __init__(
        self,
        status_code: int,
        detail: Any = None,
        headers: Union[Dict[str, str], None] = None,
    ) -> None:
        super().__init__(status_code, detail, headers)


class NewsNotFound(HTTPException):
    """Customized exception. Raises when can't find news with specific ID from DB"""

    def __init__(
        self,
        status_code: int,
        detail: Any = None,
        headers: Union[Dict[str, str], None] = None,
    ) -> None:
        super().__init__(status_code, detail, headers)


class NewsNotInsertedException(HTTPException):
    """Customized exception. Raises when can't insert news into DB"""

    def __init__(
        self,
        status_code: int,
        detail: Any = None,
        headers: Union[Dict[str, str], None] = None,
    ) -> None:
        super().__init__(status_code, detail, headers)
