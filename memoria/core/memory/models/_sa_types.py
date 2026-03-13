"""SQLAlchemy types and constants for memory ORM models.

Self-contained so the memory module has no dependency on api/ layer.
"""

import os
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import JSON as _SA_JSON
from sqlalchemy import DateTime as _SA_DateTime
from sqlalchemy.engine import Dialect
from sqlalchemy.types import TypeDecorator

EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM") or "1024")


class DateTime6(TypeDecorator):
    """Microsecond-precision DATETIME for MatrixOne.

    Automatically strips timezone info before writing to DB,
    since MatrixOne does not accept ISO 8601 timezone suffixes.
    """

    impl = _SA_DateTime
    cache_ok = True

    def get_col_spec(self, **kw: Any) -> str:
        return "DATETIME(6)"

    def process_bind_param(self, value: Any | None, dialect: Dialect) -> Any | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            if value.tzinfo is not None:
                value = value.astimezone(timezone.utc).replace(tzinfo=None)
        return value


class NullableJSON(TypeDecorator):
    """JSON type that stores Python None as SQL NULL, not JSON 'null'."""

    impl = _SA_JSON
    cache_ok = True

    def bind_processor(self, dialect: Dialect):
        impl_processor = self.impl_instance.bind_processor(dialect)

        def process(value: Any | None) -> str | None:
            if value is None:
                return None
            return impl_processor(value) if impl_processor else value

        return process
