"""Custom SQLAlchemy types for MatrixOne compatibility."""

from typing import Any

from sqlalchemy import JSON as _SA_JSON
from sqlalchemy import DateTime as _SA_DateTime
from sqlalchemy.engine import Dialect
from sqlalchemy.types import TypeDecorator


class DateTime6(TypeDecorator):
    """Microsecond-precision DATETIME for MatrixOne.

    SQLAlchemy's generic ``DateTime(timezone=...)`` ignores fractional-second
    precision — it always emits ``DATETIME`` in DDL.  MySQL's dialect-specific
    ``DATETIME(fsp=6)`` works for MySQL but MatrixOne's SQLAlchemy dialect
    doesn't register a ``visit_DATETIME`` compiler, so it fails at DDL time.

    This TypeDecorator wraps the generic DateTime and overrides DDL rendering
    to emit ``DATETIME(6)`` — compatible with both MatrixOne and MySQL dialects.
    """

    impl = _SA_DateTime
    cache_ok = True

    def get_col_spec(self, **kw: Any) -> str:
        return "DATETIME(6)"


class NullableJSON(TypeDecorator):
    """JSON type that stores Python None as SQL NULL, not JSON 'null'.

    MatrixOne's MySQL-compatible dialect serialises Python None to the
    JSON literal ``null`` via the impl's bind_processor.  This wrapper
    short-circuits that: when the value is None we return None directly
    (SQL NULL) and only delegate to the impl for real values.
    """

    impl = _SA_JSON
    cache_ok = True

    def bind_processor(self, dialect: Dialect):
        impl_processor = self.impl_instance.bind_processor(dialect)

        def process(value: Any | None) -> str | None:
            if value is None:
                return None  # SQL NULL, not JSON 'null'
            return impl_processor(value) if impl_processor else value

        return process
