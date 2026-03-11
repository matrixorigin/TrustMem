"""DbConsumer — base class for components that need short-lived DB sessions."""

from contextlib import contextmanager
from typing import Callable, Iterator

from sqlalchemy.orm import Session

# Canonical type alias — use everywhere instead of bare ``Callable``.
DbFactory = Callable[[], Session]


class DbConsumer:
    """Base for components that acquire DB sessions on demand.

    Each ``_db()`` call creates a fresh session from the factory and closes
    it when the block exits::

        class MyComponent(DbConsumer):
            def do_work(self):
                with self._db() as db:
                    db.execute(...)
                    db.commit()
    """

    def __init__(self, db_factory: Callable[[], Session]):
        if not callable(db_factory):
            raise TypeError(f"db_factory must be callable, got {type(db_factory).__name__}")
        self._db_factory = db_factory

    @contextmanager
    def _db(self) -> Iterator[Session]:
        db = self._db_factory()
        try:
            yield db
        except Exception:
            try:
                db.rollback()
            except Exception:
                pass
            raise
        finally:
            try:
                db.close()
            except Exception:
                pass
