"""Database engine and session management.

Supports SQLite (dev), PostgreSQL, and PostgreSQL + TimescaleDB (production).
TimescaleDB hypertables are auto-created for time-series tables when detected.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.common.config import get_settings
from src.common.logging import get_logger

logger = get_logger(__name__)

_engine: Engine | None = None
_SessionFactory: sessionmaker | None = None

# Tables that should become TimescaleDB hypertables (table_name -> time_column)
TIMESERIES_TABLES: dict[str, str] = {
    "price_bars_1h": "timestamp",
    "cg_funding_1h": "timestamp",
    "cg_oi_1h": "timestamp",
    "cg_liquidations_1h": "timestamp",
    "cg_long_short_1h": "timestamp",
    "cg_taker_flow_1h": "timestamp",
    "feature_store_1h": "timestamp",
    "labels_1h": "timestamp",
    "regimes_1h": "timestamp",
    "signals": "timestamp",
    "model_live_scores": "timestamp",
    "account_snapshots": "timestamp",
    "fills": "timestamp",
    "orders": "timestamp",
}


def _is_postgres(url: str) -> bool:
    return url.startswith("postgresql") or url.startswith("postgres://")


def get_engine() -> Engine:
    """Get or create the SQLAlchemy engine."""
    global _engine
    if _engine is None:
        settings = get_settings()
        url = settings.database_url
        connect_args: dict = {}

        if url.startswith("sqlite"):
            connect_args["check_same_thread"] = False
            _engine = create_engine(
                url,
                echo=False,
                pool_pre_ping=True,
                connect_args=connect_args,
            )
            # Enable WAL mode for SQLite
            @event.listens_for(_engine, "connect")
            def _set_sqlite_pragma(dbapi_conn, connection_record):  # type: ignore
                cursor = dbapi_conn.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()

            logger.info("database_engine_created", backend="sqlite")

        elif _is_postgres(url):
            _engine = create_engine(
                url,
                echo=False,
                pool_pre_ping=True,
                pool_size=10,
                max_overflow=20,
                pool_recycle=3600,
            )
            logger.info("database_engine_created", backend="postgresql")

        else:
            _engine = create_engine(
                url,
                echo=False,
                pool_pre_ping=True,
                connect_args=connect_args,
            )
            logger.info("database_engine_created", backend="other")

    return _engine


def get_session_factory() -> sessionmaker:
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _SessionFactory


def get_session() -> Session:
    """Create a new session."""
    return get_session_factory()()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """Provide a transactional scope around a series of operations."""
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _has_timescaledb(engine: Engine) -> bool:
    """Check if TimescaleDB extension is available."""
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT 1 FROM pg_extension WHERE extname = 'timescaledb'")
            )
            return result.scalar() is not None
    except Exception:
        return False


def _enable_timescaledb(engine: Engine) -> None:
    """Enable TimescaleDB extension if not already enabled."""
    try:
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE"))
            conn.commit()
        logger.info("timescaledb_extension_enabled")
    except Exception as e:
        logger.warning("timescaledb_extension_failed", error=str(e))


def _create_hypertables(engine: Engine) -> None:
    """Convert time-series tables to TimescaleDB hypertables.

    Idempotent: skips tables that are already hypertables.
    Uses migrate_data => true to handle existing data.
    """
    with engine.connect() as conn:
        for table_name, time_col in TIMESERIES_TABLES.items():
            try:
                # Check if table exists
                exists = conn.execute(
                    text(
                        "SELECT 1 FROM information_schema.tables "
                        "WHERE table_name = :tbl"
                    ),
                    {"tbl": table_name},
                ).scalar()
                if not exists:
                    continue

                # Check if already a hypertable
                is_hyper = conn.execute(
                    text(
                        "SELECT 1 FROM timescaledb_information.hypertables "
                        "WHERE hypertable_name = :tbl"
                    ),
                    {"tbl": table_name},
                ).scalar()
                if is_hyper:
                    continue

                # Convert to hypertable
                conn.execute(
                    text(
                        f"SELECT create_hypertable(:tbl, :col, "
                        f"migrate_data => true, if_not_exists => true)"
                    ),
                    {"tbl": table_name, "col": time_col},
                )
                logger.info(
                    "hypertable_created",
                    table=table_name,
                    time_column=time_col,
                )
            except Exception as e:
                logger.warning(
                    "hypertable_creation_skipped",
                    table=table_name,
                    error=str(e),
                )
        conn.commit()


def _setup_timescale_policies(engine: Engine) -> None:
    """Set up TimescaleDB compression and retention policies."""
    with engine.connect() as conn:
        # Enable compression on high-volume tables with 7-day policy
        compress_tables = [
            "price_bars_1h", "cg_funding_1h", "cg_oi_1h",
            "cg_liquidations_1h", "feature_store_1h",
        ]
        for tbl in compress_tables:
            try:
                is_hyper = conn.execute(
                    text(
                        "SELECT 1 FROM timescaledb_information.hypertables "
                        "WHERE hypertable_name = :tbl"
                    ),
                    {"tbl": tbl},
                ).scalar()
                if not is_hyper:
                    continue

                conn.execute(
                    text(
                        f"ALTER TABLE {tbl} SET ("
                        f"timescaledb.compress, "
                        f"timescaledb.compress_segmentby = 'symbol'"
                        f")"
                    )
                )
                conn.execute(
                    text(
                        f"SELECT add_compression_policy(:tbl, INTERVAL '7 days', "
                        f"if_not_exists => true)"
                    ),
                    {"tbl": tbl},
                )
                logger.info("compression_policy_added", table=tbl)
            except Exception as e:
                logger.debug("compression_policy_skipped", table=tbl, error=str(e))

        # Retention: drop raw data older than 2 years for non-essential tables
        retention_tables = {
            "cg_liquidations_1h": "730 days",
            "cg_taker_flow_1h": "730 days",
            "cg_long_short_1h": "730 days",
            "model_live_scores": "365 days",
            "account_snapshots": "365 days",
        }
        for tbl, interval in retention_tables.items():
            try:
                is_hyper = conn.execute(
                    text(
                        "SELECT 1 FROM timescaledb_information.hypertables "
                        "WHERE hypertable_name = :tbl"
                    ),
                    {"tbl": tbl},
                ).scalar()
                if not is_hyper:
                    continue

                conn.execute(
                    text(
                        f"SELECT add_retention_policy(:tbl, INTERVAL :interval, "
                        f"if_not_exists => true)"
                    ),
                    {"tbl": tbl, "interval": interval},
                )
                logger.info("retention_policy_added", table=tbl, interval=interval)
            except Exception as e:
                logger.debug("retention_policy_skipped", table=tbl, error=str(e))

        conn.commit()


def init_db() -> None:
    """Create all tables and configure TimescaleDB if on PostgreSQL.

    For dev/testing with SQLite, simply creates tables.
    For production PostgreSQL, also:
      1. Enables TimescaleDB extension
      2. Converts time-series tables to hypertables
      3. Sets up compression and retention policies
    """
    from src.storage.models import Base

    engine = get_engine()
    Base.metadata.create_all(engine)
    logger.info("database_tables_created")

    # TimescaleDB setup for PostgreSQL
    url = get_settings().database_url
    if _is_postgres(url):
        _enable_timescaledb(engine)
        if _has_timescaledb(engine):
            _create_hypertables(engine)
            _setup_timescale_policies(engine)
            logger.info("timescaledb_setup_complete")
        else:
            logger.info(
                "timescaledb_not_available",
                hint="Install TimescaleDB for optimal time-series performance. "
                     "Falling back to standard PostgreSQL.",
            )
