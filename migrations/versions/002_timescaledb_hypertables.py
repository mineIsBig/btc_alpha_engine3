"""Add TimescaleDB hypertables for time-series tables.

Revision ID: 002
Revises: 001
Create Date: 2026-03-18 00:00:00.000000

This migration is PostgreSQL + TimescaleDB only.
On SQLite or PostgreSQL without TimescaleDB, it is a no-op.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Tables to convert to hypertables: (table_name, time_column)
HYPERTABLES = [
    ("price_bars_1h", "timestamp"),
    ("cg_funding_1h", "timestamp"),
    ("cg_oi_1h", "timestamp"),
    ("cg_liquidations_1h", "timestamp"),
    ("cg_long_short_1h", "timestamp"),
    ("cg_taker_flow_1h", "timestamp"),
    ("feature_store_1h", "timestamp"),
    ("labels_1h", "timestamp"),
    ("regimes_1h", "timestamp"),
    ("signals", "timestamp"),
    ("model_live_scores", "timestamp"),
    ("account_snapshots", "timestamp"),
    ("fills", "timestamp"),
    ("orders", "timestamp"),
]

# Tables to enable compression on
COMPRESS_TABLES = [
    "price_bars_1h",
    "cg_funding_1h",
    "cg_oi_1h",
    "cg_liquidations_1h",
    "feature_store_1h",
]


def _is_timescaledb() -> bool:
    """Check if we're on PostgreSQL with TimescaleDB."""
    conn = op.get_bind()
    if "sqlite" in str(conn.engine.url):
        return False
    try:
        result = conn.execute(
            sa.text("SELECT 1 FROM pg_extension WHERE extname = 'timescaledb'")
        )
        return result.scalar() is not None
    except Exception:
        return False


def upgrade() -> None:
    conn = op.get_bind()

    # Skip entirely for SQLite
    if "sqlite" in str(conn.engine.url):
        return

    # Try to enable TimescaleDB
    try:
        conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE"))
    except Exception:
        return  # No TimescaleDB available, skip

    if not _is_timescaledb():
        return

    # Convert tables to hypertables
    for table_name, time_col in HYPERTABLES:
        try:
            conn.execute(
                sa.text(
                    f"SELECT create_hypertable('{table_name}', '{time_col}', "
                    f"migrate_data => true, if_not_exists => true)"
                )
            )
        except Exception:
            pass  # Table might not exist yet or other issue

    # Enable compression
    for tbl in COMPRESS_TABLES:
        try:
            # Check if the table has a symbol column for segmentby
            result = conn.execute(
                sa.text(
                    "SELECT 1 FROM information_schema.columns "
                    "WHERE table_name = :tbl AND column_name = 'symbol'"
                ),
                {"tbl": tbl},
            )
            has_symbol = result.scalar() is not None

            if has_symbol:
                conn.execute(
                    sa.text(
                        f"ALTER TABLE {tbl} SET ("
                        f"timescaledb.compress, "
                        f"timescaledb.compress_segmentby = 'symbol'"
                        f")"
                    )
                )
            else:
                conn.execute(
                    sa.text(f"ALTER TABLE {tbl} SET (timescaledb.compress)")
                )

            conn.execute(
                sa.text(
                    f"SELECT add_compression_policy('{tbl}', INTERVAL '7 days', "
                    f"if_not_exists => true)"
                )
            )
        except Exception:
            pass

    # Add continuous aggregates for common walk-forward queries
    try:
        conn.execute(
            sa.text("""
                CREATE MATERIALIZED VIEW IF NOT EXISTS price_bars_4h
                WITH (timescaledb.continuous) AS
                SELECT
                    time_bucket('4 hours', timestamp) AS timestamp,
                    symbol,
                    first(open, timestamp) AS open,
                    max(high) AS high,
                    min(low) AS low,
                    last(close, timestamp) AS close,
                    sum(volume) AS volume
                FROM price_bars_1h
                GROUP BY time_bucket('4 hours', timestamp), symbol
                WITH NO DATA
            """)
        )
        conn.execute(
            sa.text(
                "SELECT add_continuous_aggregate_policy('price_bars_4h', "
                "start_offset => INTERVAL '24 hours', "
                "end_offset => INTERVAL '1 hour', "
                "schedule_interval => INTERVAL '4 hours', "
                "if_not_exists => true)"
            )
        )
    except Exception:
        pass


def downgrade() -> None:
    conn = op.get_bind()

    if "sqlite" in str(conn.engine.url):
        return

    if not _is_timescaledb():
        return

    # Drop continuous aggregates
    try:
        conn.execute(
            sa.text("DROP MATERIALIZED VIEW IF EXISTS price_bars_4h CASCADE")
        )
    except Exception:
        pass

    # Note: Cannot easily revert hypertables to regular tables.
    # The data remains accessible; the hypertable metadata is preserved.
