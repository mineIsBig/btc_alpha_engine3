#!/usr/bin/env python3
"""Bootstrap the database: create tables and seed instruments."""
import sys
sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv()

from src.common.logging import setup_logging, get_logger
from src.common.config import load_yaml_config
from src.storage.database import init_db, session_scope
from src.storage.models import Instrument

setup_logging()
logger = get_logger(__name__)


def main() -> None:
    logger.info("bootstrapping_database")

    # Create all tables
    init_db()
    logger.info("tables_created")

    # Seed instruments
    assets = load_yaml_config("assets.yaml")
    with session_scope() as session:
        for inst in assets.get("instruments", []):
            existing = session.query(Instrument).filter_by(symbol=inst["symbol"]).first()
            if existing is None:
                session.add(Instrument(
                    symbol=inst["symbol"],
                    exchange_symbol=inst.get("exchange_symbol"),
                    coinglass_symbol=inst.get("coinglass_symbol"),
                    hyperliquid_symbol=inst.get("hyperliquid_symbol"),
                    tick_size=inst.get("tick_size", 0.1),
                    lot_size=inst.get("lot_size", 0.001),
                    min_notional=inst.get("min_notional", 10.0),
                    max_position_usd=inst.get("max_position_usd", 100000.0),
                    enabled=inst.get("enabled", True),
                ))
                logger.info("instrument_seeded", symbol=inst["symbol"])

    logger.info("bootstrap_complete")


if __name__ == "__main__":
    main()
