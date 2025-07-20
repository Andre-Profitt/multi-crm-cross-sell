#!/usr/bin/env python3
"""Train the cross-sell model using data stored in the database."""

import asyncio
import click
import logging

from src.orchestrator import CrossSellOrchestrator
from src.utils.logging_config import setup_logging

@click.command()
@click.option("--config", default="config/orgs.json", help="Path to org config file")
def main(config: str) -> None:
    """Train the ensemble model on real Salesforce data."""

    setup_logging(level=logging.INFO)
    orchestrator = CrossSellOrchestrator(config_path=config)

    async def run():
        await orchestrator.initialize()
        # Training uses data already stored in DB
        await orchestrator._train_model({})
        await orchestrator.stop()

    asyncio.run(run())


if __name__ == "__main__":
    main()
