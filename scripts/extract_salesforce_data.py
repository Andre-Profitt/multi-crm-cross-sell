#!/usr/bin/env python3
"""Extract Salesforce data defined in config and store it in the database."""

import asyncio
import click
import logging

from src.orchestrator import CrossSellOrchestrator
from src.utils.logging_config import setup_logging

@click.command()
@click.option("--config", default="config/orgs.json", help="Path to org config file")
def main(config: str) -> None:
    """Run data extraction for all configured Salesforce orgs."""

    setup_logging(level=logging.INFO)
    orchestrator = CrossSellOrchestrator(config_path=config)

    async def run():
        await orchestrator.initialize()
        await orchestrator._extract_all_org_data()
        await orchestrator.stop()

    asyncio.run(run())


if __name__ == "__main__":
    main()
