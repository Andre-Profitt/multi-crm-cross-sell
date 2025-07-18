#!/usr/bin/env python3
"""Main entry point for the Multi-CRM Cross-Sell Intelligence Platform"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import click
from src.orchestrator import CrossSellOrchestrator

@click.command()
@click.option('--run-once', is_flag=True, help='Run pipeline once')
@click.option('--schedule', type=click.Choice(['daily', 'weekly', 'hourly']), help='Schedule periodic runs')
@click.option('--config', default='config/orgs.json', help='Path to organization config')
def main(run_once, schedule, config):
    """Multi-CRM Cross-Sell Intelligence Platform"""
    orchestrator = CrossSellOrchestrator(config_path=config)
    
    if schedule:
        orchestrator.schedule_runs(schedule)
    else:
        orchestrator.run_pipeline()

if __name__ == '__main__':
    main()
