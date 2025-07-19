#!/usr/bin/env python3
"""
Main entry point for the Multi-CRM Cross-Sell Intelligence Platform
Fixed to properly handle all CLI options
"""

import sys
import os
import asyncio
import logging
import click
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.orchestrator import CrossSellOrchestrator
from src.utils.logging_config import setup_logging


@click.command()
@click.option(
    '--run-once', 
    is_flag=True, 
    help='Run pipeline once and exit'
)
@click.option(
    '--schedule', 
    type=click.Choice(['hourly', 'daily', 'weekly']), 
    help='Schedule periodic runs'
)
@click.option(
    '--config', 
    default='config/orgs.json', 
    type=click.Path(exists=True),
    help='Path to organization config file'
)
@click.option(
    '--debug',
    is_flag=True,
    help='Enable debug logging'
)
@click.option(
    '--log-file',
    type=click.Path(),
    help='Log to file instead of console'
)
@click.option(
    '--dry-run',
    is_flag=True,
    help='Validate configuration without running pipeline'
)
@click.option(
    '--force-train',
    is_flag=True,
    help='Force retraining of ML models'
)
def main(run_once, schedule, config, debug, log_file, dry_run, force_train):
    """
    Multi-CRM Cross-Sell Intelligence Platform
    
    Examples:
        # Run once with default config
        python main.py --run-once
        
        # Schedule daily runs
        python main.py --schedule daily
        
        # Debug mode with custom config
        python main.py --run-once --debug --config config/test_orgs.json
        
        # Dry run to validate setup
        python main.py --dry-run
    """
    # Setup logging
    log_level = logging.DEBUG if debug else logging.INFO
    setup_logging(level=log_level, log_file=log_file)
    logger = logging.getLogger(__name__)
    
    # Log startup info
    logger.info("=" * 60)
    logger.info("Multi-CRM Cross-Sell Intelligence Platform")
    logger.info("=" * 60)
    logger.info(f"Config file: {config}")
    logger.info(f"Debug mode: {debug}")
    
    # Create orchestrator
    orchestrator = CrossSellOrchestrator(config_path=config)
    
    # Handle dry run
    if dry_run:
        logger.info("Performing dry run - validating configuration...")
        try:
            # Validate config
            if not orchestrator.orgs:
                logger.error("No organizations configured!")
                sys.exit(1)
            
            logger.info(f"Found {len(orchestrator.orgs)} organizations:")
            for org in orchestrator.orgs:
                logger.info(f"  - {org['org_name']} ({org['org_id']})")
            
            # Validate ML config
            logger.info(f"ML Config: {orchestrator.ml_config}")
            
            # Check required directories
            required_dirs = ['logs', 'outputs', 'models', 'data', 'exports']
            for dir_name in required_dirs:
                path = Path(dir_name)
                if not path.exists():
                    logger.warning(f"Creating missing directory: {dir_name}")
                    path.mkdir(exist_ok=True)
            
            logger.info("✓ Configuration validation passed!")
            return
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            sys.exit(1)
    
    # Run async main
    asyncio.run(async_main(orchestrator, run_once, schedule, force_train))


async def async_main(orchestrator, run_once, schedule, force_train):
    """Async main function to handle orchestrator operations"""
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize orchestrator
        await orchestrator.initialize()
        
        # Handle force training
        if force_train:
            logger.info("Force training ML models...")
            await orchestrator._train_model({})
        
        # Handle execution modes
        if schedule:
            # Schedule periodic runs
            logger.info(f"Scheduling {schedule} pipeline runs")
            orchestrator.schedule_runs(schedule)
            
            # Keep the scheduler running
            try:
                logger.info("Scheduler started. Press Ctrl+C to stop.")
                while True:
                    await asyncio.sleep(60)  # Check every minute
                    
                    # Log scheduler status
                    if orchestrator.scheduler.running:
                        jobs = orchestrator.scheduler.get_jobs()
                        if jobs:
                            next_run = jobs[0].next_run_time
                            logger.debug(f"Next scheduled run: {next_run}")
                            
            except KeyboardInterrupt:
                logger.info("Received interrupt signal, shutting down...")
                
        elif run_once or not schedule:
            # Run once (default behavior if no schedule specified)
            logger.info("Running pipeline once...")
            await orchestrator.run_pipeline()
            logger.info("Pipeline run completed successfully!")
            
        else:
            # This shouldn't happen, but handle it gracefully
            logger.error("No execution mode specified. Use --run-once or --schedule")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
        
    finally:
        # Clean shutdown
        await orchestrator.stop()
        logger.info("Shutdown complete.")


# Entry point with better error handling
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)
