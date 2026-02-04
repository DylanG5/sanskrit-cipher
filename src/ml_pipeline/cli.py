#!/usr/bin/env python3
"""
ML Pipeline CLI for Sanskrit Cipher

Usage:
    python -m ml_pipeline.cli run                          # Run all enabled processors
    python -m ml_pipeline.cli run --processors segmentation # Run only segmentation
    python -m ml_pipeline.cli run --resume                  # Resume from last failure
    python -m ml_pipeline.cli run --dry-run                 # Test run (no DB updates)
    python -m ml_pipeline.cli run --force --limit 1000      # Force reprocess 1000 fragments
    python -m ml_pipeline.cli run --collection BLL --limit 1000  # Process 1000 BLL fragments
    python -m ml_pipeline.cli list                          # List available processors
    python -m ml_pipeline.cli status                        # Show processing status
    python -m ml_pipeline.cli migrate                       # Run database migration
"""

import argparse
import sys
from pathlib import Path

from ml_pipeline.core.orchestrator import PipelineOrchestrator
from ml_pipeline.core.registry import ProcessorRegistry
from ml_pipeline.core.database import DatabaseManager
from ml_pipeline.core.utils import load_config, resolve_path, setup_logging


def cmd_run(args):
    """Run the pipeline"""
    config_path = args.config or 'config.yaml'

    if not Path(config_path).exists():
        print(f"Error: Configuration file not found: {config_path}")
        print("Please create config.yaml or specify --config path")
        sys.exit(1)

    orchestrator = PipelineOrchestrator(config_path)

    processor_names = args.processors.split(',') if args.processors else None
    fragment_ids = args.fragment_ids.split(',') if args.fragment_ids else None
    collection = args.collection if hasattr(args, 'collection') else None

    orchestrator.run(
        processor_names=processor_names,
        resume=args.resume,
        dry_run=args.dry_run,
        limit=args.limit,
        fragment_ids=fragment_ids,
        force=args.force,
        collection=collection
    )


def cmd_list(args):
    """List available processors"""
    print("Discovering processors...")
    registry = ProcessorRegistry()
    registry.discover()

    processors = registry.list_all()

    if not processors:
        print("No processors found.")
        print("\nMake sure processor files are in the processors/ directory")
        print("and they subclass BaseProcessor.")
        return

    print(f"\nAvailable processors ({len(processors)}):")
    print("=" * 50)

    config_path = args.config or 'config.yaml'

    # Try to load config to show enabled status
    try:
        config = load_config(config_path)
        processors_config = config.get('processors', {})
    except:
        processors_config = {}

    for name in sorted(processors):
        processor_class = registry.get(name)

        # Get status from config
        enabled = processors_config.get(name, {}).get('enabled', False)
        status = "✓ ENABLED" if enabled else "  disabled"

        print(f"  {status}  {name}")

        # Show description if we can instantiate (requires config)
        if enabled and name in processors_config:
            try:
                # Create a minimal logger for setup
                import logging
                logger = logging.getLogger(f'processor.{name}')
                logger.setLevel(logging.WARNING)

                processor = processor_class(processors_config[name], logger)
                metadata = processor.get_metadata()
                print(f"           v{metadata.version}: {metadata.description}")
                processor.cleanup()
            except:
                pass

    print()


def cmd_status(args):
    """Show processing status"""
    config_path = args.config or 'config.yaml'

    if not Path(config_path).exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)

    try:
        # Load config to get database path
        config = load_config(config_path)
        base_dir = Path(config_path).parent
        db_path = resolve_path(config['database']['path'], base_dir)

        # Connect to database
        db = DatabaseManager(str(db_path))
        stats = db.get_processing_stats()
        db.disconnect()

        # Display stats
        print("\nProcessing Status:")
        print("=" * 50)
        print(f"  Total fragments:     {stats['total']:,}")
        print(f"  Completed:           {stats['completed']:,}")
        print(f"  Pending:             {stats['pending']:,}")
        print(f"  Failed:              {stats['failed']:,}")

        if stats['total'] > 0:
            progress = stats['completed'] / stats['total'] * 100
            print(f"  Progress:            {progress:.1f}%")

        print()

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def cmd_migrate(args):
    """Run database migration"""
    config_path = args.config or 'config.yaml'

    if not Path(config_path).exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)

    try:
        # Load config to get database path
        config = load_config(config_path)
        logger = setup_logging(config)
        base_dir = Path(config_path).parent
        db_path = resolve_path(config['database']['path'], base_dir)

        print(f"Running migration on: {db_path}")

        # Connect to database and run migration
        db = DatabaseManager(str(db_path), logger)
        db.run_migration()
        db.disconnect()

        print("Migration completed successfully!")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="ML Pipeline for Sanskrit Cipher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all enabled processors
  python -m ml_pipeline.cli run

  # Run specific processor
  python -m ml_pipeline.cli run --processors segmentation

  # Dry run on 10 fragments
  python -m ml_pipeline.cli run --dry-run --limit 10

  # Resume from last processed fragment
  python -m ml_pipeline.cli run --resume

  # Force reprocess 1000 fragments
  python -m ml_pipeline.cli run --force --limit 1000

  # Process only BLL collection fragments
  python -m ml_pipeline.cli run --collection BLL --limit 1000

  # Process BLX fragments with force reprocessing
  python -m ml_pipeline.cli run --collection BLX --force

  # List available processors
  python -m ml_pipeline.cli list

  # Check processing status
  python -m ml_pipeline.cli status

  # Run database migration
  python -m ml_pipeline.cli migrate
        """
    )

    parser.add_argument(
        '--config', '-c',
        help='Path to config.yaml (default: ./config.yaml)'
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Run command
    run_parser = subparsers.add_parser('run', help='Run the pipeline')
    run_parser.add_argument(
        '--processors', '-p',
        help='Comma-separated list of processors to run (default: all enabled)'
    )
    run_parser.add_argument(
        '--resume', '-r',
        action='store_true',
        help='Resume from last processed fragment'
    )
    run_parser.add_argument(
        '--dry-run', '-d',
        action='store_true',
        help='Dry run (no database updates or cache file generation)'
    )
    run_parser.add_argument(
        '--fragment-ids',
        help='Process specific fragment IDs (comma-separated)'
    )
    run_parser.add_argument(
        '--limit', '-l',
        type=int,
        help='Limit number of fragments to process'
    )
    run_parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Force reprocessing of fragments even if already processed with same model version'
    )
    run_parser.add_argument(
        '--collection',
        help='Filter fragments by collection (e.g., BLL, BLX, BLL811). Matches image_path prefix.'
    )

    # List command
    list_parser = subparsers.add_parser('list', help='List available processors')

    # Status command
    status_parser = subparsers.add_parser('status', help='Show processing status')

    # Migrate command
    migrate_parser = subparsers.add_parser('migrate', help='Run database migration')

    args = parser.parse_args()

    if args.command == 'run':
        cmd_run(args)
    elif args.command == 'list':
        cmd_list(args)
    elif args.command == 'status':
        cmd_status(args)
    elif args.command == 'migrate':
        cmd_migrate(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
