"""
Pipeline orchestrator for running ML models on fragments.

Manages the execution of multiple processors on the fragment database.
"""

from typing import List, Optional, Dict, Any
from pathlib import Path
import cv2

from ml_pipeline.core.utils import load_config, setup_logging, resolve_path
from ml_pipeline.core.database import DatabaseManager
from ml_pipeline.core.registry import ProcessorRegistry
from ml_pipeline.core.processor import BaseProcessor


class PipelineOrchestrator:
    """
    Orchestrates the ML pipeline execution.

    - Loads configuration
    - Initializes processors
    - Iterates through fragments
    - Manages database transactions
    - Handles errors gracefully
    """

    def __init__(self, config_path: str):
        """
        Initialize the orchestrator.

        Args:
            config_path: Path to config.yaml file
        """
        # Load configuration
        self.config_path = Path(config_path)
        self.config = load_config(str(self.config_path))

        # Setup logging
        self.logger = setup_logging(self.config)

        # Resolve paths relative to config file location
        base_dir = self.config_path.parent

        db_path = resolve_path(self.config['database']['path'], base_dir)
        self.data_dir = resolve_path(self.config['data_dir'], base_dir)
        self.cache_dir = resolve_path(self.config['cache_dir'], base_dir)

        # Initialize database
        self.db = DatabaseManager(str(db_path), self.logger)

        # Initialize processor registry
        self.registry = ProcessorRegistry()
        self.registry.discover()

        self.processors: List[BaseProcessor] = []

        self.logger.info("Pipeline orchestrator initialized")
        self.logger.info(f"Database: {db_path}")
        self.logger.info(f"Data directory: {self.data_dir}")
        self.logger.info(f"Cache directory: {self.cache_dir}")

    def run(
        self,
        processor_names: Optional[List[str]] = None,
        resume: bool = False,
        dry_run: bool = False,
        limit: Optional[int] = None,
        fragment_ids: Optional[List[str]] = None
    ) -> None:
        """
        Run the pipeline.

        Args:
            processor_names: List of processors to run (None = all enabled)
            resume: Resume from last successfully processed fragment
            dry_run: Don't commit database changes
            limit: Limit number of fragments to process
            fragment_ids: Process specific fragment IDs only
        """
        try:
            # Initialize processors
            self._initialize_processors(processor_names)

            if not self.processors:
                self.logger.error("No processors enabled")
                return

            # Get fragments to process
            fragments = self._get_fragments(resume, limit, fragment_ids)

            if not fragments:
                self.logger.info("No fragments to process")
                return

            total = len(fragments)
            self.logger.info(f"Processing {total} fragments with {len(self.processors)} processor(s)")

            if dry_run:
                self.logger.info("DRY RUN MODE - No database changes will be saved")

            # Process each fragment
            processed = 0
            failed = 0

            for idx, fragment in enumerate(fragments, 1):
                try:
                    self.logger.info(f"[{idx}/{total}] Processing {fragment.fragment_id}")
                    success = self._process_fragment(fragment, dry_run)

                    if success:
                        processed += 1
                    else:
                        failed += 1

                except Exception as e:
                    self.logger.error(f"Error processing {fragment.fragment_id}: {e}")
                    failed += 1

                    if not self.config['processing'].get('continue_on_error', True):
                        self.logger.error("Stopping pipeline due to error")
                        break

            # Summary
            self.logger.info(f"\nPipeline completed:")
            self.logger.info(f"  Total: {total}")
            self.logger.info(f"  Processed: {processed}")
            self.logger.info(f"  Failed: {failed}")

        finally:
            # Cleanup
            self._cleanup_processors()
            self.db.disconnect()

    def _initialize_processors(self, processor_names: Optional[List[str]]) -> None:
        """
        Initialize processors based on configuration.

        Args:
            processor_names: Specific processors to run, or None for all enabled
        """
        processors_config = self.config.get('processors', {})

        # Determine which processors to load
        if processor_names:
            # Use specified processors
            to_load = processor_names
        else:
            # Use all enabled processors from config
            to_load = [
                name for name, cfg in processors_config.items()
                if cfg.get('enabled', False)
            ]

        self.logger.info(f"Loading processors: {', '.join(to_load)}")

        for name in to_load:
            try:
                # Get processor config
                processor_config = processors_config.get(name, {}).copy()

                if not processor_config:
                    self.logger.warning(f"No configuration found for processor: {name}")
                    continue

                # Resolve model paths relative to config file
                base_dir = self.config_path.parent
                if 'model_path' in processor_config:
                    processor_config['model_path'] = str(resolve_path(
                        processor_config['model_path'], base_dir
                    ))
                if 'meta_path' in processor_config:
                    processor_config['meta_path'] = str(resolve_path(
                        processor_config['meta_path'], base_dir
                    ))

                # Get processor class from registry
                processor_class = self.registry.get(name)

                # Instantiate processor
                processor = processor_class(processor_config, self.logger)

                self.processors.append(processor)

                metadata = processor.get_metadata()
                self.logger.info(
                    f"  Loaded {metadata.name} v{metadata.version}: {metadata.description}"
                )

            except Exception as e:
                self.logger.error(f"Failed to load processor '{name}': {e}")

    def _get_fragments(
        self,
        resume: bool,
        limit: Optional[int],
        fragment_ids: Optional[List[str]]
    ) -> List:
        """
        Get fragments to process from database.

        Args:
            resume: Resume from last processed fragment
            limit: Maximum number of fragments
            fragment_ids: Specific fragment IDs to process

        Returns:
            List of FragmentRecord objects
        """
        offset = 0

        if resume:
            last_id = self.db.get_last_processed_id()
            if last_id:
                self.logger.info(f"Resuming from fragment ID {last_id}")
                offset = last_id

        return self.db.get_fragments(
            limit=limit,
            offset=offset,
            fragment_ids=fragment_ids
        )

    def _process_fragment(self, fragment, dry_run: bool) -> bool:
        """
        Process a single fragment through all processors.

        Args:
            fragment: FragmentRecord to process
            dry_run: Don't commit changes

        Returns:
            True if all processors succeeded, False otherwise
        """
        all_updates = {}
        all_cache_files = {}
        any_success = False

        # Run each processor
        for processor in self.processors:
            metadata = processor.get_metadata()

            # Check if fragment should be processed
            if not processor.should_process(fragment):
                self.logger.debug(
                    f"  Skipping {metadata.name} (already processed with v{metadata.version})"
                )
                continue

            self.logger.debug(f"  Running {metadata.name}...")

            # Process
            result = processor.process(fragment, str(self.data_dir))

            if result.success:
                self.logger.debug(f"    ✓ {metadata.name} succeeded")
                all_updates.update(result.updates)

                if result.cache_files:
                    all_cache_files.update(result.cache_files)

                any_success = True

            else:
                self.logger.warning(
                    f"    ✗ {metadata.name} failed: {result.error}"
                )

                # Update with error
                if result.updates:
                    all_updates.update(result.updates)

        # Update database if we have changes
        if all_updates and not dry_run:
            try:
                self.db.update_fragment(fragment.id, all_updates)
            except Exception as e:
                self.logger.error(f"Failed to update database for {fragment.fragment_id}: {e}")
                return False

        # Generate cache files
        if all_cache_files and not dry_run:
            try:
                self._generate_cache_files(all_cache_files)
            except Exception as e:
                self.logger.error(f"Failed to generate cache files for {fragment.fragment_id}: {e}")
                # Don't fail the whole fragment for cache generation errors

        return any_success

    def _generate_cache_files(self, cache_files: Dict[str, Any]) -> None:
        """
        Generate cache files (e.g., transparent PNGs).

        Args:
            cache_files: Dict of filename -> image data (numpy array)
        """
        # Create segmented cache directory
        segmented_dir = self.cache_dir / 'segmented'
        segmented_dir.mkdir(parents=True, exist_ok=True)

        for filename, image_data in cache_files.items():
            output_path = segmented_dir / filename

            # Write PNG file
            success = cv2.imwrite(str(output_path), image_data)

            if success:
                self.logger.debug(f"    Saved cache file: {filename}")
            else:
                self.logger.warning(f"    Failed to save cache file: {filename}")

    def _cleanup_processors(self) -> None:
        """Cleanup all processor resources."""
        self.logger.info("Cleaning up processors...")
        for processor in self.processors:
            try:
                processor.cleanup()
            except Exception as e:
                self.logger.error(f"Error cleaning up processor: {e}")

        self.processors = []
