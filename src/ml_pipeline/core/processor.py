"""
Base processor interface for ML models in the pipeline.

This module defines the abstract base class that all ML model processors must implement,
along with supporting dataclasses for metadata, fragment records, and processing results.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import logging


@dataclass
class ProcessorMetadata:
    """Metadata about a processor"""
    name: str
    version: str
    description: str
    model_path: Optional[str] = None
    requires_gpu: bool = False
    batch_size: int = 1


@dataclass
class FragmentRecord:
    """Represents a fragment from the database"""
    id: int
    fragment_id: str
    image_path: str
    edge_piece: Optional[bool] = None
    has_top_edge: Optional[bool] = None
    has_bottom_edge: Optional[bool] = None
    has_left_edge: Optional[bool] = None
    has_right_edge: Optional[bool] = None
    line_count: Optional[int] = None
    script_type: Optional[str] = None
    segmentation_coords: Optional[str] = None  # JSON string
    notes: Optional[str] = None
    processing_status: Optional[str] = None
    segmentation_model_version: Optional[str] = None
    classification_model_version: Optional[str] = None
    last_processed_at: Optional[str] = None
    processing_error: Optional[str] = None
    # Scale detection fields
    scale_unit: Optional[str] = None
    pixels_per_unit: Optional[float] = None
    scale_detection_status: Optional[str] = None
    scale_model_version: Optional[str] = None


@dataclass
class ProcessingResult:
    """Result from processing a single fragment"""
    success: bool
    updates: Dict[str, Any] = field(default_factory=dict)
    cache_files: Optional[Dict[str, Any]] = None  # filename -> image data
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseProcessor(ABC):
    """
    Abstract base class for all ML model processors.

    Each processor represents a single ML model that can process fragments
    and produce database updates.

    To implement a new processor:
    1. Subclass BaseProcessor
    2. Implement all abstract methods (_setup, get_metadata, process, cleanup)
    3. Optionally override should_process() for skip logic
    4. Add configuration to config.yaml
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize the processor.

        Args:
            config: Configuration dictionary for this processor
            logger: Logger instance for output
        """
        self.config = config
        self.logger = logger
        self._setup()

    @abstractmethod
    def _setup(self) -> None:
        """
        Load model and initialize resources. Called once at startup.

        This method should:
        - Load the ML model from disk
        - Initialize any preprocessing pipelines
        - Set up device (CPU/GPU)
        - Cache any configuration needed for inference
        """
        pass

    @abstractmethod
    def get_metadata(self) -> ProcessorMetadata:
        """
        Return processor metadata.

        Returns:
            ProcessorMetadata with name, version, description, etc.
        """
        pass

    @abstractmethod
    def process(self, fragment: FragmentRecord, data_dir: str) -> ProcessingResult:
        """
        Process a single fragment and return updates.

        This method should:
        1. Load the fragment image from data_dir/fragment.image_path
        2. Run ML model inference
        3. Return ProcessingResult with database field updates

        Args:
            fragment: Fragment database record
            data_dir: Base directory containing fragment images

        Returns:
            ProcessingResult with:
            - success: True if processing succeeded
            - updates: Dict of database fields to update (e.g., {'line_count': 5})
            - cache_files: Optional dict of filename -> image data for caching
            - error: Optional error message if success=False
            - metadata: Optional additional metadata (e.g., confidence scores)
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """
        Clean up resources (called at shutdown).

        This method should:
        - Release GPU memory
        - Close file handles
        - Clean up temporary files
        """
        pass

    def should_process(self, fragment: FragmentRecord) -> bool:
        """
        Determine if this fragment needs processing.

        Override to implement skip logic (e.g., already processed with same model version).

        Args:
            fragment: Fragment database record

        Returns:
            True if fragment should be processed, False to skip
        """
        return True
