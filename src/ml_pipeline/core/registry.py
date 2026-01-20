"""
Processor registry for discovering and managing ML model processors.

This module provides automatic discovery of all processor classes
in the processors/ directory.
"""

from typing import Dict, List, Type
import inspect
import importlib
import pkgutil


class ProcessorRegistry:
    """
    Discovers and manages all available processors.

    Automatically finds all BaseProcessor subclasses in the processors/ directory.
    """

    def __init__(self):
        self._processors: Dict[str, Type] = {}

    def discover(self) -> None:
        """
        Auto-discover all processor classes in processors/ module.

        Scans the ml_pipeline.processors package for any classes that:
        1. Subclass BaseProcessor
        2. Are not the BaseProcessor class itself
        3. Are concrete (not abstract)

        Processors are registered by name (class name with 'Processor' suffix removed).
        """
        # Avoid circular import
        from ml_pipeline.core.processor import BaseProcessor

        try:
            # Import the processors package
            processors_package = importlib.import_module('ml_pipeline.processors')
            processors_path = processors_package.__path__

            # Iterate through all modules in the processors package
            for importer, modname, ispkg in pkgutil.iter_modules(processors_path):
                if not ispkg and not modname.startswith('_'):
                    # Import the module
                    module = importlib.import_module(f'ml_pipeline.processors.{modname}')

                    # Find all classes in the module
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        # Check if it's a BaseProcessor subclass (but not BaseProcessor itself)
                        if (issubclass(obj, BaseProcessor) and
                            obj != BaseProcessor and
                            not inspect.isabstract(obj)):
                            # Register using lowercase name without 'Processor' suffix
                            processor_name = obj.__name__.replace('Processor', '').lower()
                            self._processors[processor_name] = obj

        except ImportError as e:
            # Processors package doesn't exist yet or has import errors
            print(f"Warning: Could not import processors package: {e}")

    def register(self, name: str, processor_class: Type) -> None:
        """
        Manually register a processor.

        Args:
            name: Name to register the processor under
            processor_class: Processor class to register
        """
        self._processors[name] = processor_class

    def get(self, name: str) -> Type:
        """
        Get processor class by name.

        Args:
            name: Processor name

        Returns:
            Processor class

        Raises:
            KeyError: If processor not found
        """
        if name not in self._processors:
            raise KeyError(
                f"Processor '{name}' not found. "
                f"Available processors: {', '.join(self.list_all())}"
            )
        return self._processors[name]

    def list_all(self) -> List[str]:
        """
        List all available processor names.

        Returns:
            List of registered processor names
        """
        return list(self._processors.keys())

    def __contains__(self, name: str) -> bool:
        """Check if a processor is registered."""
        return name in self._processors

    def __len__(self) -> int:
        """Return number of registered processors."""
        return len(self._processors)
