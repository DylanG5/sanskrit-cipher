# ML Pipeline for Sanskrit Cipher

Extensible pipeline for processing 20K+ manuscript fragments through multiple ML models.

## Overview

This pipeline processes manuscript fragment images through various ML models (segmentation, classification, etc.) and populates the SQLite database with predictions. The architecture is designed for easy extension - adding new ML models is as simple as creating a new processor file.

## Features

- **Plugin Architecture**: Easy to add new ML models
- **Database Integration**: Automatic updates to SQLite database
- **Progress Tracking**: Model versions, timestamps, and error logging
- **Resume Capability**: Continue from where you left off
- **Configuration-Driven**: Enable/disable models via YAML
- **CLI Interface**: Simple command-line interface

## Quick Start

### 1. Install Dependencies

```bash
cd src/ml_pipeline
pip install -r requirements.txt
```

### 2. Run Database Migration

Add the required fields to the database:

```bash
python -m ml_pipeline.cli migrate
```

### 3. List Available Processors

```bash
python -m ml_pipeline.cli list
```

### 4. Run Pipeline

```bash
# Run all enabled processors on all fragments
python -m ml_pipeline.cli run

# Run specific processor
python -m ml_pipeline.cli run --processors segmentation

# Dry run on 10 fragments (no database changes)
python -m ml_pipeline.cli run --dry-run --limit 10

# Resume from last processed fragment
python -m ml_pipeline.cli run --resume
```

### 5. Check Status

```bash
python -m ml_pipeline.cli status
```

## Architecture

```
ml_pipeline/
├── core/                      # Core infrastructure
│   ├── processor.py          # Base processor interface
│   ├── orchestrator.py       # Pipeline orchestrator
│   ├── database.py           # Database operations
│   ├── registry.py           # Processor discovery
│   └── utils.py              # Utilities
│
├── processors/               # ML model processors (plugins)
│   ├── segmentation_processor.py
│   ├── classification_processor.py
│   └── [your_processor.py]  # Add new processors here
│
├── config.yaml              # Configuration
└── cli.py                   # CLI interface
```

## Configuration

Edit `config.yaml` to:
- Enable/disable processors
- Adjust model paths
- Configure processing options

```yaml
processors:
  segmentation:
    enabled: true
    model_path: "../models/segmentation/best.pt"
    config:
      confidence_threshold: 0.25
      model_version: "1.0"

  classification:
    enabled: true
    model_path: "../models/classification/best_model.pt"
    config:
      model_version: "1.0"
```

## Adding New Processors

See `processors/README.md` for a detailed guide on implementing new ML models.

**Quick steps:**

1. Create `processors/your_processor.py`
2. Subclass `BaseProcessor`
3. Implement required methods:
   - `_setup()` - Load model
   - `get_metadata()` - Return metadata
   - `process()` - Run inference
   - `cleanup()` - Release resources
4. Add configuration to `config.yaml`
5. Run: `python -m ml_pipeline.cli run --processors your_processor`

## CLI Reference

### Commands

- `run` - Execute the pipeline
- `list` - List available processors
- `status` - Show processing statistics
- `migrate` - Run database migration

### Run Options

- `--processors`, `-p` - Comma-separated processor names
- `--resume`, `-r` - Resume from last processed fragment
- `--dry-run`, `-d` - Test run without database changes
- `--limit`, `-l` - Limit number of fragments
- `--fragment-ids` - Process specific fragments
- `--config`, `-c` - Path to config file

### Examples

```bash
# Process first 100 fragments with segmentation only
python -m ml_pipeline.cli run --processors segmentation --limit 100

# Resume processing after interruption
python -m ml_pipeline.cli run --resume

# Process specific fragments
python -m ml_pipeline.cli run --fragment-ids "BLL1_001,BLL1_002,BLL1_003"

# Dry run to test configuration
python -m ml_pipeline.cli run --dry-run --limit 5
```

## Database Schema

The pipeline adds these fields to the `fragments` table:

- `processing_status` - pending/completed/failed
- `segmentation_model_version` - Segmentation model version
- `classification_model_version` - Classification model version
- `last_processed_at` - Timestamp
- `processing_error` - Error message if failed

## Output

### Database Updates

Processors update the database with their predictions:
- **Segmentation**: `segmentation_coords` (JSON contours)
- **Classification**: `line_count` (0-15)

### Cache Files

Segmentation processor generates transparent PNGs in:
```
web/web-canvas/electron/resources/cache/segmented/
  {fragment_id}_segmented.png
```

### Logs

Logs are written to:
- `logs/pipeline.log` - All activity
- `logs/errors.log` - Errors only

## Performance

With a single GPU:
- **Segmentation**: ~1-2 sec/fragment
- **Classification**: ~0.1 sec/fragment
- **Total for 20K fragments**: ~8-12 hours

## Troubleshooting

**ImportError: No module named 'ultralytics'**
```bash
pip install ultralytics
```

**Database not found**
- Check `database.path` in `config.yaml`
- Ensure path is relative to `config.yaml` location

**No processors found**
- Run `python -m ml_pipeline.cli list` to see available processors
- Check that processor files are in `processors/` directory
- Ensure processors subclass `BaseProcessor`

**Model file not found**
- Check `model_path` in processor config
- Ensure path is relative to `config.yaml` location

## Development

### Running Tests

```bash
pytest tests/
```

### Adding Logging

```python
self.logger.info("Processing started")
self.logger.warning("Low confidence: 0.3")
self.logger.error("Processing failed")
```

## License

Part of the Sanskrit Cipher project.
