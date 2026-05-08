#!/usr/bin/env python3
"""Demo script: Extract images and text from DOCX with image storage."""

import sys
import logging
from pathlib import Path

# Add repo to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipelines.ingestion.app.utils.image_storage import ImageStorage, ImageExtractor
from pipelines.ingestion.app.sources.docx_parser import DocxParser
from app.core.config import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def demo_image_extraction():
    """Demo: Extract images from a DOCX file."""
    
    logger.info("=" * 60)
    logger.info("DOCX Image Extraction Demo")
    logger.info("=" * 60)
    
    # Initialize storage
    storage_dir = Path(__file__).resolve().parent / "data" / "demo_images"
    image_storage = ImageStorage(base_dir=str(storage_dir))
    logger.info(f"✓ Image storage initialized: {storage_dir}")
    
    # Find a sample DOCX file
    docx_dir = Path(__file__).resolve().parent / "data" / "documents"
    if not docx_dir.exists():
        logger.warning(f"Demo DOCX directory not found: {docx_dir}")
        logger.info("Creating sample directory...")
        docx_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"  -> Please add DOCX files to: {docx_dir}")
        return 1
    
    docx_files = list(docx_dir.glob("*.docx"))
    if not docx_files:
        logger.warning(f"No DOCX files found in {docx_dir}")
        return 1
    
    sample_docx = docx_files[0]
    logger.info(f"✓ Found sample DOCX: {sample_docx.name}")
    
    # Extract images using ImageExtractor
    logger.info("\nExtracting images...")
    extractor = ImageExtractor(image_storage)
    
    image_ids, position_map = extractor.extract_from_docx(
        str(sample_docx),
        ocr_callback=None,  # Skip OCR for demo
        max_images=10
    )
    
    logger.info(f"✓ Extracted {len(image_ids)} images")
    if image_ids:
        for img_id in image_ids:
            meta = image_storage.get_image_metadata(img_id)
            logger.info(f"  - {img_id}: {meta.get('original_filename')} ({meta.get('size_bytes')} bytes)")
    
    # Show image index
    logger.info("\nImage Index:")
    index = image_storage.get_index()
    logger.info(f"✓ Total images stored: {len(index)}")
    
    # Parse DOCX with DocxParser to show full integration
    logger.info(f"\nParsing DOCX with text + image metadata...")
    docx_parser = DocxParser(
        provider="ollama",
        vision_base_url="http://localhost:11434",
        vision_model="llava",
        enable_ocr=False,  # Skip OCR for demo
        image_storage=image_storage,
    )
    
    sections = docx_parser.parse_file(str(sample_docx))
    logger.info(f"✓ Parsed into {len(sections)} sections")
    
    for section in sections:
        if section.image_ids:
            logger.info(
                f"  - Section: {section.section_title} "
                f"({len(section.image_ids)} images: {', '.join(section.image_ids)})"
            )
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("✓ Demo Complete!")
    logger.info(f"  Images stored at: {storage_dir}/images/")
    logger.info(f"  Metadata at: {storage_dir}/image_metadata.json")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(demo_image_extraction())
