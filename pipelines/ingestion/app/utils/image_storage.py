"""Image extraction and storage management for DOCX ingestion."""

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import io

logger = logging.getLogger(__name__)


class ImageStorage:
    """Manages extraction and storage of images from DOCX files."""

    def __init__(self, base_dir: str):
        """Initialize image storage.
        
        Args:
            base_dir: Root directory for storing extracted images
        """
        self.base_dir = Path(base_dir)
        self.images_dir = self.base_dir / "images"
        self.metadata_file = self.base_dir / "image_metadata.json"
        
        # Create directories
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing metadata
        self._metadata: Dict[str, dict] = self._load_metadata()

    def _load_metadata(self) -> Dict[str, dict]:
        """Load existing image metadata from file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load image metadata: {e}")
        return {}

    def _save_metadata(self) -> None:
        """Persist image metadata to file."""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self._metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save image metadata: {e}")

    def save_image(
        self, 
        image_bytes: bytes, 
        filename: str,
        source_file: str = "",
        alt_text: str = ""
    ) -> str:
        """Save an image and return its unique ID.
        
        Args:
            image_bytes: Raw image data
            filename: Original filename (e.g., "image1.jpg")
            source_file: Source DOCX filename
            alt_text: Alternative/OCR text for the image
            
        Returns:
            Unique image ID for referencing
        """
        # Generate deterministic ID from content hash
        image_hash = hashlib.md5(image_bytes).hexdigest()[:8]
        image_id = f"img_{image_hash}"
        
        # Get file extension
        ext = Path(filename).suffix.lower() or ".jpg"
        output_filename = f"{image_id}{ext}"
        output_path = self.images_dir / output_filename
        
        # Save image only if not already exists
        if not output_path.exists():
            try:
                output_path.write_bytes(image_bytes)
                logger.debug(f"Saved image: {output_filename}")
            except Exception as e:
                logger.error(f"Failed to save image {filename}: {e}")
                return ""
        
        # Update metadata
        self._metadata[image_id] = {
            "image_id": image_id,
            "filename": output_filename,
            "original_filename": filename,
            "source_file": source_file,
            "alt_text": alt_text,
            "size_bytes": len(image_bytes),
            "hash": image_hash
        }
        self._save_metadata()
        
        return image_id

    def get_image_path(self, image_id: str) -> Path:
        """Get full path to an image file."""
        if image_id not in self._metadata:
            return None
        
        filename = self._metadata[image_id]["filename"]
        return self.images_dir / filename

    def get_image_bytes(self, image_id: str) -> bytes:
        """Get raw image bytes."""
        path = self.get_image_path(image_id)
        if path and path.exists():
            try:
                return path.read_bytes()
            except Exception as e:
                logger.error(f"Failed to read image {image_id}: {e}")
        return b""

    def get_image_metadata(self, image_id: str) -> dict:
        """Get metadata for an image."""
        return self._metadata.get(image_id, {})

    def get_index(self) -> Dict[str, dict]:
        """Get complete image index/metadata."""
        return self._metadata.copy()


class ImageExtractor:
    """Extract images from DOCX documents."""

    def __init__(self, storage: ImageStorage):
        self.storage = storage

    def extract_from_docx(
        self, 
        docx_path: str, 
        ocr_callback=None,
        max_images: int = None
    ) -> Tuple[List[str], Dict[str, str]]:
        """Extract images from a DOCX file.
        
        Args:
            docx_path: Path to DOCX file
            ocr_callback: Optional async function to OCR images
            max_images: Max images to extract (None = all)
            
        Returns:
            Tuple of (image_ids, image_position_map)
            image_position_map: {position_in_text: image_id}
        """
        import zipfile
        from docx import Document
        
        docx_path = Path(docx_path)
        image_ids = []
        position_map = {}  # Text position -> image_id
        
        try:
            # Extract images from DOCX ZIP structure
            with zipfile.ZipFile(str(docx_path), 'r') as zf:
                media_files = [n for n in zf.namelist() if n.startswith("word/media/")]
                
                for idx, media_name in enumerate(media_files):
                    if max_images and idx >= max_images:
                        break
                    
                    image_bytes = zf.read(media_name)
                    if not image_bytes:
                        continue
                    
                    # OCR if available
                    alt_text = ""
                    if ocr_callback:
                        try:
                            alt_text = ocr_callback(image_bytes)
                        except Exception as e:
                            logger.debug(f"OCR failed for {media_name}: {e}")
                    
                    # Save image
                    image_id = self.storage.save_image(
                        image_bytes=image_bytes,
                        filename=media_name.split('/')[-1],
                        source_file=docx_path.name,
                        alt_text=alt_text
                    )
                    
                    if image_id:
                        image_ids.append(image_id)
                        # Position key: use index as marker in text
                        position_map[f"[IMAGE_{idx}]"] = image_id
                        logger.info(f"Extracted {image_id} from {docx_path.name}")
            
        except Exception as e:
            logger.error(f"Failed to extract images from {docx_path}: {e}")
        
        return image_ids, position_map

