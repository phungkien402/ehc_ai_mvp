# DOCX Image Extraction & Storage - Usage Guide

## Overview

The `ehc_ai_mvp` ingestion pipeline now extracts images from DOCX files and stores them separately, while maintaining references in the vector database.

### Data Flow

```
DOCX File (text + images)
    ↓
Extract Text + Extract Images
    ↓
    ├─→ Text: Vector DB (with image_ids metadata)
    └─→ Images: Separate Storage (data/docx_images/images/)
         └─→ Metadata: data/docx_images/image_metadata.json
```

## Components

### 1. ImageStorage (`app/utils/image_storage.py`)
Manages image extraction and storage:
- Saves images with deterministic IDs (MD5 hash-based)
- Maintains metadata index of all stored images
- Provides retrieval methods for images

### 2. ImageExtractor (`app/utils/image_storage.py`)
Extracts images from DOCX files:
- Reads images from DOCX ZIP structure
- Calls OCR callback for alternative text
- Returns image IDs and position maps

### 3. Updated DocxParser (`app/sources/docx_parser.py`)
Now includes image extraction:
- Initializes `ImageStorage` 
- Extracts images during parsing
- Associates image IDs with text sections via `ModuleDocSection.image_ids`

## Usage

### Running Ingestion with Image Extraction

```bash
cd /home/phungkien/ehc_ai_mvp

# Ingest DOCX files with image extraction
python pipelines/ingestion/main.py \
  --source docx \
  --docx-dir data/documents \
  --enable-images

# Or with specific file
python pipelines/ingestion/main.py \
  --source docx \
  --docx-file data/documents/user_guide.docx \
  --enable-images
```

### Output Structure

After ingestion:
```
data/docx_images/
├── images/
│   ├── image_a1b2c3d4.jpg
│   ├── image_e5f6g7h8.png
│   └── ...
└── image_metadata.json
```

**image_metadata.json example:**
```json
{
  "img_a1b2c3d4": {
    "image_id": "img_a1b2c3d4",
    "filename": "image_a1b2c3d4.jpg",
    "original_filename": "media1.jpg",
    "source_file": "user_guide.docx",
    "alt_text": "Screenshot showing menu options",
    "size_bytes": 45320,
    "hash": "a1b2c3d4"
  },
  "img_e5f6g7h8": {
    ...
  }
}
```

## Data Model Changes

### ModuleDocSection
```python
@dataclass
class ModuleDocSection:
    # ... existing fields ...
    image_ids: List[str] = field(default_factory=list)  # NEW
    # image_ids contains list of image_XXXXX IDs stored in ImageStorage
```

### FAQChunk
```python
@dataclass
class FAQChunk:
    # ... existing fields ...
    image_ids: List[str] = field(default_factory=list)  # NEW
    # Inherits from ModuleDocSection during chunking
```

### QdrantPayload (Qdrant vector DB)
```python
@dataclass
class QdrantPayload:
    # ... existing fields ...
    image_ids: List[str] = field(default_factory=list)  # NEW
    # Stored in Qdrant metadata for retrieval
```

## Using Images in RAG/LLM Responses

### In RAG Retrieval

When retrieving chunks from Qdrant:
```python
# Search in Qdrant
results = qdrant.search(query_embedding, limit=5)

for result in results:
    chunk_id = result.payload["chunk_id"]
    image_ids = result.payload["image_ids"]  # Get associated images
    
    # Retrieve images from storage
    for image_id in image_ids:
        image_bytes = image_storage.get_image_bytes(image_id)
        # Use image_bytes to display or send to LLM
```

### In LLM Context

For multimodal LLMs like Claude or GPT-4V:
```python
# Get text chunk
text = result.payload["content_brief"]

# Get associated images
image_ids = result.payload["image_ids"]
images = []
for image_id in image_ids:
    image_bytes = image_storage.get_image_bytes(image_id)
    images.append(base64.b64encode(image_bytes).decode())

# Send to LLM with both text and images
response = llm.generate(
    text=text,
    images=images  # Include actual image data
)
```

## OCR Configuration

Images can have OCR text extracted and stored as `alt_text` in metadata:

```python
# In DocxParser initialization
docx_parser = DocxParser(
    enable_ocr=True,           # Enable OCR
    max_ocr_images=6,          # Limit OCR to first 6 images
    image_storage=image_storage,
    # ... other params
)
```

OCR text is available in `image_metadata.json`:
```json
{
  "img_a1b2c3d4": {
    "alt_text": "Menu: File | Edit | View | Help",  # OCR extracted text
    ...
  }
}
```

## Configuration

In `.env`:
```bash
# DOCX Processing
DOCX_INPUT_DIR=data/documents
DOCX_OCR_ENABLED=true
DOCX_OCR_MAX_IMAGES=6

# Image Storage (auto-created)
DOCX_IMAGES_OUTPUT_DIR=data/docx_images
```

## Troubleshooting

### Images not being extracted
1. Check DOCX file structure: `unzip -l file.docx | grep word/media/`
2. Verify image size is within bounds (15KB - 3MB)
3. Check logs: `tail -f logs/ingestion.log | grep -i image`

### OCR failing
1. OCR requires vision LLM endpoint (e.g., Ollama with vision model)
2. Falls back to Tesseract if vision endpoint unavailable
3. Use `--disable-docx-ocr` to skip OCR entirely

### Image metadata corrupted
1. Check JSON syntax: `python -m json.tool data/docx_images/image_metadata.json`
2. Manually edit or regenerate by re-running ingestion

## Next Steps

1. **Implement image retrieval in RAG workflow** - modify LangGraph to fetch images alongside text
2. **Add image display in UI** - show images in dashboard alongside text results
3. **Implement vector search on images** - use CLIP or similar for image-based search
4. **Add image quality metrics** - track extracted image quality and OCR accuracy

---

For issues or questions, check `pipelines/ingestion/logs/ingestion.log`
