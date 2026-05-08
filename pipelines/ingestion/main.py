"""Main entry point for ingestion pipeline."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, "/home/phungkien/ehc_ai_mvp")

from app.core.config import config
from shared.py.utils.logging import setup_logging
from app.sources.redmine_client import RedmineClient
from app.sources.docx_parser import DocxParser
from app.processors.chunker import FAQChunker
from app.sinks.qdrant_writer import QdrantWriter
from app.utils.image_storage import ImageStorage

# Setup logging
setup_logging(log_level="INFO", log_file="logs/ingestion.log")
logger = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_input_path(path_str: str) -> str:
    candidate = Path(path_str)
    if candidate.is_absolute():
        return str(candidate)
    return str((REPO_ROOT / candidate).resolve())


def main():
    """CLI entry point for ingestion pipeline."""
    
    parser = argparse.ArgumentParser(description="Ingest FAQ and DOCX sources into Qdrant")
    parser.add_argument(
        "--source",
        choices=["faq", "docx", "both"],
        default="faq",
        help="Which source to ingest",
    )
    parser.add_argument("--project", default="FAQ", help="Redmine project key")
    parser.add_argument("--status", default="*", help="Issue status IDs (comma-separated, default all)")
    parser.add_argument("--docx-dir", default=config.DOCX_INPUT_DIR, help="Directory containing DOCX files")
    parser.add_argument("--docx-file", action="append", default=[], help="Specific DOCX file path(s), can be repeated")
    parser.add_argument("--disable-docx-ocr", action="store_true", help="Disable OCR on DOCX embedded images")
    parser.add_argument("--dry-run", action="store_true", help="Fetch but don't write to Qdrant")
    
    args = parser.parse_args()
    
    try:
        logger.info("=== Ingestion Pipeline Start ===")
        logger.info(f"Redmine: {config.REDMINE_URL}/{args.project}")
        logger.info(
            "Qdrant collections: faq=%s, docs=%s",
            config.QDRANT_COLLECTION,
            config.QDRANT_DOCS_COLLECTION,
        )
        embedding_base_url = config.VLLM_EMBEDDING_URL if config.MODEL_PROVIDER == "vllm" else config.OLLAMA_BASE_URL
        logger.info(
            "Embedding provider: %s @ %s (model: %s)",
            config.MODEL_PROVIDER,
            embedding_base_url,
            config.OLLAMA_EMBEDDING_MODEL,
        )
        embedding_model = config.OLLAMA_EMBEDDING_MODEL
        if config.MODEL_PROVIDER == "vllm" and "/" not in embedding_model:
            # Keep ingestion resilient when env still carries Ollama-style names.
            embedding_model = "BAAI/bge-m3"
            logger.info("Adjusted embedding model for vLLM ingestion -> %s", embedding_model)
        
        # Step 1: Fetch/parse sources
        faqs = []
        doc_sections = []
        if args.source in {"faq", "both"}:
            logger.info("Step 1A: Fetching FAQ from Redmine...")
            redmine = RedmineClient(config.REDMINE_URL, config.REDMINE_API_KEY)
            status_ids = args.status.split(",") if args.status != "*" else ["*"]
            faqs = redmine.fetch_faq_issues(project_key=args.project, status_ids=status_ids)
            logger.info("  -> Fetched %d FAQ issues", len(faqs))

        if args.source in {"docx", "both"}:
            logger.info("Step 1B: Parsing DOCX module guides...")
            vision_base_url = config.VLLM_VISION_URL if config.MODEL_PROVIDER == "vllm" else config.OLLAMA_BASE_URL
            
            # Initialize image storage for DOCX images
            image_storage_dir = REPO_ROOT / "data" / "docx_images"
            image_storage = ImageStorage(base_dir=str(image_storage_dir))
            logger.info(f"  -> Image storage: {image_storage_dir}")
            
            docx_parser = DocxParser(
                provider=config.MODEL_PROVIDER,
                vision_base_url=vision_base_url,
                vision_model=config.OLLAMA_VISION_MODEL,
                enable_ocr=(config.DOCX_OCR_ENABLED and not args.disable_docx_ocr),
                max_ocr_images=config.DOCX_OCR_MAX_IMAGES,
                image_storage=image_storage,
            )
            if args.docx_file:
                for path in args.docx_file:
                    doc_sections.extend(docx_parser.parse_file(_resolve_input_path(path)))
            else:
                doc_sections = docx_parser.parse_directory(_resolve_input_path(args.docx_dir))
            logger.info("  -> Parsed %d DOCX sections", len(doc_sections))

        if not faqs and not doc_sections:
            logger.warning("No source data found for ingestion.")
            return 1

        # Step 2: Chunk
        logger.info("Step 2: Chunking sources...")
        chunker = FAQChunker(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
        faq_chunks = chunker.process_faq_batch(faqs) if faqs else []
        doc_chunks = chunker.process_module_doc_batch(doc_sections) if doc_sections else []
        logger.info("  -> Generated chunks: faq=%d, docx=%d", len(faq_chunks), len(doc_chunks))
        
        # Step 3: Embed & Write to Qdrant
        if args.dry_run:
            logger.info("Dry-run mode; skipping Qdrant write")
        else:
            logger.info("Step 3: Embedding and upserting to Qdrant...")
            writer = QdrantWriter(
                qdrant_url=config.QDRANT_URL,
                ollama_url=embedding_base_url,
                collection_name=config.QDRANT_COLLECTION,
                embedding_model=embedding_model,
                batch_size=config.BATCH_SIZE
            )

            written_faq = writer.write_chunks(faq_chunks, collection_name=config.QDRANT_COLLECTION) if faq_chunks else 0
            written_docs = writer.write_chunks(doc_chunks, collection_name=config.QDRANT_DOCS_COLLECTION) if doc_chunks else 0
            logger.info("  -> Wrote chunks: faq=%d, docx=%d", written_faq, written_docs)
        
        logger.info("=== Ingestion Pipeline Complete ===")
        return 0
    
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
