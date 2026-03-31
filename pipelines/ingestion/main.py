"""Main entry point for ingestion pipeline."""

import argparse
import logging
import sys

sys.path.insert(0, "/home/phungkien/ehc_ai_mvp")

from app.core.config import config
from shared.py.utils.logging import setup_logging
from app.sources.redmine_client import RedmineClient
from app.processors.chunker import FAQChunker
from app.sinks.qdrant_writer import QdrantWriter

# Setup logging
setup_logging(log_level="INFO", log_file="logs/ingestion.log")
logger = logging.getLogger(__name__)


def main():
    """CLI entry point for ingestion pipeline."""
    
    parser = argparse.ArgumentParser(description="Ingest FAQ from Redmine to Qdrant")
    parser.add_argument("--project", default="FAQ", help="Redmine project key")
    parser.add_argument("--status", default="*", help="Issue status IDs (comma-separated, default all)")
    parser.add_argument("--dry-run", action="store_true", help="Fetch but don't write to Qdrant")
    
    args = parser.parse_args()
    
    try:
        logger.info("=== Ingestion Pipeline Start ===")
        logger.info(f"Redmine: {config.REDMINE_URL}/{args.project}")
        logger.info(f"Qdrant: {config.QDRANT_URL}/{config.QDRANT_COLLECTION}")
        logger.info(f"Ollama: {config.OLLAMA_BASE_URL} (model: {config.OLLAMA_EMBEDDING_MODEL})")
        
        # Step 1: Fetch from Redmine
        logger.info("Step 1: Fetching FAQ from Redmine...")
        redmine = RedmineClient(config.REDMINE_URL, config.REDMINE_API_KEY)
        status_ids = args.status.split(",") if args.status != "*" else ["*"]
        faqs = redmine.fetch_faq_issues(project_key=args.project, status_ids=status_ids)
        logger.info(f"  → Fetched {len(faqs)} FAQ issues")
        
        if not faqs:
            logger.warning("No FAQ issues fetched. Check Redmine connection and API key.")
            return 1
        
        # Step 2: Chunk
        logger.info("Step 2: Chunking FAQ...")
        chunker = FAQChunker(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
        chunks = chunker.process_faq_batch(faqs)
        logger.info(f"  → Generated {len(chunks)} chunks")
        
        # Step 3: Embed & Write to Qdrant
        if args.dry_run:
            logger.info("Dry-run mode; skipping Qdrant write")
        else:
            logger.info("Step 3: Embedding and upserting to Qdrant...")
            writer = QdrantWriter(
                qdrant_url=config.QDRANT_URL,
                ollama_url=config.OLLAMA_BASE_URL,
                collection_name=config.QDRANT_COLLECTION,
                embedding_model=config.OLLAMA_EMBEDDING_MODEL,
                batch_size=config.BATCH_SIZE
            )
            written = writer.write_chunks(chunks)
            logger.info(f"  → Wrote {written} chunks to Qdrant")
        
        logger.info("=== Ingestion Pipeline Complete ===")
        return 0
    
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
