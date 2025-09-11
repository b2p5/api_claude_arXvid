import arxiv
import os
import argparse
import time
import random
import urllib.request
import re
from datetime import datetime
from config import get_config
from logger import get_logger, log_info, log_error, log_warning
from retry_utils import download_with_retry
from pdf_validator import validate_pdf

def search_arxiv(query):
    """
    Searches for papers on arXiv based on a query.

    Args:
        query: The search term.

    Returns:
        A generator of arxiv.Result objects.
    """
    config = get_config()
    logger = get_logger()
    
    logger.log_operation_start("arXiv search", query=query, max_results=config.arxiv.max_results)
    
    try:
        client = arxiv.Client()
        
        # Map string to arxiv enum
        sort_criterion = arxiv.SortCriterion.Relevance
        if config.arxiv.sort_criterion == "LastUpdatedDate":
            sort_criterion = arxiv.SortCriterion.LastUpdatedDate
        elif config.arxiv.sort_criterion == "SubmittedDate":
            sort_criterion = arxiv.SortCriterion.SubmittedDate
        
        search = arxiv.Search(
            query=query,
            max_results=config.arxiv.max_results,
            sort_by=sort_criterion
        )
        
        results = client.results(search)
        logger.log_operation_success("arXiv search", query=query)
        return results
        
    except Exception as e:
        logger.log_operation_failure("arXiv search", e, query=query)
        raise

def sanitize_filename(title):
    """Sanitizes a string to be used as a filename."""
    # Remove invalid characters for Windows filenames
    sanitized = re.sub(r'[\\/:*?"<>|]', "", title)
    # Replace spaces with underscores
    sanitized = sanitized.replace(" ", "_")
    return sanitized

def save_and_download_results(concept, results):
    """
    Saves the search results to a markdown file and downloads the PDFs.

    Args:
        concept: The search concept.
        results: A list of arxiv.Result objects.
    """
    config = get_config()
    logger = get_logger()
    
    logger.log_operation_start("Save and download results", concept=concept, count=len(results))
    
    directory = config.arxiv.get_concept_path(concept)
    os.makedirs(directory, exist_ok=True)
    
    date_str = datetime.now().strftime("%y_%m_%d")
    md_filename = f"arxiv_{date_str}.md"
    md_file_path = os.path.join(directory, md_filename)
    
    successful_downloads = 0
    failed_downloads = 0
    skipped_downloads = 0
    
    try:
        with open(md_file_path, "w", encoding="utf-8") as f:
            f.write(f"# Papers found for '{concept}'\n\n")
            
            for i, result in enumerate(results):
                pdf_filename = f"{sanitize_filename(result.title)}.pdf"
                f.write(f"- **{result.title}**: [PDF](./{pdf_filename})\n")
                
                pdf_save_path = os.path.join(directory, pdf_filename)

                # Check if PDF already exists
                if os.path.exists(pdf_save_path):
                    log_info(f"Skipping download - file exists", filename=pdf_filename)
                    skipped_downloads += 1
                    continue

                # Download delay
                delay = random.randint(config.arxiv.min_delay_seconds, config.arxiv.max_delay_seconds)
                log_info(f"Waiting {delay}s before download {i+1}/{len(results)}", paper=result.title[:50])
                time.sleep(delay)
                
                # Download with retry logic
                log_info(f"Starting download", url=result.pdf_url, filepath=pdf_save_path)
                
                if download_with_retry(result.pdf_url, pdf_save_path, timeout=config.arxiv.download_timeout):
                    # Validate downloaded PDF
                    validation_result = validate_pdf(pdf_save_path)
                    
                    if validation_result.is_valid:
                        log_info("Download and validation successful", filename=pdf_filename, size_mb=validation_result.size_mb)
                        successful_downloads += 1
                    else:
                        log_warning("Downloaded PDF failed validation", filename=pdf_filename, errors=len(validation_result.errors))
                        # Keep the file but mark as potentially problematic
                        successful_downloads += 1
                else:
                    log_error("Download failed after retries", url=result.pdf_url)
                    failed_downloads += 1
        
        logger.log_operation_success(
            "Save and download results", 
            concept=concept,
            successful=successful_downloads,
            failed=failed_downloads,
            skipped=skipped_downloads
        )
        
        log_info(f"Results summary", 
                directory=directory, 
                successful=successful_downloads, 
                failed=failed_downloads, 
                skipped=skipped_downloads)
                
    except Exception as e:
        logger.log_operation_failure("Save and download results", e, concept=concept)
        raise

if __name__ == "__main__":
    logger = get_logger()
    
    try:
        parser = argparse.ArgumentParser(description="Search for papers on arXiv.")
        parser.add_argument("concept", type=str, help="The concept to search for.")
        args = parser.parse_args()

        log_info("Starting arXiv search script", concept=args.concept)
        
        results = search_arxiv(args.concept)
        results_list = list(results)
        
        if results_list:
            log_info(f"Found {len(results_list)} papers", concept=args.concept)
            save_and_download_results(args.concept, results_list)
        else:
            log_warning(f"No papers found", concept=args.concept)
            print(f"No papers found for '{args.concept}'.")
            
        log_info("arXiv search script completed successfully")
        
    except KeyboardInterrupt:
        log_warning("Script interrupted by user")
        print("\nScript interrupted by user.")
    except Exception as e:
        log_error("Script failed", error=e)
        print(f"Script failed: {e}")
        exit(1)