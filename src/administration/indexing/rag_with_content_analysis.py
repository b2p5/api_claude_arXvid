"""
RAG System with Advanced Content Analysis
Enhanced version of the RAG database vector script with content analysis capabilities.
"""

import os
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any

import dotenv

from config import get_config
from logger import get_logger, log_info, log_warning, log_error
from enhanced_rag_processor import EnhancedRAGProcessor
from content_analysis_db import ContentAnalysisDatabase


def main():
    """Main function for RAG system with content analysis."""
    dotenv.load_dotenv()
    
    parser = argparse.ArgumentParser(
        description="Create/update RAG database with advanced content analysis"
    )
    parser.add_argument("--input-dir", type=str, 
                       help="Directory containing PDF files (overrides config)")
    parser.add_argument("--force", action="store_true",
                       help="Force reprocessing of existing papers")
    parser.add_argument("--disable-analysis", action="store_true",
                       help="Disable content analysis (basic processing only)")
    parser.add_argument("--max-workers", type=int, default=4,
                       help="Maximum number of parallel workers")
    parser.add_argument("--stats", action="store_true",
                       help="Show detailed statistics")
    parser.add_argument("--search-concepts", nargs="+",
                       help="Search papers by concepts")
    parser.add_argument("--citation-network", action="store_true",
                       help="Display citation network")
    parser.add_argument("--export-analysis", type=str,
                       help="Export analysis results to JSON file")
    
    args = parser.parse_args()
    
    logger = get_logger()
    config = get_config()
    
    try:
        logger.log_operation_start("RAG with Content Analysis")
        
        # Initialize processor
        processor = EnhancedRAGProcessor(
            enable_content_analysis=not args.disable_analysis
        )
        
        # Handle different operations
        if args.stats:
            show_statistics(processor)
            
        elif args.search_concepts:
            search_by_concepts(processor, args.search_concepts)
            
        elif args.citation_network:
            show_citation_network(processor)
            
        elif args.export_analysis:
            export_analysis_results(processor, args.export_analysis)
            
        else:
            # Main processing
            process_documents(processor, args)
        
        logger.log_operation_success("RAG with Content Analysis")
        
    except KeyboardInterrupt:
        log_warning("Processing interrupted by user")
        print("\nProcessing interrupted by user.")
    except Exception as e:
        logger.log_operation_failure("RAG with Content Analysis", e)
        print(f"Processing failed: {e}")
        return 1
    
    return 0


def process_documents(processor: EnhancedRAGProcessor, args):
    """Process documents with content analysis."""
    config = get_config()
    
    # Determine input directory
    input_dir = args.input_dir or config.arxiv.documents_root
    
    if not os.path.exists(input_dir):
        print(f"Input directory not found: {input_dir}")
        return
    
    print(f"Processing documents from: {input_dir}")
    print(f"Content analysis: {'Enabled' if not args.disable_analysis else 'Disabled'}")
    print(f"Max workers: {args.max_workers}")
    
    # Collect PDF files
    pdf_files = collect_pdf_files(input_dir)
    
    if not pdf_files:
        print("No PDF files found in the specified directory")
        return
    
    print(f"Found {len(pdf_files)} PDF files")
    
    # Process with enhanced pipeline
    start_time = time.time()
    
    results = processor.process_papers_with_analysis(
        pdf_files=pdf_files,
        max_workers=args.max_workers
    )
    
    end_time = time.time()
    
    # Display results
    print(f"\n=== Processing Results ===")
    print(f"Total processing time: {end_time - start_time:.2f} seconds")
    print(f"Papers processed: {results['stats']['processed_papers']}/{results['stats']['total_papers']}")
    
    if not args.disable_analysis:
        print(f"Content analyses created: {results['stats']['analyses_created']}")
        print(f"References extracted: {results['stats']['references_extracted']}")
        print(f"Concepts extracted: {results['stats']['concepts_extracted']}")
    
    if results['success']:
        print("✅ Processing completed successfully")
    else:
        print(f"❌ Processing failed: {results.get('error', 'Unknown error')}")


def collect_pdf_files(root_dir: str) -> List[str]:
    """Collect all PDF files from directory tree."""
    pdf_files = []
    
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    
    return pdf_files


def show_statistics(processor: EnhancedRAGProcessor):
    """Display comprehensive statistics."""
    print("\n=== System Statistics ===")
    
    stats = processor.get_enhanced_statistics()
    
    # Basic stats
    print(f"Total papers processed: {stats.get('processed_papers', 0)}")
    
    if processor.enable_content_analysis:
        print(f"\n--- Content Analysis Stats ---")
        print(f"Content analyses: {stats.get('total_analyses', 0)}")
        print(f"References extracted: {stats.get('total_references', 0)}")
        print(f"Concepts identified: {stats.get('total_concepts', 0)}")
        print(f"Topics identified: {stats.get('total_topics', 0)}")
        print(f"Citation relationships: {stats.get('total_citations', 0)}")
        
        # Top concepts
        top_concepts = stats.get('top_concepts', [])
        if top_concepts:
            print(f"\n--- Top Concepts ---")
            for i, (concept, frequency) in enumerate(top_concepts[:10], 1):
                print(f"{i:2d}. {concept}: {frequency} occurrences")
        
        # Technical level distribution
        tech_levels = stats.get('technical_levels', {})
        if tech_levels:
            print(f"\n--- Technical Level Distribution ---")
            for level, count in tech_levels.items():
                print(f"{level.capitalize()}: {count} papers")


def search_by_concepts(processor: EnhancedRAGProcessor, concepts: List[str]):
    """Search papers by concepts."""
    print(f"\n=== Searching for concepts: {', '.join(concepts)} ===")
    
    results = processor.search_by_concepts(concepts, limit=20)
    
    if not results:
        print("No papers found with the specified concepts")
        return
    
    print(f"Found {len(results)} papers:")
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['title']}")
        print(f"   Matching concepts: {', '.join(result['matching_concepts'])}")
        print(f"   Importance score: {result['importance_score']:.3f}")
        if result['summary']:
            print(f"   Summary: {result['summary'][:100]}...")


def show_citation_network(processor: EnhancedRAGProcessor):
    """Display citation network."""
    print("\n=== Citation Network ===")
    
    network = processor.get_citation_network()
    
    if not network:
        print("No citation relationships found")
        return
    
    print(f"Papers with outgoing citations: {len(network)}")
    total_citations = sum(len(cited) for cited in network.values())
    print(f"Total citation relationships: {total_citations}")
    
    print("\n--- Top Citing Papers ---")
    sorted_papers = sorted(network.items(), key=lambda x: len(x[1]), reverse=True)
    
    for i, (citing_paper, cited_papers) in enumerate(sorted_papers[:10], 1):
        print(f"{i:2d}. {citing_paper[:60]}...")
        print(f"    Cites {len(cited_papers)} papers")
        if len(cited_papers) <= 3:
            for cited in cited_papers:
                print(f"      → {cited[:50]}...")
        else:
            for cited in cited_papers[:2]:
                print(f"      → {cited[:50]}...")
            print(f"      → ... and {len(cited_papers) - 2} more")


def export_analysis_results(processor: EnhancedRAGProcessor, output_file: str):
    """Export analysis results to JSON file."""
    print(f"\n=== Exporting Analysis Results to {output_file} ===")
    
    try:
        import json
        
        # Collect all analysis data
        export_data = {
            'metadata': {
                'export_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'content_analysis_enabled': processor.enable_content_analysis
            },
            'statistics': processor.get_enhanced_statistics(),
            'citation_network': processor.get_citation_network(),
            'concept_cooccurrence': {}
        }
        
        if processor.enable_content_analysis:
            export_data['concept_cooccurrence'] = processor.get_concept_cooccurrence()
        
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Analysis results exported successfully")
        print(f"File size: {os.path.getsize(output_file):,} bytes")
        
    except Exception as e:
        print(f"❌ Export failed: {e}")


def run_demo():
    """Run a demonstration of the content analysis features."""
    print("=== Content Analysis Demo ===")
    
    config = get_config()
    
    # Check if we have any processed papers
    db = ContentAnalysisDatabase()
    stats = db.get_analysis_statistics()
    
    if stats.get('total_analyses', 0) == 0:
        print("No content analyses found. Please run the main processing first:")
        print("python rag_with_content_analysis.py")
        return
    
    print(f"Found {stats['total_analyses']} analyzed papers")
    
    # Show top concepts
    if stats.get('top_concepts'):
        print("\n--- Top Concepts ---")
        for concept, frequency in stats['top_concepts'][:10]:
            print(f"  {concept}: {frequency} occurrences")
    
    # Show citation network size
    processor = EnhancedRAGProcessor(enable_content_analysis=True)
    network = processor.get_citation_network()
    if network:
        print(f"\n--- Citation Network ---")
        print(f"Papers with citations: {len(network)}")
        total_citations = sum(len(cited) for cited in network.values())
        print(f"Total citations: {total_citations}")
    
    print("\nFor more detailed analysis, use:")
    print("  python rag_with_content_analysis.py --stats")
    print("  python rag_with_content_analysis.py --citation-network")
    print("  python rag_with_content_analysis.py --search-concepts machine learning")


if __name__ == "__main__":
    # Check if this is a demo run
    if len(os.sys.argv) == 1:
        run_demo()
    else:
        exit(main())