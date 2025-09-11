"""
Example usage of the Content Analysis System.
Demonstrates all the advanced content analysis features.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from content_analysis import ContentAnalysisEngine, create_content_analysis_engine
from content_analysis_db import ContentAnalysisDatabase
from enhanced_rag_processor import EnhancedRAGProcessor


def demonstrate_reference_extraction():
    """Demonstrate reference extraction capabilities."""
    print("=== Reference Extraction Demo ===")
    
    sample_content = """
    This work builds on previous research by Smith et al. (2020) and the 
    transformer architecture described in Vaswani et al. (2017). For deep 
    learning fundamentals, see "Deep Learning" (Goodfellow et al., 2016).
    
    Recent advances in arXiv:2103.15691 and doi:10.1038/nature12345 have 
    shown promising results. The evaluation follows benchmarks from NeurIPS 
    2019 and ICML 2020 proceedings.
    """
    
    engine = create_content_analysis_engine()
    
    # Extract references
    references = engine.reference_extractor.extract_references(sample_content)
    
    print(f"Found {len(references)} references:")
    for i, ref in enumerate(references, 1):
        print(f"\n{i}. {ref.raw_text}")
        print(f"   Authors: {ref.authors}")
        print(f"   Year: {ref.year}")
        print(f"   arXiv ID: {ref.arxiv_id}")
        print(f"   DOI: {ref.doi}")
        print(f"   Confidence: {ref.confidence:.2f}")
        if ref.context:
            print(f"   Context: {ref.context[:100]}...")


def demonstrate_concept_extraction():
    """Demonstrate concept extraction capabilities."""
    print("\n=== Concept Extraction Demo ===")
    
    sample_content = """
    This paper presents a novel Transformer architecture for Natural Language Processing.
    The model uses Multi-Head Attention mechanisms and incorporates BERT-style pretraining.
    Our Convolutional Neural Network baseline uses Batch Normalization and Dropout.
    
    The Deep Learning approach leverages Transfer Learning from large Language Models.
    We evaluate on the GLUE benchmark using standard Machine Learning techniques.
    The Reinforcement Learning agent uses Deep Q-Networks for decision making.
    """
    
    engine = create_content_analysis_engine()
    
    # Extract concepts
    concepts = engine.concept_extractor.extract_concepts(sample_content)
    
    print(f"Found {len(concepts)} key concepts:")
    for i, concept in enumerate(concepts[:10], 1):  # Top 10
        print(f"\n{i}. {concept.term}")
        print(f"   Frequency: {concept.frequency}")
        print(f"   Importance: {concept.importance_score:.3f}")
        if concept.context_examples:
            print(f"   Example: {concept.context_examples[0][:80]}...")


def demonstrate_topic_classification():
    """Demonstrate topic classification capabilities."""
    print("\n=== Topic Classification Demo ===")
    
    sample_papers = [
        {
            "id": "paper1",
            "title": "Deep Learning for Computer Vision",
            "summary": "Convolutional neural networks for image classification and object detection",
            "content": "computer vision deep learning CNN image recognition object detection convolution"
        },
        {
            "id": "paper2",
            "title": "Transformer Models for NLP",
            "summary": "Attention mechanisms and language understanding with transformers",
            "content": "natural language processing transformer attention BERT GPT language model"
        },
        {
            "id": "paper3",
            "title": "Reinforcement Learning in Robotics",
            "summary": "Robot control using deep reinforcement learning algorithms",
            "content": "robotics reinforcement learning robot control autonomous navigation manipulation"
        }
    ]
    
    engine = create_content_analysis_engine()
    
    # Classify topics
    topics_by_paper = engine.topic_classifier.classify_topics(sample_papers)
    
    for paper_id, topics in topics_by_paper.items():
        paper = next(p for p in sample_papers if p['id'] == paper_id)
        print(f"\nPaper: {paper['title']}")
        print(f"Topics identified: {len(topics)}")
        
        for topic in topics[:3]:  # Top 3 topics
            print(f"  - {topic.name} (weight: {topic.weight:.3f})")
            print(f"    Keywords: {', '.join(topic.keywords[:5])}")


def demonstrate_section_analysis():
    """Demonstrate section analysis capabilities."""
    print("\n=== Section Analysis Demo ===")
    
    sample_paper = """
    Abstract
    
    This paper presents a novel approach to machine learning that improves 
    accuracy by 15% over existing methods.
    
    1. Introduction
    
    Machine learning has become increasingly important. This work addresses 
    the limitation of current approaches by proposing a new architecture.
    
    2. Methodology
    
    Our approach consists of three main components:
    - Feature extraction using neural networks
    - Attention mechanism for weighting
    - Ensemble method for prediction
    
    3. Results
    
    We evaluate on five datasets. Results show significant improvements:
    Dataset A: 92.3% accuracy, Dataset B: 89.7% accuracy.
    
    4. Conclusion
    
    This work presents a novel approach that achieves state-of-the-art results.
    Future work will explore applications to other domains.
    """
    
    engine = create_content_analysis_engine()
    
    # Analyze sections
    sections = engine.section_analyzer.analyze_sections(sample_paper)
    
    print(f"Found {len(sections)} sections:")
    for section in sections:
        print(f"\nSection: {section.title}")
        print(f"Type: {section.section_type.value}")
        print(f"Content length: {len(section.content)} characters")
        
        if section.summary:
            print(f"Summary: {section.summary}")
        
        if section.key_points:
            print(f"Key points: {len(section.key_points)}")
            for point in section.key_points[:2]:  # Top 2
                print(f"  • {point[:80]}...")


def demonstrate_full_analysis():
    """Demonstrate complete paper analysis."""
    print("\n=== Full Paper Analysis Demo ===")
    
    sample_paper_content = """
    arXiv:2306.12345v1 [cs.LG] 20 Jun 2023
    
    Deep Attention Networks for Multi-Modal Learning
    
    John Smith, Jane Doe, Bob Wilson
    University of AI Research
    
    Abstract
    
    We present a novel deep attention network architecture for multi-modal 
    learning tasks. Our approach combines visual and textual information 
    using cross-modal attention mechanisms, achieving state-of-the-art 
    results on three benchmark datasets.
    
    1. Introduction
    
    Multi-modal learning has gained significant attention in recent years.
    Previous work by Vaswani et al. (2017) introduced transformer architectures,
    while Devlin et al. (2019) showed the effectiveness of BERT for language tasks.
    
    2. Related Work
    
    Recent advances in attention mechanisms include works on visual attention
    (Xu et al., 2015) and cross-modal attention for VQA (Anderson et al., 2018).
    
    3. Methodology
    
    Our Deep Attention Network (DAN) consists of:
    • Visual encoder using ResNet-50 backbone
    • Text encoder using BERT-base
    • Cross-modal attention fusion layer
    • Multi-task learning objective
    
    4. Experiments
    
    We evaluate on VQA 2.0, COCO Captions, and Flickr30K datasets.
    Results show 3.2% improvement over previous best methods.
    
    5. Conclusion
    
    We propose DAN, a novel architecture for multi-modal learning.
    Our method achieves state-of-the-art performance across multiple benchmarks.
    
    References
    
    [1] Vaswani, A., et al. (2017). Attention is all you need. NeurIPS.
    [2] Devlin, J., et al. (2019). BERT. arXiv:1810.04805.
    [3] Anderson, P., et al. (2018). Bottom-up attention. CVPR.
    """
    
    engine = create_content_analysis_engine()
    
    # Perform complete analysis
    analysis = engine.analyze_paper(
        paper_id="demo_paper",
        title="Deep Attention Networks for Multi-Modal Learning",
        content=sample_paper_content
    )
    
    print(f"Paper: {analysis.title}")
    print(f"Technical Level: {analysis.technical_level}")
    print(f"Overall Summary: {analysis.overall_summary}")
    
    print(f"\n--- Analysis Results ---")
    print(f"References found: {len(analysis.references)}")
    print(f"Concepts identified: {len(analysis.concepts)}")
    print(f"Topics classified: {len(analysis.topics)}")
    print(f"Sections analyzed: {len(analysis.sections)}")
    print(f"Main contributions: {len(analysis.main_contributions)}")
    
    # Show top references
    if analysis.references:
        print(f"\n--- Top References ---")
        for ref in analysis.references[:3]:
            print(f"• {ref.raw_text}")
            if ref.arxiv_id:
                print(f"  arXiv: {ref.arxiv_id}")
    
    # Show top concepts
    if analysis.concepts:
        print(f"\n--- Top Concepts ---")
        for concept in analysis.concepts[:5]:
            print(f"• {concept.term} (importance: {concept.importance_score:.3f})")
    
    # Show topics
    if analysis.topics:
        print(f"\n--- Topics ---")
        for topic in analysis.topics[:3]:
            print(f"• {topic.name} (weight: {topic.weight:.3f})")
    
    # Show contributions
    if analysis.main_contributions:
        print(f"\n--- Main Contributions ---")
        for i, contrib in enumerate(analysis.main_contributions, 1):
            print(f"{i}. {contrib}")


def demonstrate_database_integration():
    """Demonstrate database storage and retrieval."""
    print("\n=== Database Integration Demo ===")
    
    # This would require the database to be set up
    try:
        db = ContentAnalysisDatabase()
        
        # Get statistics
        stats = db.get_analysis_statistics()
        
        print(f"Database Statistics:")
        print(f"  Total analyses: {stats.get('total_analyses', 0)}")
        print(f"  Total references: {stats.get('total_references', 0)}")
        print(f"  Total concepts: {stats.get('total_concepts', 0)}")
        
        # Show top concepts if available
        top_concepts = stats.get('top_concepts', [])
        if top_concepts:
            print(f"\n--- Top Concepts in Database ---")
            for concept, frequency in top_concepts[:5]:
                print(f"• {concept}: {frequency} occurrences")
        
    except Exception as e:
        print(f"Database not available: {e}")
        print("Run 'python rag_with_content_analysis.py' first to populate database")


def demonstrate_enhanced_processor():
    """Demonstrate the enhanced RAG processor."""
    print("\n=== Enhanced RAG Processor Demo ===")
    
    try:
        processor = EnhancedRAGProcessor(enable_content_analysis=True)
        
        # Show capabilities
        print("Enhanced processor initialized with:")
        print("✓ Content analysis enabled")
        print("✓ Reference extraction")
        print("✓ Concept identification")
        print("✓ Topic classification")
        print("✓ Section analysis")
        
        # Get statistics
        stats = processor.get_enhanced_statistics()
        print(f"\nCurrent statistics:")
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value}")
        
        # Demonstrate search by concepts (if data available)
        sample_concepts = ["machine learning", "neural network"]
        results = processor.search_by_concepts(sample_concepts, limit=3)
        
        if results:
            print(f"\nSearch results for {sample_concepts}:")
            for result in results:
                print(f"• {result['title'][:60]}...")
        else:
            print(f"\nNo results found for {sample_concepts} (database may be empty)")
        
    except Exception as e:
        print(f"Enhanced processor demo failed: {e}")


def main():
    """Run all demonstrations."""
    print("🚀 Content Analysis System Demonstrations\n")
    
    demonstrations = [
        ("Reference Extraction", demonstrate_reference_extraction),
        ("Concept Extraction", demonstrate_concept_extraction),
        ("Topic Classification", demonstrate_topic_classification),
        ("Section Analysis", demonstrate_section_analysis),
        ("Full Paper Analysis", demonstrate_full_analysis),
        ("Database Integration", demonstrate_database_integration),
        ("Enhanced Processor", demonstrate_enhanced_processor)
    ]
    
    for name, demo_func in demonstrations:
        try:
            demo_func()
            print(f"\n✅ {name} demonstration completed")
        except Exception as e:
            print(f"\n❌ {name} demonstration failed: {e}")
        
        print("-" * 60)
    
    print("\n🎯 Next Steps:")
    print("1. Process your documents: python rag_with_content_analysis.py --input-dir /path/to/pdfs")
    print("2. View statistics: python rag_with_content_analysis.py --stats")
    print("3. Search by concepts: python rag_with_content_analysis.py --search-concepts 'deep learning'")
    print("4. Run tests: python test_content_analysis.py")


if __name__ == "__main__":
    main()