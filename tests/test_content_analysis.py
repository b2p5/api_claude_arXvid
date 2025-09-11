"""
Comprehensive test suite for the Content Analysis System.
Tests reference extraction, concept detection, topic classification, and section analysis.
"""

import os
import sys
import unittest
import tempfile
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from content_analysis import (
    ReferenceExtractor, ConceptExtractor, TopicClassifier, SectionAnalyzer,
    ContentAnalysisEngine, Reference, Concept, Topic, Section, ContentAnalysis,
    SectionType
)
from logger import get_logger


class TestReferenceExtractor(unittest.TestCase):
    """Test suite for reference extraction functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.extractor = ReferenceExtractor()
        
        # Sample content with various reference formats
        self.sample_content = """
        This work builds upon previous research by Smith et al. (2020) and extends 
        the findings presented in arXiv:2103.12345. The methodology follows the 
        approach described in [Johnson2019] and incorporates techniques from 
        "Deep Learning" (Goodfellow et al., 2016). Recent work on transformer 
        architectures (Vaswani et al., 2017) has shown significant improvements.
        
        For implementation details, see https://doi.org/10.1038/nature12345 and
        the code repository at arXiv:2006.98765v2. The evaluation follows 
        benchmarks established in NeurIPS 2019 and ICML 2020 conferences.
        """
    
    def test_arxiv_extraction(self):
        """Test arXiv ID extraction."""
        references = self.extractor.extract_references(self.sample_content)
        
        arxiv_refs = [ref for ref in references if ref.arxiv_id]
        self.assertGreater(len(arxiv_refs), 0)
        
        # Check for specific arXiv IDs
        arxiv_ids = [ref.arxiv_id for ref in arxiv_refs]
        self.assertIn("2103.12345", arxiv_ids)
        self.assertIn("2006.98765", arxiv_ids)
    
    def test_doi_extraction(self):
        """Test DOI extraction."""
        references = self.extractor.extract_references(self.sample_content)
        
        doi_refs = [ref for ref in references if ref.doi]
        self.assertGreater(len(doi_refs), 0)
        
        # Check for specific DOI
        dois = [ref.doi for ref in doi_refs]
        self.assertTrue(any("10.1038/nature12345" in doi for doi in dois))
    
    def test_author_year_extraction(self):
        """Test author and year extraction."""
        references = self.extractor.extract_references(self.sample_content)
        
        # Check for author extraction
        author_refs = [ref for ref in references if ref.authors]
        self.assertGreater(len(author_refs), 0)
        
        # Check for year extraction
        year_refs = [ref for ref in references if ref.year]
        self.assertGreater(len(year_refs), 0)
        
        years = [ref.year for ref in year_refs]
        self.assertIn(2020, years)
        self.assertIn(2016, years)
        self.assertIn(2017, years)
    
    def test_venue_extraction(self):
        """Test venue extraction."""
        references = self.extractor.extract_references(self.sample_content)
        
        venue_refs = [ref for ref in references if ref.venue]
        self.assertGreater(len(venue_refs), 0)
        
        venues = [ref.venue for ref in venue_refs if ref.venue]
        venue_text = " ".join(venues).lower()
        self.assertTrue(any(conf in venue_text for conf in ["neurips", "icml"]))
    
    def test_confidence_scoring(self):
        """Test confidence scoring for references."""
        references = self.extractor.extract_references(self.sample_content)
        
        for ref in references:
            self.assertGreaterEqual(ref.confidence, 0.0)
            self.assertLessEqual(ref.confidence, 1.0)
        
        # References with arXiv IDs should have high confidence
        arxiv_refs = [ref for ref in references if ref.arxiv_id]
        for ref in arxiv_refs:
            self.assertGreaterEqual(ref.confidence, 0.8)
    
    def test_deduplication(self):
        """Test reference deduplication."""
        # Content with duplicate references
        duplicate_content = """
        Smith et al. (2020) proposed a method. Later, Smith et al. (2020) 
        extended this work. The same authors (Smith et al., 2020) also showed...
        """
        
        references = self.extractor.extract_references(duplicate_content)
        
        # Should deduplicate similar references
        author_year_pairs = [(ref.authors, ref.year) for ref in references if ref.authors and ref.year]
        unique_pairs = list(set((tuple(authors), year) for authors, year in author_year_pairs))
        
        # There should be fewer unique pairs than total references with authors/years
        self.assertLessEqual(len(unique_pairs), len(author_year_pairs))


class TestConceptExtractor(unittest.TestCase):
    """Test suite for concept extraction functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.extractor = ConceptExtractor()
        
        # Sample content with technical concepts
        self.sample_content = """
        This paper presents a novel Deep Neural Network architecture for 
        Natural Language Processing tasks. The proposed Transformer model 
        uses Self-Attention mechanisms and Multi-Head Attention to process 
        sequential data. Our Convolutional Neural Network (CNN) baseline 
        incorporates Batch Normalization and Dropout regularization.
        
        The machine learning approach leverages Gradient Descent optimization 
        with Adam optimizer. We evaluate on ImageNet dataset and achieve 
        state-of-the-art performance using Transfer Learning techniques.
        The reinforcement learning agent uses Deep Q-Learning algorithms.
        """
    
    def test_pattern_concept_extraction(self):
        """Test pattern-based concept extraction."""
        concepts = self.extractor.extract_concepts(self.sample_content)
        
        self.assertGreater(len(concepts), 0)
        
        # Check for specific technical terms
        concept_terms = [c.term.lower() for c in concepts]
        expected_terms = [
            "deep neural network", "transformer", "attention", 
            "convolutional neural network", "machine learning"
        ]
        
        found_terms = [term for term in expected_terms 
                      if any(term in ct for ct in concept_terms)]
        self.assertGreater(len(found_terms), 2)
    
    def test_frequency_counting(self):
        """Test concept frequency counting."""
        # Content with repeated terms
        repeated_content = """
        Machine learning is important. Machine learning algorithms are powerful.
        Deep learning is a subset of machine learning. Deep learning uses neural networks.
        Neural networks are the foundation of deep learning systems.
        """
        
        concepts = self.extractor.extract_concepts(repeated_content)
        
        # Find machine learning concept
        ml_concepts = [c for c in concepts if "machine learning" in c.term.lower()]
        if ml_concepts:
            self.assertGreaterEqual(ml_concepts[0].frequency, 2)
        
        # Find deep learning concept
        dl_concepts = [c for c in concepts if "deep learning" in c.term.lower()]
        if dl_concepts:
            self.assertGreaterEqual(dl_concepts[0].frequency, 2)
    
    def test_importance_scoring(self):
        """Test concept importance scoring."""
        concepts = self.extractor.extract_concepts(self.sample_content)
        
        for concept in concepts:
            self.assertGreaterEqual(concept.importance_score, 0.0)
            self.assertLessEqual(concept.importance_score, 1.0)
        
        # Should be sorted by importance
        scores = [c.importance_score for c in concepts]
        self.assertEqual(scores, sorted(scores, reverse=True))
    
    def test_context_extraction(self):
        """Test context example extraction."""
        concepts = self.extractor.extract_concepts(self.sample_content)
        
        for concept in concepts:
            if concept.context_examples:
                for context in concept.context_examples:
                    self.assertIsInstance(context, str)
                    self.assertGreater(len(context), 0)
    
    def test_stop_words_filtering(self):
        """Test filtering of stop words and common terms."""
        concepts = self.extractor.extract_concepts(self.sample_content)
        
        concept_terms = [c.term.lower() for c in concepts]
        
        # Should not contain common stop words
        stop_words = ["the", "and", "or", "in", "on", "at", "to", "for"]
        for stop_word in stop_words:
            self.assertNotIn(stop_word, concept_terms)


class TestTopicClassifier(unittest.TestCase):
    """Test suite for topic classification functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.classifier = TopicClassifier(n_topics=5)
        
        # Sample papers with different topics
        self.sample_papers = [
            {
                "id": "1",
                "title": "Deep Learning for Computer Vision",
                "summary": "This paper presents convolutional neural networks for image classification",
                "content": "Computer vision deep learning CNN image recognition object detection"
            },
            {
                "id": "2", 
                "title": "Natural Language Processing with Transformers",
                "summary": "We propose transformer models for text understanding and generation",
                "content": "NLP transformer attention language model text processing BERT GPT"
            },
            {
                "id": "3",
                "title": "Reinforcement Learning for Robotics", 
                "summary": "Robotic control using deep reinforcement learning algorithms",
                "content": "Robotics control reinforcement learning robot manipulation autonomous"
            },
            {
                "id": "4",
                "title": "Machine Learning Optimization",
                "summary": "Novel optimization techniques for training neural networks",
                "content": "Optimization gradient descent Adam SGD training convergence"
            },
            {
                "id": "5",
                "title": "Graph Neural Networks",
                "summary": "Learning on graph-structured data using neural networks",
                "content": "Graph neural networks GNN node embedding graph convolution"
            }
        ]
    
    def test_topic_classification(self):
        """Test basic topic classification."""
        topics_by_paper = self.classifier.classify_topics(self.sample_papers)
        
        self.assertEqual(len(topics_by_paper), len(self.sample_papers))
        
        for paper_id, topics in topics_by_paper.items():
            self.assertIsInstance(topics, list)
            for topic in topics:
                self.assertIsInstance(topic, Topic)
                self.assertGreater(len(topic.keywords), 0)
                self.assertGreaterEqual(topic.weight, 0.0)
                self.assertLessEqual(topic.weight, 1.0)
    
    def test_lda_classification(self):
        """Test LDA-based topic classification."""
        topics_by_paper = self.classifier.classify_topics(self.sample_papers)
        
        # Should find LDA topics
        lda_topics = []
        for topics in topics_by_paper.values():
            lda_topics.extend([t for t in topics if t.name.startswith("LDA_")])
        
        self.assertGreater(len(lda_topics), 0)
    
    def test_kmeans_classification(self):
        """Test K-Means clustering classification."""
        topics_by_paper = self.classifier.classify_topics(self.sample_papers)
        
        # Should find cluster topics
        cluster_topics = []
        for topics in topics_by_paper.values():
            cluster_topics.extend([t for t in topics if t.name.startswith("Cluster_")])
        
        self.assertGreater(len(cluster_topics), 0)
    
    def test_topic_naming(self):
        """Test topic name generation."""
        topics_by_paper = self.classifier.classify_topics(self.sample_papers)
        
        topic_names = []
        for topics in topics_by_paper.values():
            topic_names.extend([t.name for t in topics])
        
        # Should have reasonable topic names
        self.assertTrue(all(len(name) > 0 for name in topic_names))
        
        # Should identify some domain-specific topics
        all_names = " ".join(topic_names).lower()
        expected_domains = ["learning", "vision", "language", "network"]
        found_domains = [domain for domain in expected_domains if domain in all_names]
        self.assertGreater(len(found_domains), 0)
    
    def test_empty_papers(self):
        """Test handling of empty paper list."""
        topics_by_paper = self.classifier.classify_topics([])
        self.assertEqual(len(topics_by_paper), 0)
    
    def test_single_paper(self):
        """Test classification with single paper."""
        single_paper = [self.sample_papers[0]]
        topics_by_paper = self.classifier.classify_topics(single_paper)
        
        self.assertEqual(len(topics_by_paper), 1)
        self.assertIn("1", topics_by_paper)


class TestSectionAnalyzer(unittest.TestCase):
    """Test suite for section analysis functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.analyzer = SectionAnalyzer()
        
        # Sample paper content with sections
        self.sample_content = """
        Abstract
        
        This paper presents a novel approach to machine learning that improves
        accuracy by 15% over existing methods. We evaluate on multiple datasets
        and show consistent improvements.
        
        1. Introduction
        
        Machine learning has become increasingly important in recent years.
        This work addresses the limitation of current approaches by proposing
        a new architecture that combines multiple techniques.
        
        2. Related Work
        
        Previous work in this area includes Smith et al. (2020) and Jones (2019).
        These approaches have shown promising results but suffer from scalability
        issues that we address in this work.
        
        3. Methodology
        
        Our approach consists of three main components:
        - Feature extraction using deep neural networks
        - Attention mechanism for relevance weighting
        - Ensemble method for final prediction
        
        4. Results
        
        We evaluate our method on five standard datasets. The results show
        significant improvements: Dataset A: 92.3% accuracy, Dataset B: 89.7%
        accuracy, Dataset C: 94.1% accuracy.
        
        5. Conclusion
        
        This work presents a novel machine learning approach that achieves
        state-of-the-art results. Future work will explore applications to
        other domains and investigate theoretical foundations.
        
        References
        
        [1] Smith, J. et al. (2020). Machine Learning Advances.
        [2] Jones, A. (2019). Deep Learning Methods.
        """
    
    def test_section_splitting(self):
        """Test splitting content into sections."""
        sections = self.analyzer.analyze_sections(self.sample_content)
        
        self.assertGreater(len(sections), 1)
        
        # Check that we found major sections
        section_types = [s.section_type for s in sections]
        expected_types = [
            SectionType.ABSTRACT, SectionType.INTRODUCTION, 
            SectionType.METHODOLOGY, SectionType.RESULTS, SectionType.CONCLUSION
        ]
        
        found_types = [st for st in expected_types if st in section_types]
        self.assertGreater(len(found_types), 2)
    
    def test_section_classification(self):
        """Test section type classification."""
        sections = self.analyzer.analyze_sections(self.sample_content)
        
        # Find abstract section
        abstract_sections = [s for s in sections if s.section_type == SectionType.ABSTRACT]
        if abstract_sections:
            self.assertIn("novel approach", abstract_sections[0].content.lower())
        
        # Find methodology section  
        method_sections = [s for s in sections if s.section_type == SectionType.METHODOLOGY]
        if method_sections:
            self.assertIn("approach consists", method_sections[0].content.lower())
        
        # Find results section
        result_sections = [s for s in sections if s.section_type == SectionType.RESULTS]
        if result_sections:
            self.assertIn("accuracy", result_sections[0].content.lower())
    
    def test_key_point_extraction(self):
        """Test extraction of key points from sections."""
        sections = self.analyzer.analyze_sections(self.sample_content)
        
        sections_with_points = [s for s in sections if s.key_points]
        self.assertGreater(len(sections_with_points), 0)
        
        # Check that key points are meaningful
        for section in sections_with_points:
            for point in section.key_points:
                self.assertIsInstance(point, str)
                self.assertGreater(len(point), 10)  # Reasonable length
    
    def test_section_titles(self):
        """Test section title extraction and assignment."""
        sections = self.analyzer.analyze_sections(self.sample_content)
        
        for section in sections:
            self.assertIsInstance(section.title, str)
            self.assertGreater(len(section.title), 0)
    
    def test_empty_content(self):
        """Test handling of empty content."""
        sections = self.analyzer.analyze_sections("")
        # Should handle gracefully, might return empty list or single unknown section
        self.assertIsInstance(sections, list)
    
    def test_unstructured_content(self):
        """Test handling of content without clear sections."""
        unstructured = "This is just a paragraph of text without any clear section structure."
        sections = self.analyzer.analyze_sections(unstructured)
        
        # Should still create at least one section
        self.assertGreaterEqual(len(sections), 1)


class TestContentAnalysisEngine(unittest.TestCase):
    """Test suite for the main content analysis engine."""
    
    def setUp(self):
        """Set up test environment."""
        self.engine = ContentAnalysisEngine()
        
        # Sample paper data
        self.sample_paper_id = "test_paper_1"
        self.sample_title = "Machine Learning for Natural Language Processing"
        self.sample_content = """
        Abstract: This paper presents a novel transformer-based approach for NLP tasks.
        
        Introduction: Natural language processing has evolved significantly with deep learning.
        
        We propose a new architecture that combines attention mechanisms with 
        convolutional layers. The method achieves state-of-the-art results on 
        GLUE benchmark with 95.2% accuracy.
        
        References: 
        - Vaswani et al. (2017) Attention is All You Need
        - Devlin et al. (2019) BERT: arXiv:1810.04805
        """
    
    def test_single_paper_analysis(self):
        """Test analysis of a single paper."""
        analysis = self.engine.analyze_paper(
            self.sample_paper_id, 
            self.sample_title, 
            self.sample_content
        )
        
        self.assertIsInstance(analysis, ContentAnalysis)
        self.assertEqual(analysis.paper_id, self.sample_paper_id)
        self.assertEqual(analysis.title, self.sample_title)
        
        # Should have extracted various components
        self.assertIsInstance(analysis.references, list)
        self.assertIsInstance(analysis.concepts, list)
        self.assertIsInstance(analysis.topics, list)
        self.assertIsInstance(analysis.sections, list)
    
    def test_reference_extraction_integration(self):
        """Test integration of reference extraction."""
        analysis = self.engine.analyze_paper(
            self.sample_paper_id,
            self.sample_title,
            self.sample_content
        )
        
        # Should find references
        self.assertGreater(len(analysis.references), 0)
        
        # Should find arXiv reference
        arxiv_refs = [r for r in analysis.references if r.arxiv_id]
        self.assertGreater(len(arxiv_refs), 0)
    
    def test_concept_extraction_integration(self):
        """Test integration of concept extraction."""
        analysis = self.engine.analyze_paper(
            self.sample_paper_id,
            self.sample_title,
            self.sample_content
        )
        
        # Should find concepts
        self.assertGreater(len(analysis.concepts), 0)
        
        # Should find NLP-related concepts
        concept_terms = [c.term.lower() for c in analysis.concepts]
        nlp_terms = ["natural language", "transformer", "attention", "bert"]
        found_nlp_terms = [term for term in nlp_terms 
                          if any(term in ct for ct in concept_terms)]
        self.assertGreater(len(found_nlp_terms), 0)
    
    def test_section_analysis_integration(self):
        """Test integration of section analysis."""
        analysis = self.engine.analyze_paper(
            self.sample_paper_id,
            self.sample_title,
            self.sample_content
        )
        
        # Should find sections
        self.assertGreater(len(analysis.sections), 0)
        
        # Should identify abstract and introduction
        section_types = [s.section_type for s in analysis.sections]
        self.assertIn(SectionType.ABSTRACT, section_types)
    
    def test_technical_level_assessment(self):
        """Test technical level assessment."""
        analysis = self.engine.analyze_paper(
            self.sample_paper_id,
            self.sample_title,
            self.sample_content
        )
        
        # Should assign a technical level
        self.assertIn(analysis.technical_level, ["basic", "medium", "advanced"])
    
    def test_main_contributions_extraction(self):
        """Test extraction of main contributions."""
        # Content with clear contributions
        contribution_content = """
        Our main contributions are: 1) A novel architecture, 2) State-of-the-art results.
        We propose a new method that achieves 95% accuracy.
        """
        
        analysis = self.engine.analyze_paper("test", "Test Paper", contribution_content)
        
        # Should find contributions
        self.assertGreater(len(analysis.main_contributions), 0)
    
    def test_overall_summary_generation(self):
        """Test overall summary generation."""
        analysis = self.engine.analyze_paper(
            self.sample_paper_id,
            self.sample_title,
            self.sample_content
        )
        
        # Should generate a summary
        self.assertIsInstance(analysis.overall_summary, str)
        self.assertGreater(len(analysis.overall_summary), 0)
    
    def test_corpus_analysis(self):
        """Test analysis of multiple papers (corpus)."""
        papers = [
            {
                "id": "paper1",
                "title": "Deep Learning for Vision",
                "content": "Computer vision using convolutional neural networks"
            },
            {
                "id": "paper2", 
                "title": "NLP with Transformers",
                "content": "Natural language processing using transformer models"
            }
        ]
        
        analyses = self.engine.analyze_corpus(papers)
        
        self.assertEqual(len(analyses), 2)
        self.assertIn("paper1", analyses)
        self.assertIn("paper2", analyses)
        
        for analysis in analyses.values():
            self.assertIsInstance(analysis, ContentAnalysis)


def run_performance_benchmarks():
    """Run performance benchmarks for the content analysis system."""
    print("\n=== Content Analysis Performance Benchmarks ===")
    
    import time
    
    engine = ContentAnalysisEngine()
    
    # Test content of varying sizes
    small_content = "This is a small paper about machine learning." * 10
    medium_content = "This is a medium-sized paper about deep learning and neural networks." * 100
    large_content = "This is a large paper about artificial intelligence, machine learning, deep learning, and neural networks." * 1000
    
    test_cases = [
        ("Small paper", small_content),
        ("Medium paper", medium_content),
        ("Large paper", large_content)
    ]
    
    for name, content in test_cases:
        start_time = time.time()
        
        try:
            analysis = engine.analyze_paper(f"test_{name.lower()}", name, content)
            end_time = time.time()
            
            print(f"\n{name}:")
            print(f"  Processing time: {end_time - start_time:.3f}s")
            print(f"  Content length: {len(content):,} chars")
            print(f"  References found: {len(analysis.references)}")
            print(f"  Concepts found: {len(analysis.concepts)}")
            print(f"  Topics found: {len(analysis.topics)}")
            print(f"  Sections found: {len(analysis.sections)}")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    # Test corpus analysis
    print(f"\nCorpus Analysis Benchmark:")
    start_time = time.time()
    
    try:
        papers = [
            {"id": f"paper_{i}", "title": f"Test Paper {i}", "content": medium_content}
            for i in range(5)
        ]
        
        analyses = engine.analyze_corpus(papers)
        end_time = time.time()
        
        print(f"  Processing time: {end_time - start_time:.3f}s")
        print(f"  Papers analyzed: {len(analyses)}")
        print(f"  Avg time per paper: {(end_time - start_time) / len(analyses):.3f}s")
        
    except Exception as e:
        print(f"  Error: {e}")
    
    print("=== Benchmarks Complete ===\n")


if __name__ == '__main__':
    # Run unit tests
    print("Running Content Analysis System Tests...")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestReferenceExtractor))
    suite.addTests(loader.loadTestsFromTestCase(TestConceptExtractor))
    suite.addTests(loader.loadTestsFromTestCase(TestTopicClassifier))
    suite.addTests(loader.loadTestsFromTestCase(TestSectionAnalyzer))
    suite.addTests(loader.loadTestsFromTestCase(TestContentAnalysisEngine))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print results summary
    print(f"\n=== Test Results Summary ===")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFailures:")
        for test, trace in result.failures:
            print(f"- {test}: {trace.splitlines()[-1]}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, trace in result.errors:
            print(f"- {test}: {trace.splitlines()[-1]}")
    
    # Run performance benchmarks if tests passed
    if not result.failures and not result.errors:
        run_performance_benchmarks()
    
    # Exit with error code if tests failed
    exit_code = 0 if (len(result.failures) + len(result.errors)) == 0 else 1
    exit(exit_code)