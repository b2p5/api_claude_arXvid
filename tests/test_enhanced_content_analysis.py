#!/usr/bin/env python3
"""
Test script for enhanced content analysis features.
Validates the improvements made to reference extraction, concept detection,
topic classification, and section analysis.
"""

import os
import sys
import traceback
from typing import Dict, Any, List

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from content_analysis import (
    ContentAnalysisEngine,
    ReferenceExtractor, 
    ConceptExtractor,
    TopicClassifier,
    SectionAnalyzer
)
from logger import get_logger, log_info, log_warning, log_error


class ContentAnalysisValidator:
    """Validator for enhanced content analysis features."""
    
    def __init__(self):
        self.logger = get_logger()
        self.test_results = []
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run comprehensive validation tests."""
        log_info("Starting enhanced content analysis validation")
        
        test_cases = [
            ("Reference Extraction", self.test_reference_extraction),
            ("Concept Detection", self.test_concept_detection),
            ("Topic Classification", self.test_topic_classification),  
            ("Section Analysis", self.test_section_analysis),
            ("End-to-End Analysis", self.test_end_to_end_analysis)
        ]
        
        results = {
            'tests_passed': 0,
            'tests_failed': 0,
            'test_details': [],
            'overall_success': False
        }
        
        for test_name, test_func in test_cases:
            try:
                print(f"\n=== Testing {test_name} ===")
                test_result = test_func()
                
                if test_result['success']:
                    results['tests_passed'] += 1
                    print(f"[PASS] {test_name}: PASSED")
                else:
                    results['tests_failed'] += 1
                    print(f"[FAIL] {test_name}: FAILED - {test_result.get('error', 'Unknown error')}")
                
                results['test_details'].append({
                    'name': test_name,
                    'result': test_result
                })
                
            except Exception as e:
                results['tests_failed'] += 1
                print(f"[ERROR] {test_name}: ERROR - {str(e)}")
                results['test_details'].append({
                    'name': test_name,
                    'result': {'success': False, 'error': str(e)}
                })
        
        results['overall_success'] = results['tests_failed'] == 0
        
        print(f"\n=== Test Summary ===")
        print(f"Tests passed: {results['tests_passed']}")
        print(f"Tests failed: {results['tests_failed']}")
        print(f"Overall result: {'PASSED' if results['overall_success'] else 'FAILED'}")
        
        return results
    
    def test_reference_extraction(self) -> Dict[str, Any]:
        """Test enhanced reference extraction capabilities."""
        extractor = ReferenceExtractor()
        
        # Sample text with various citation formats
        test_text = """
        This work builds on Attention is All You Need (Vaswani et al., 2017) and BERT (Devlin et al., 2019).
        Recent advances in generative models [Brown et al., 2020] have shown promising results.
        For more details, see arXiv:2106.09685 and https://arxiv.org/abs/2301.00005.
        The approach is similar to work published in NeurIPS 2022 and ICLR 2023.
        DOI: 10.1038/s41586-021-03819-2 provides additional context.
        OpenReview paper: https://openreview.net/forum?id=abc123
        """
        
        try:
            references = extractor.extract_references(test_text)
            
            # Validate results
            expected_patterns = ['vaswani', 'devlin', 'brown', '2106.09685', '2301.00005']
            found_patterns = []
            
            for ref in references:
                if any(pattern in ref.raw_text.lower() for pattern in expected_patterns):
                    found_patterns.extend([p for p in expected_patterns if p in ref.raw_text.lower()])
            
            success = len(references) >= 5 and len(set(found_patterns)) >= 3
            
            return {
                'success': success,
                'references_found': len(references),
                'patterns_matched': len(set(found_patterns)),
                'details': [{'raw_text': r.raw_text, 'confidence': r.confidence} for r in references[:5]]
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_concept_detection(self) -> Dict[str, Any]:
        """Test enhanced concept detection with modern terms."""
        extractor = ConceptExtractor()
        
        # Sample text with technical terms
        test_text = """
        We propose a novel Transformer Architecture with self-attention mechanisms.
        Our approach uses fine-tuning on pre-trained BERT models with dropout and batch normalization.
        The system implements reinforcement learning with multi-head attention for zero-shot learning.
        Experiments show improved performance on BLEU and ROUGE metrics using GPT-4 and CLIP embeddings.
        The method employs generative adversarial networks (GANs) and variational autoencoders (VAEs).
        """
        
        try:
            concepts = extractor.extract_concepts(test_text, min_frequency=1)
            
            # Expected modern terms
            expected_terms = [
                'transformer', 'self-attention', 'bert', 'fine-tuning', 
                'reinforcement learning', 'multi-head attention', 'zero-shot',
                'bleu', 'rouge', 'gpt', 'clip', 'gan', 'vae'
            ]
            
            found_terms = [c.term.lower() for c in concepts]
            matched_terms = [term for term in expected_terms if any(term in found_term for found_term in found_terms)]
            
            success = len(concepts) >= 10 and len(matched_terms) >= 8
            
            return {
                'success': success,
                'concepts_found': len(concepts),
                'expected_matches': len(matched_terms),
                'top_concepts': [(c.term, c.importance_score) for c in concepts[:5]]
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_topic_classification(self) -> Dict[str, Any]:
        """Test enhanced topic classification with new categories."""
        classifier = TopicClassifier()
        
        # Sample papers with different topics
        test_papers = [
            {
                'id': 'paper1',
                'title': 'Attention Is All You Need: Transformer Networks for NLP',
                'content': 'We propose the Transformer, a novel neural network architecture based on attention mechanisms for natural language processing tasks.'
            },
            {
                'id': 'paper2', 
                'title': 'Generative Adversarial Networks for Image Synthesis',
                'content': 'This paper introduces GANs for computer vision tasks, enabling realistic image generation through adversarial training.'
            },
            {
                'id': 'paper3',
                'title': 'Deep Reinforcement Learning for Robot Control',
                'content': 'We apply reinforcement learning algorithms to robotic manipulation tasks using deep neural networks and policy gradients.'
            }
        ]
        
        try:
            topics_by_paper = classifier.classify_topics(test_papers)
            
            # Check if we got topics for each paper
            papers_with_topics = len([pid for pid in topics_by_paper.keys() if topics_by_paper[pid]])
            
            # Check for expected topic categories
            all_topics = []
            for topics in topics_by_paper.values():
                all_topics.extend([t.name.lower() for t in topics])
            
            expected_categories = ['nlp', 'generative', 'reinforcement', 'computer vision', 'robotics']
            found_categories = [cat for cat in expected_categories if any(cat in topic for topic in all_topics)]
            
            success = papers_with_topics >= 2 and len(found_categories) >= 2
            
            return {
                'success': success,
                'papers_classified': papers_with_topics,
                'total_topics': len(all_topics),
                'category_matches': len(found_categories),
                'sample_topics': all_topics[:10]
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_section_analysis(self) -> Dict[str, Any]:
        """Test enhanced section analysis with modern section types."""
        analyzer = SectionAnalyzer()
        
        # Sample paper content with sections
        test_content = """
        Abstract
        This paper presents a novel approach to machine learning using attention mechanisms.
        
        1. Introduction
        Recent advances in deep learning have shown promising results in various domains.
        
        2. Related Work
        Prior work in this area has focused on traditional approaches.
        
        3. Methodology
        Our approach utilizes a Transformer architecture with self-attention.
        
        4. Experiments and Results
        We evaluate our method on standard benchmarks and show improved performance.
        
        5. Ablation Study
        We conduct ablation studies to understand the contribution of each component.
        
        6. Limitations
        Our method has several limitations that should be addressed in future work.
        
        7. Conclusion and Future Work
        We conclude that our approach shows promising results.
        
        References
        [1] Vaswani et al., 2017
        """
        
        try:
            sections = analyzer.analyze_sections(test_content)
            
            # Check if we identified different section types
            section_types = [s.section_type.value for s in sections]
            expected_types = ['abstract', 'introduction', 'related_work', 'methodology', 'results', 'discussion', 'conclusion', 'references']
            
            matched_types = [t for t in expected_types if t in section_types]
            sections_with_content = len([s for s in sections if len(s.content) > 50])
            
            success = len(sections) >= 6 and len(matched_types) >= 4 and sections_with_content >= 4
            
            return {
                'success': success,
                'sections_found': len(sections),
                'types_matched': len(matched_types),
                'sections_with_content': sections_with_content,
                'section_types': section_types
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_end_to_end_analysis(self) -> Dict[str, Any]:
        """Test complete end-to-end content analysis."""
        engine = ContentAnalysisEngine()
        
        # Sample paper for full analysis
        paper_content = """
        Abstract
        We present ChatGPT, a conversational AI system based on the GPT-3.5 architecture.
        Our approach uses reinforcement learning from human feedback (RLHF) to improve response quality.
        
        Introduction  
        Large language models have revolutionized natural language processing.
        Recent work on instruction following (Brown et al., 2020) has shown promising results.
        
        Methodology
        Our system builds on the Transformer architecture with self-attention mechanisms.
        We employ fine-tuning techniques and multi-head attention for improved performance.
        The training process uses reinforcement learning with policy gradient methods.
        
        Experiments
        We evaluate on standard NLP benchmarks including BLEU and ROUGE metrics.
        Results show significant improvements over baseline models including BERT and GPT-2.
        
        Conclusion
        Our conversational AI system demonstrates state-of-the-art performance.
        Future work will focus on reducing hallucinations and improving factual accuracy.
        
        References
        [1] Brown, T. et al. (2020). Language Models are Few-Shot Learners. arXiv:2005.14165
        [2] Vaswani, A. et al. (2017). Attention Is All You Need. NIPS 2017.
        """
        
        try:
            analysis = engine.analyze_paper(
                paper_id="test_paper",
                title="ChatGPT: A Conversational AI System",
                content=paper_content
            )
            
            # Validate comprehensive analysis
            has_references = len(analysis.references) > 0
            has_concepts = len(analysis.concepts) > 0
            has_topics = len(analysis.topics) > 0
            has_sections = len(analysis.sections) > 0
            has_summary = analysis.overall_summary is not None
            has_contributions = len(analysis.main_contributions) > 0
            
            # Check for expected elements
            reference_quality = any('brown' in r.raw_text.lower() or 'vaswani' in r.raw_text.lower() for r in analysis.references)
            concept_quality = any('transformer' in c.term.lower() or 'attention' in c.term.lower() for c in analysis.concepts)
            
            success = (has_references and has_concepts and has_topics and 
                      has_sections and has_summary and reference_quality and concept_quality)
            
            return {
                'success': success,
                'references': len(analysis.references),
                'concepts': len(analysis.concepts),
                'topics': len(analysis.topics),
                'sections': len(analysis.sections),
                'has_summary': has_summary,
                'contributions': len(analysis.main_contributions),
                'technical_level': analysis.technical_level,
                'reference_quality': reference_quality,
                'concept_quality': concept_quality
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}


def main():
    """Run validation tests for enhanced content analysis."""
    print("Enhanced Content Analysis Validation Suite")
    print("=" * 50)
    
    try:
        validator = ContentAnalysisValidator()
        results = validator.run_all_tests()
        
        # Export detailed results
        if len(sys.argv) > 1 and sys.argv[1] == '--export':
            import json
            output_file = 'content_analysis_test_results.json'
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nDetailed results exported to {output_file}")
        
        return 0 if results['overall_success'] else 1
        
    except Exception as e:
        print(f"Validation suite failed: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())