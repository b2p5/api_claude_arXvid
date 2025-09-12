"""
Advanced Content Analysis System for arXiv Papers
Implements reference extraction, concept detection, topic classification, and section summarization.
"""

import os
import re
import sqlite3
import json
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum

# Optional imports - make them fail gracefully
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    spacy = None
    SPACY_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.decomposition import LatentDirichletAllocation
    SKLEARN_AVAILABLE = True
except ImportError:
    TfidfVectorizer = None
    KMeans = None
    LatentDirichletAllocation = None
    SKLEARN_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False
from langchain_deepseek import ChatDeepSeek
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser

from config import get_config
from logger import get_logger, log_info, log_warning, log_error
from core.analysis import knowledge_graph


class SectionType(Enum):
    """Types of sections in academic papers."""
    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    RELATED_WORK = "related_work"
    METHODOLOGY = "methodology"
    RESULTS = "results"
    DISCUSSION = "discussion"
    CONCLUSION = "conclusion"
    REFERENCES = "references"
    UNKNOWN = "unknown"


@dataclass
class Reference:
    """A reference/citation found in a paper."""
    raw_text: str
    paper_title: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    venue: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    confidence: float = 0.0
    context: Optional[str] = None  # Context where citation appears
    
    
@dataclass
class Concept:
    """A key concept or technical term extracted from content."""
    term: str
    frequency: int
    importance_score: float
    context_examples: List[str] = field(default_factory=list)
    related_terms: List[str] = field(default_factory=list)
    definition: Optional[str] = None


@dataclass
class Topic:
    """A topic identified in the content."""
    name: str
    keywords: List[str]
    weight: float
    description: Optional[str] = None


@dataclass
class Section:
    """A section of a paper with its content and analysis."""
    section_type: SectionType
    title: str
    content: str
    summary: Optional[str] = None
    key_points: List[str] = field(default_factory=list)
    concepts: List[Concept] = field(default_factory=list)


@dataclass
class ContentAnalysis:
    """Complete content analysis results for a paper."""
    paper_id: str
    title: str
    references: List[Reference] = field(default_factory=list)
    concepts: List[Concept] = field(default_factory=list)
    topics: List[Topic] = field(default_factory=list)
    sections: List[Section] = field(default_factory=list)
    overall_summary: Optional[str] = None
    technical_level: str = "medium"  # "basic", "medium", "advanced"
    main_contributions: List[str] = field(default_factory=list)


class ReferenceExtractor:
    """Extracts and parses references/citations from paper content."""
    
    def __init__(self):
        self.logger = get_logger()
        
        # Enhanced citation patterns including modern formats
        self.citation_patterns = [
            # arXiv patterns (including new format)
            r'arXiv:(\d{4}\.\d{4,5}v?\d?)',
            r'https?://arxiv\.org/abs/(\d{4}\.\d{4,5}v?\d?)',
            # DOI patterns
            r'doi:?\s*(?:https?://(?:dx\.)?doi\.org/)?([^\s\]]+)',
            r'https?://doi\.org/([^\s\]]+)',
            # URL patterns for papers
            r'https?://(?:www\.)?(openreview\.net|proceedings\.mlr\.press|aclanthology\.org)/[^\s\]]+',
            # Standard citations (Author, Year) - improved
            r'([A-Z][a-z]+(?:\s+et\s+al\.?)?)\s*\((\d{4}[a-z]?)\)',
            # Multiple authors (Author1 & Author2, Year)
            r'([A-Z][a-z]+(?:\s*&\s*[A-Z][a-z]+)*)\s*\((\d{4}[a-z]?)\)',
            # Square bracket citations [1], [Author2020], [Author et al. 2020]
            r'\[([^\]]+)\]',
            # IEEE style citations (1), (Author, 2020), (Author et al., 2020)
            r'\(([^)]+(?:\d{4}[^)]*)?)\)',
            # Nature/Science style superscript references (would need HTML parsing)
            r'\^(\d+)',
        ]
        
        # Enhanced venue patterns with modern conferences
        self.venue_patterns = [
            r'(?:in|proc\.?)\s+([A-Z][A-Za-z\s&]+(?:Conference|Workshop|Symposium|Journal))',
            # Major AI/ML conferences
            r'(?:NeurIPS|ICML|ICLR|CVPR|ICCV|ECCV|WACV|BMVC)',
            # NLP conferences
            r'(?:ACL|EMNLP|NAACL|EACL|COLING|TACL|CoNLL)',
            # Journals and preprint servers
            r'(?:Nature|Science|Cell|PNAS|ArXiv|JMLR|TPAMI|IJCV)',
            # Specialized conferences
            r'(?:AAAI|IJCAI|AISTATS|UAI|KDD|SIGIR|WWW|CIKM)',
            # Robotics conferences
            r'(?:ICRA|IROS|RSS|CoRL)',
            # Theory conferences
            r'(?:STOC|FOCS|SODA|COLT)',
            # Systems conferences
            r'(?:OSDI|SOSP|NSDI|MLSys|SysML)',
        ]
    
    def extract_references(self, content: str, context_window: int = 100) -> List[Reference]:
        """Extract references from paper content."""
        references = []
        
        try:
            # Find potential citation contexts
            citation_contexts = self._find_citation_contexts(content, context_window)
            
            for context_text, citations in citation_contexts:
                for citation in citations:
                    ref = self._parse_citation(citation, context_text)
                    if ref:
                        references.append(ref)
            
            # Deduplicate references
            references = self._deduplicate_references(references)
            
            log_info("Reference extraction completed", 
                    references_found=len(references))
            
        except Exception as e:
            log_error("Reference extraction failed", error=str(e))
        
        return references
    
    def _find_citation_contexts(self, content: str, window: int) -> List[Tuple[str, List[str]]]:
        """Find citation contexts in the text."""
        contexts = []
        
        for pattern in self.citation_patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                start = max(0, match.start() - window)
                end = min(len(content), match.end() + window)
                context = content[start:end]
                citation = match.group(0)
                contexts.append((context, [citation]))
        
        return contexts
    
    def _parse_citation(self, citation: str, context: str) -> Optional[Reference]:
        """Parse a single citation string."""
        ref = Reference(raw_text=citation, context=context[:200])
        
        # Extract arXiv ID
        arxiv_match = re.search(r'arXiv:(\d{4}\.\d{4,5}v?\d?)', citation, re.IGNORECASE)
        if arxiv_match:
            ref.arxiv_id = arxiv_match.group(1)
            ref.confidence = 0.9
        
        # Extract DOI
        doi_match = re.search(r'doi:?\s*(?:https?://(?:dx\.)?doi\.org/)?([^\s\]]+)', citation, re.IGNORECASE)
        if doi_match:
            ref.doi = doi_match.group(1)
            ref.confidence = max(ref.confidence, 0.8)
        
        # Extract year
        year_match = re.search(r'(\d{4})', citation)
        if year_match:
            ref.year = int(year_match.group(1))
            ref.confidence = max(ref.confidence, 0.6)
        
        # Extract potential author names
        author_matches = re.findall(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', citation)
        if author_matches:
            ref.authors = [author.strip() for author in author_matches[:3]]  # Max 3 authors
            ref.confidence = max(ref.confidence, 0.5)
        
        # Extract venue information from context
        venue_match = None
        for pattern in self.venue_patterns:
            venue_match = re.search(pattern, context, re.IGNORECASE)
            if venue_match:
                ref.venue = venue_match.group(1).strip() if venue_match.groups() else venue_match.group(0)
                ref.confidence = max(ref.confidence, 0.7)
                break
        
        return ref if ref.confidence > 0.3 else None
    
    def _deduplicate_references(self, references: List[Reference]) -> List[Reference]:
        """Remove duplicate references."""
        unique_refs = []
        seen = set()
        
        for ref in references:
            # Create a signature for deduplication
            signature = (ref.arxiv_id, ref.doi, tuple(ref.authors), ref.year)
            if signature not in seen:
                seen.add(signature)
                unique_refs.append(ref)
        
        return unique_refs


class ConceptExtractor:
    """Extracts key concepts and technical terms from paper content."""
    
    def __init__(self):
        self.logger = get_logger()
        self.config = get_config()
        
        # Try to load spaCy model
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                log_warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
                self.nlp = None
        else:
            log_warning("spaCy not available. Install with: pip install spacy")
            self.nlp = None
        
        # Enhanced technical term patterns
        self.technical_patterns = [
            r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\s*(?:Algorithm|Model|Method|Framework|System|Network|Architecture|Transformer|Attention|Mechanism)\b',
            r'\b(?:deep|machine|artificial|neural|convolutional|recurrent|generative|adversarial|reinforcement)\s+[a-z]+(?:\s+[a-z]+)?\b',
            r'\b(?:BERT|GPT|ResNet|VGG|AlexNet|LSTM|GRU|CNN|RNN|GAN|VAE|CLIP|DALL-E|T5|BLEU|ROUGE)\b',  # Popular model names
            r'\b[A-Z]{2,}(?:-[A-Z]{2,})*\b',  # Acronyms
            r'\b\w+(?:_\w+)+\b',  # Snake_case terms
            r'\b(?:self-attention|multi-head|cross-entropy|backpropagation|gradient descent|dropout|batch normalization)\b',  # Technical concepts
            r'\b(?:fine-tuning|pre-training|transfer learning|zero-shot|few-shot|in-context learning)\b',  # Learning paradigms
        ]
        
        # Stop words for concept filtering
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 
            'by', 'from', 'this', 'that', 'these', 'those', 'is', 'are', 'was', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'cannot', 'fig', 'figure', 'table', 'section',
            'paper', 'approach', 'method', 'work', 'study', 'research', 'analysis', 'results'
        }
    
    def extract_concepts(self, content: str, min_frequency: int = 2) -> List[Concept]:
        """Extract key concepts from content."""
        concepts = []
        
        try:
            # Extract technical terms using patterns
            pattern_concepts = self._extract_pattern_concepts(content, min_frequency)
            
            # Extract NER concepts if spaCy is available
            if self.nlp:
                ner_concepts = self._extract_ner_concepts(content, min_frequency)
                concepts.extend(ner_concepts)
            
            concepts.extend(pattern_concepts)
            
            # Calculate importance scores
            concepts = self._calculate_importance_scores(concepts, content)
            
            # Deduplicate and sort
            concepts = self._deduplicate_concepts(concepts)
            concepts.sort(key=lambda c: c.importance_score, reverse=True)
            
            log_info("Concept extraction completed", 
                    concepts_found=len(concepts))
            
        except Exception as e:
            log_error("Concept extraction failed", error=str(e))
        
        return concepts
    
    def _extract_pattern_concepts(self, content: str, min_frequency: int) -> List[Concept]:
        """Extract concepts using regex patterns."""
        concepts = []
        term_counts = Counter()
        term_contexts = defaultdict(list)
        
        for pattern in self.technical_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                term = match.lower().strip()
                if len(term) > 2 and term not in self.stop_words:
                    term_counts[term] += 1
                    # Find context
                    context_match = re.search(rf'\b{re.escape(match)}\b.{{0,50}}', content, re.IGNORECASE)
                    if context_match:
                        term_contexts[term].append(context_match.group(0))
        
        for term, frequency in term_counts.items():
            if frequency >= min_frequency:
                concepts.append(Concept(
                    term=term,
                    frequency=frequency,
                    importance_score=0.0,  # Will be calculated later
                    context_examples=term_contexts[term][:3]
                ))
        
        return concepts
    
    def _extract_ner_concepts(self, content: str, min_frequency: int) -> List[Concept]:
        """Extract concepts using Named Entity Recognition."""
        concepts = []
        
        if not self.nlp:
            return concepts
        
        # Process in chunks to avoid memory issues
        chunk_size = 1000000  # 1MB chunks
        term_counts = Counter()
        term_contexts = defaultdict(list)
        
        for i in range(0, len(content), chunk_size):
            chunk = content[i:i + chunk_size]
            try:
                doc = self.nlp(chunk)
                
                for ent in doc.ents:
                    if ent.label_ in ["PRODUCT", "ORG", "GPE", "EVENT", "WORK_OF_ART"]:
                        term = ent.text.lower().strip()
                        if len(term) > 2 and term not in self.stop_words:
                            term_counts[term] += 1
                            term_contexts[term].append(ent.sent.text[:100])
                            
            except Exception as e:
                log_warning("NER processing chunk failed", error=str(e))
                continue
        
        for term, frequency in term_counts.items():
            if frequency >= min_frequency:
                concepts.append(Concept(
                    term=term,
                    frequency=frequency,
                    importance_score=0.0,
                    context_examples=term_contexts[term][:3]
                ))
        
        return concepts
    
    def _calculate_importance_scores(self, concepts: List[Concept], content: str) -> List[Concept]:
        """Calculate importance scores for concepts."""
        if not concepts:
            return concepts
        
        # Create TF-IDF vectorizer
        try:
            terms = [concept.term for concept in concepts]
            frequencies = [concept.frequency for concept in concepts]
            
            # Simple importance scoring based on frequency and term characteristics
            for i, concept in enumerate(concepts):
                score = 0.0
                
                # Frequency component (normalized)
                max_freq = max(frequencies) if frequencies else 1
                score += (concept.frequency / max_freq) * 0.4
                
                # Length component (longer terms often more specific)
                score += min(len(concept.term.split()) / 5.0, 1.0) * 0.3
                
                # Capitalization component (proper nouns, acronyms)
                if any(word[0].isupper() for word in concept.term.split() if word):
                    score += 0.2
                
                # Context diversity component
                unique_contexts = len(set(concept.context_examples))
                score += min(unique_contexts / 3.0, 1.0) * 0.1
                
                concept.importance_score = score
                
        except Exception as e:
            log_warning("Importance scoring failed", error=str(e))
            # Fallback to frequency-based scoring
            max_freq = max(c.frequency for c in concepts) if concepts else 1
            for concept in concepts:
                concept.importance_score = concept.frequency / max_freq
        
        return concepts
    
    def _deduplicate_concepts(self, concepts: List[Concept]) -> List[Concept]:
        """Remove duplicate concepts."""
        unique_concepts = {}
        
        for concept in concepts:
            # Use normalized term as key
            key = concept.term.lower().strip()
            if key not in unique_concepts or concept.importance_score > unique_concepts[key].importance_score:
                unique_concepts[key] = concept
        
        return list(unique_concepts.values())


class TopicClassifier:
    """Automatically classifies papers by topic using unsupervised methods."""
    
    def __init__(self, n_topics: int = 10):
        self.logger = get_logger()
        self.n_topics = n_topics
        self.vectorizer = None
        self.lda_model = None
        self.kmeans_model = None
        
        # Enhanced predefined topic keywords for better labeling
        self.topic_keywords = {
            'machine_learning': ['machine learning', 'deep learning', 'neural network', 'algorithm', 'model', 'training', 'supervised', 'unsupervised'],
            'nlp': ['natural language', 'text', 'language model', 'nlp', 'linguistics', 'transformer', 'bert', 'gpt', 'tokenization'],
            'computer_vision': ['computer vision', 'image', 'visual', 'cnn', 'object detection', 'segmentation', 'classification', 'resnet'],
            'generative_ai': ['generative', 'gan', 'vae', 'diffusion', 'stable diffusion', 'dall-e', 'gpt', 'generation'],
            'robotics': ['robot', 'robotics', 'control', 'autonomous', 'manipulation', 'navigation', 'sensor', 'actuator'],
            'reinforcement_learning': ['reinforcement learning', 'rl', 'policy', 'reward', 'q-learning', 'actor-critic', 'exploration'],
            'ai_systems': ['artificial intelligence', 'ai system', 'intelligent', 'autonomous', 'agent', 'multi-agent'],
            'optimization': ['optimization', 'algorithm', 'solution', 'problem', 'performance', 'gradient', 'convergence'],
            'data_science': ['data', 'analysis', 'statistics', 'dataset', 'mining', 'big data', 'analytics', 'visualization'],
            'security': ['security', 'privacy', 'attack', 'defense', 'vulnerability', 'encryption', 'adversarial'],
            'theory': ['theoretical', 'mathematical', 'proof', 'theorem', 'analysis', 'complexity', 'bounds'],
            'multimodal': ['multimodal', 'cross-modal', 'vision-language', 'audio-visual', 'fusion', 'clip'],
            'applications': ['application', 'real-world', 'practical', 'implementation', 'system', 'deployment', 'production']
        }
    
    def classify_topics(self, papers: List[Dict[str, str]]) -> Dict[str, List[Topic]]:
        """Classify papers by topic using multiple methods."""
        topics_by_paper = {}
        
        try:
            if not papers:
                return topics_by_paper
            
            # Prepare texts
            texts = []
            paper_ids = []
            for paper in papers:
                text = f"{paper.get('title', '')} {paper.get('summary', '')} {paper.get('content', '')}"
                texts.append(text)
                paper_ids.append(paper.get('id', ''))
            
            # Check if scikit-learn is available
            if not SKLEARN_AVAILABLE:
                log_error("scikit-learn not available. Install with: pip install scikit-learn")
                return topics_by_paper
            
            # Handle special case of single paper
            n_docs = len(texts)
            if n_docs == 1:
                # For single paper, create simple topic based on most frequent words
                return self._classify_single_paper_fallback(papers[0], paper_ids[0])
            
            # Vectorize texts with parameters adjusted for small datasets
            try:
                self.vectorizer = TfidfVectorizer(
                    max_features=min(1000, n_docs * 50),
                    stop_words='english',
                    ngram_range=(1, 2),
                    min_df=1,  # Always 1 for small datasets
                    max_df=0.95  # More lenient for small datasets
                )
                
                X = self.vectorizer.fit_transform(texts)
                feature_names = self.vectorizer.get_feature_names_out()
            except ValueError as e:
                if "max_df corresponds to < documents than min_df" in str(e):
                    # Fallback for very small datasets
                    log_warning("Using basic TF-IDF for very small dataset")
                    self.vectorizer = TfidfVectorizer(
                        max_features=100,
                        stop_words='english',
                        min_df=1,
                        max_df=1.0,
                        ngram_range=(1, 1)  # Unigrams only
                    )
                    X = self.vectorizer.fit_transform(texts)
                    feature_names = self.vectorizer.get_feature_names_out()
                else:
                    raise
            
            # LDA Topic Modeling
            lda_topics = self._classify_with_lda(X, feature_names, paper_ids)
            
            # K-Means Clustering
            kmeans_topics = self._classify_with_kmeans(X, feature_names, paper_ids)
            
            # Combine results
            for paper_id in paper_ids:
                combined_topics = []
                if paper_id in lda_topics:
                    combined_topics.extend(lda_topics[paper_id])
                if paper_id in kmeans_topics:
                    combined_topics.extend(kmeans_topics[paper_id])
                topics_by_paper[paper_id] = combined_topics
            
            log_info("Topic classification completed", 
                    papers_classified=len(topics_by_paper))
            
        except Exception as e:
            log_error("Topic classification failed", error=str(e))
        
        return topics_by_paper
    
    def _classify_single_paper_fallback(self, paper: Dict[str, str], paper_id: str) -> Dict[str, List[Topic]]:
        """Fallback classification for single paper using keyword matching."""
        text = f"{paper.get('title', '')} {paper.get('summary', '')} {paper.get('content', '')}"
        text_lower = text.lower()
        
        matched_topics = []
        
        # Match against predefined topics
        for topic_name, keywords in self.topic_keywords.items():
            score = 0
            matched_keywords = []
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1
                    matched_keywords.append(keyword)
            
            if score > 0:
                topic = Topic(
                    name=topic_name.replace('_', ' ').title(),
                    keywords=matched_keywords,
                    weight=score / len(keywords),
                    description=f"Topic identified by keyword matching (score: {score}/{len(keywords)})"
                )
                matched_topics.append(topic)
        
        # Sort by weight and take top topics
        matched_topics.sort(key=lambda t: t.weight, reverse=True)
        return {paper_id: matched_topics[:3]}  # Return top 3 topics
    
    def _classify_with_lda(self, X, feature_names: List[str], paper_ids: List[str]) -> Dict[str, List[Topic]]:
        """Classify using Latent Dirichlet Allocation."""
        topics_by_paper = {}
        
        try:
            if not SKLEARN_AVAILABLE:
                log_warning("scikit-learn not available for LDA classification")
                return topics_by_paper
            
            self.lda_model = LatentDirichletAllocation(
                n_components=min(self.n_topics, len(paper_ids)),
                random_state=42,
                max_iter=100
            )
            
            doc_topic_probs = self.lda_model.fit_transform(X)
            
            # Get topic keywords
            topic_keywords_lda = []
            for topic_idx in range(self.lda_model.n_components):
                top_keywords_idx = self.lda_model.components_[topic_idx].argsort()[-10:][::-1]
                keywords = [feature_names[i] for i in top_keywords_idx]
                topic_keywords_lda.append(keywords)
            
            # Assign topics to papers
            for i, paper_id in enumerate(paper_ids):
                paper_topics = []
                topic_probs = doc_topic_probs[i]
                
                # Get topics with probability > threshold
                threshold = 0.1
                for topic_idx, prob in enumerate(topic_probs):
                    if prob > threshold:
                        topic_name = self._generate_topic_name(topic_keywords_lda[topic_idx])
                        topic = Topic(
                            name=f"LDA_{topic_name}",
                            keywords=topic_keywords_lda[topic_idx][:5],
                            weight=float(prob),
                            description=f"Topic identified by LDA with {prob:.2f} probability"
                        )
                        paper_topics.append(topic)
                
                topics_by_paper[paper_id] = paper_topics
                
        except Exception as e:
            log_warning("LDA classification failed", error=str(e))
        
        return topics_by_paper
    
    def _classify_with_kmeans(self, X, feature_names: List[str], paper_ids: List[str]) -> Dict[str, List[Topic]]:
        """Classify using K-Means clustering."""
        topics_by_paper = {}
        
        try:
            if not SKLEARN_AVAILABLE:
                log_warning("scikit-learn not available for K-Means classification")
                return topics_by_paper
            
            n_clusters = min(self.n_topics, len(paper_ids))
            self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = self.kmeans_model.fit_predict(X)
            
            # Get cluster centers and keywords
            cluster_keywords = []
            for cluster_idx in range(n_clusters):
                center = self.kmeans_model.cluster_centers_[cluster_idx]
                top_keywords_idx = center.argsort()[-10:][::-1]
                keywords = [feature_names[i] for i in top_keywords_idx]
                cluster_keywords.append(keywords)
            
            # Assign clusters to papers
            for i, paper_id in enumerate(paper_ids):
                cluster_id = cluster_labels[i]
                topic_name = self._generate_topic_name(cluster_keywords[cluster_id])
                
                # Calculate distance to cluster center as weight
                if NUMPY_AVAILABLE:
                    distance = np.linalg.norm(X[i].toarray() - self.kmeans_model.cluster_centers_[cluster_id])
                    weight = max(0.1, 1.0 - (distance / np.max(self.kmeans_model.cluster_centers_)))
                else:
                    # Simple fallback without numpy
                    weight = 0.5
                
                topic = Topic(
                    name=f"Cluster_{topic_name}",
                    keywords=cluster_keywords[cluster_id][:5],
                    weight=float(weight),
                    description=f"Topic identified by K-Means clustering"
                )
                
                topics_by_paper[paper_id] = [topic]
                
        except Exception as e:
            log_warning("K-Means classification failed", error=str(e))
        
        return topics_by_paper
    
    def _generate_topic_name(self, keywords: List[str]) -> str:
        """Generate a readable topic name from keywords."""
        # Try to match with predefined topics
        for topic_name, topic_words in self.topic_keywords.items():
            overlap = sum(1 for kw in keywords[:5] if any(tw in kw.lower() for tw in topic_words))
            if overlap >= 2:
                return topic_name.replace('_', ' ').title()
        
        # Fallback to first two keywords
        if len(keywords) >= 2:
            return f"{keywords[0]}_{keywords[1]}"
        elif keywords:
            return keywords[0]
        else:
            return "Unknown"


class SectionAnalyzer:
    """Analyzes and summarizes different sections of papers."""
    
    def __init__(self):
        self.logger = get_logger()
        self.config = get_config()
        
        # Enhanced section identification patterns
        self.section_patterns = {
            SectionType.ABSTRACT: [r'\babstract\b', r'\bsummary\b', r'\boverview\b'],
            SectionType.INTRODUCTION: [r'\bintroduction\b', r'\bintroducing\b', r'\bmotivation\b'],
            SectionType.RELATED_WORK: [r'\brelated\s+work\b', r'\bliterature\s+review\b', r'\bbackground\b', r'\bprior\s+work\b', r'\bprevious\s+work\b'],
            SectionType.METHODOLOGY: [r'\bmethodology\b', r'\bmethod\b', r'\bapproach\b', r'\balgorithm\b', r'\bmodel\b', r'\barchitecture\b', r'\btechnique\b'],
            SectionType.RESULTS: [r'\bresults\b', r'\bexperiments\b', r'\bevaluation\b', r'\banalysis\b', r'\bfindings\b', r'\bperformance\b'],
            SectionType.DISCUSSION: [r'\bdiscussion\b', r'\banalysis\b', r'\bablation\b', r'\binsights\b', r'\blimitations\b'],
            SectionType.CONCLUSION: [r'\bconclusion\b', r'\bconcluding\b', r'\bsummary\b', r'\bfuture\s+work\b', r'\bfinal\s+remarks\b'],
            SectionType.REFERENCES: [r'\breferences\b', r'\bbibliography\b', r'\bcitations\b']
        }
        
        # Initialize LLM for summarization
        try:
            self.llm = ChatDeepSeek(model=self.config.models.llm_model)
        except Exception as e:
            log_warning("LLM initialization failed for section analysis", error=str(e))
            self.llm = None
    
    def analyze_sections(self, content: str, title: str = "") -> List[Section]:
        """Analyze and extract sections from paper content."""
        sections = []
        
        try:
            # Split content into potential sections
            raw_sections = self._split_into_sections(content)
            
            for section_text, section_title in raw_sections:
                section_type = self._classify_section_type(section_title, section_text)
                
                # Create section object
                section = Section(
                    section_type=section_type,
                    title=section_title or section_type.value.replace('_', ' ').title(),
                    content=section_text
                )
                
                # Generate summary if LLM is available
                if self.llm and len(section_text) > 200:
                    section.summary = self._generate_section_summary(section_text, section_type)
                
                # Extract key points
                section.key_points = self._extract_key_points(section_text, section_type)
                
                sections.append(section)
            
            log_info("Section analysis completed", 
                    sections_found=len(sections))
            
        except Exception as e:
            log_error("Section analysis failed", error=str(e))
        
        return sections
    
    def _split_into_sections(self, content: str) -> List[Tuple[str, str]]:
        """Split content into sections based on headers and structure."""
        sections = []
        
        # Look for section headers (numbers, capitalized lines, etc.)
        section_delimiters = [
            r'\n\s*(\d+\.?\s+[A-Z][A-Za-z\s]+)\n',  # "1. Introduction"
            r'\n\s*([A-Z][A-Z\s]{5,})\n',  # "INTRODUCTION"
            r'\n\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\n(?=\s*[A-Z])',  # "Introduction"
        ]
        
        splits = []
        for pattern in section_delimiters:
            for match in re.finditer(pattern, content):
                splits.append((match.start(), match.group(1).strip()))
        
        splits.sort(key=lambda x: x[0])
        
        if splits:
            # Add sections based on splits
            for i, (start_pos, header) in enumerate(splits):
                end_pos = splits[i + 1][0] if i + 1 < len(splits) else len(content)
                section_content = content[start_pos:end_pos].strip()
                sections.append((section_content, header))
        else:
            # No clear sections found, treat as single section
            sections.append((content, ""))
        
        return sections
    
    def _classify_section_type(self, title: str, content: str) -> SectionType:
        """Classify the type of section based on title and content."""
        title_lower = title.lower()
        content_sample = content[:500].lower()  # First 500 chars
        
        for section_type, patterns in self.section_patterns.items():
            for pattern in patterns:
                if re.search(pattern, title_lower) or re.search(pattern, content_sample):
                    return section_type
        
        return SectionType.UNKNOWN
    
    def _generate_section_summary(self, content: str, section_type: SectionType) -> str:
        """Generate a summary for a section using LLM."""
        if not self.llm:
            return ""
        
        try:
            # Create section-specific prompt
            prompts = {
                SectionType.ABSTRACT: "Summarize this abstract in 2-3 sentences, focusing on the main contribution and results:",
                SectionType.INTRODUCTION: "Summarize this introduction, highlighting the problem being addressed and the approach:",
                SectionType.METHODOLOGY: "Summarize the methodology, focusing on the key techniques and approaches used:",
                SectionType.RESULTS: "Summarize the main results and findings presented in this section:",
                SectionType.CONCLUSION: "Summarize the conclusions and main takeaways:",
            }
            
            prompt_text = prompts.get(section_type, "Provide a concise summary of this section:")
            
            template = PromptTemplate.from_template(f"{prompt_text}\n\n{{content}}\n\nSummary:")
            chain = template | self.llm | StrOutputParser()
            
            summary = chain.invoke({"content": content[:2000]})  # Limit content length
            return summary.strip()
            
        except Exception as e:
            log_warning("Section summary generation failed", error=str(e))
            return ""
    
    def _extract_key_points(self, content: str, section_type: SectionType) -> List[str]:
        """Extract key points from section content using heuristics."""
        key_points = []
        
        # Look for bullet points, numbered lists, or sentences with key indicators
        patterns = [
            r'[•\-\*]\s+([^\n]+)',  # Bullet points
            r'\d+\.\s+([^\n]+)',    # Numbered lists
            r'\b(?:importantly|significantly|notably|furthermore|moreover|however|therefore)\s+([^.]+\.)',  # Key transition words
            r'\b(?:we\s+(?:propose|present|introduce|show|demonstrate|find))\s+([^.]+\.)',  # Key contributions
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches[:3]:  # Limit to 3 per pattern
                point = match.strip()
                if len(point) > 20 and point not in key_points:
                    key_points.append(point)
        
        return key_points[:5]  # Maximum 5 key points per section


class ContentAnalysisEngine:
    """Main engine that orchestrates all content analysis components."""
    
    def __init__(self):
        self.logger = get_logger()
        self.config = get_config()
        
        # Initialize components
        self.reference_extractor = ReferenceExtractor()
        self.concept_extractor = ConceptExtractor()
        self.topic_classifier = TopicClassifier()
        self.section_analyzer = SectionAnalyzer()
        
        log_info("Content analysis engine initialized")
    
    def analyze_paper(self, paper_id: str, title: str, content: str) -> ContentAnalysis:
        """Perform complete content analysis on a single paper."""
        log_info("Starting content analysis", paper_id=paper_id)
        
        analysis = ContentAnalysis(paper_id=paper_id, title=title)
        
        try:
            # Extract references
            analysis.references = self.reference_extractor.extract_references(content)
            
            # Extract concepts
            analysis.concepts = self.concept_extractor.extract_concepts(content)
            
            # Analyze sections
            analysis.sections = self.section_analyzer.analyze_sections(content, title)
            
            # Classify topics (single paper)
            paper_data = [{"id": paper_id, "title": title, "content": content}]
            topics_result = self.topic_classifier.classify_topics(paper_data)
            analysis.topics = topics_result.get(paper_id, [])
            
            # Generate overall analysis
            analysis.overall_summary = self._generate_overall_summary(analysis)
            analysis.technical_level = self._assess_technical_level(content, analysis.concepts)
            analysis.main_contributions = self._extract_main_contributions(content, analysis.sections)
            
            log_info("Content analysis completed", 
                    paper_id=paper_id,
                    references=len(analysis.references),
                    concepts=len(analysis.concepts),
                    topics=len(analysis.topics),
                    sections=len(analysis.sections))
            
        except Exception as e:
            log_error("Content analysis failed", paper_id=paper_id, error=str(e))
        
        return analysis
    
    def analyze_corpus(self, papers: List[Dict[str, str]]) -> Dict[str, ContentAnalysis]:
        """Analyze multiple papers and find cross-paper relationships."""
        analyses = {}
        
        try:
            # Classify topics across all papers first
            topics_by_paper = self.topic_classifier.classify_topics(papers)
            
            # Analyze each paper individually
            for paper in papers:
                paper_id = paper.get('id', '')
                title = paper.get('title', '')
                content = paper.get('content', '')
                
                if paper_id and content:
                    analysis = self.analyze_paper(paper_id, title, content)
                    # Update with corpus-level topics
                    if paper_id in topics_by_paper:
                        analysis.topics = topics_by_paper[paper_id]
                    analyses[paper_id] = analysis
            
            # Find cross-paper relationships
            self._find_cross_paper_relationships(analyses)
            
            log_info("Corpus analysis completed", papers_analyzed=len(analyses))
            
        except Exception as e:
            log_error("Corpus analysis failed", error=str(e))
        
        return analyses
    
    def _generate_overall_summary(self, analysis: ContentAnalysis) -> str:
        """Generate an overall summary of the paper analysis."""
        summary_parts = []
        
        if analysis.sections:
            abstract_sections = [s for s in analysis.sections if s.section_type == SectionType.ABSTRACT]
            if abstract_sections and abstract_sections[0].summary:
                summary_parts.append(abstract_sections[0].summary)
        
        if analysis.concepts:
            top_concepts = sorted(analysis.concepts, key=lambda c: c.importance_score, reverse=True)[:5]
            concepts_text = ", ".join([c.term for c in top_concepts])
            summary_parts.append(f"Key concepts: {concepts_text}")
        
        if analysis.topics:
            topics_text = ", ".join([t.name for t in analysis.topics[:3]])
            summary_parts.append(f"Topics: {topics_text}")
        
        return " | ".join(summary_parts) if summary_parts else "Analysis completed."
    
    def _assess_technical_level(self, content: str, concepts: List[Concept]) -> str:
        """Assess the technical level of the paper."""
        # Simple heuristic based on technical concepts and mathematical content
        math_patterns = len(re.findall(r'[∑∫∂∇αβγδεθλμπρσφψω]|\\[a-zA-Z]+', content))
        technical_concepts = len([c for c in concepts if c.importance_score > 0.7])
        
        score = math_patterns * 0.1 + technical_concepts * 0.3
        
        if score > 10:
            return "advanced"
        elif score > 5:
            return "medium"
        else:
            return "basic"
    
    def _extract_main_contributions(self, content: str, sections: List[Section]) -> List[str]:
        """Extract main contributions mentioned in the paper."""
        contributions = []
        
        # Look for contribution indicators
        contribution_patterns = [
            r'(?:our|this)\s+(?:main\s+)?contribution[s]?\s+(?:is|are|include)[s]?\s*:?\s*([^.]+\.)',
            r'we\s+(?:propose|present|introduce|develop)\s+([^.]+\.)',
            r'(?:novel|new|first)\s+([^.]+\.)',
            r'(?:in\s+summary|to\s+summarize)[,:]?\s*([^.]+\.)',
        ]
        
        for pattern in contribution_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches[:2]:  # Limit per pattern
                contrib = match.strip()
                if len(contrib) > 20:
                    contributions.append(contrib)
        
        return contributions[:3]  # Maximum 3 contributions
    
    def _find_cross_paper_relationships(self, analyses: Dict[str, ContentAnalysis]):
        """Find relationships between papers in the corpus."""
        # This could be expanded to find citation relationships,
        # shared concepts, similar topics, etc.
        
        # For now, we'll add related concepts based on overlap
        paper_ids = list(analyses.keys())
        
        for i, paper_id1 in enumerate(paper_ids):
            for j, paper_id2 in enumerate(paper_ids[i+1:], i+1):
                analysis1 = analyses[paper_id1]
                analysis2 = analyses[paper_id2]
                
                # Find overlapping concepts
                concepts1 = {c.term.lower() for c in analysis1.concepts}
                concepts2 = {c.term.lower() for c in analysis2.concepts}
                overlap = concepts1.intersection(concepts2)
                
                if len(overlap) >= 2:  # Significant overlap
                    # Add related terms to concepts
                    for concept1 in analysis1.concepts:
                        if concept1.term.lower() in overlap:
                            concept1.related_terms.extend([
                                c.term for c in analysis2.concepts 
                                if c.term.lower() in overlap and c.term != concept1.term
                            ])


def create_content_analysis_engine() -> ContentAnalysisEngine:
    """Create and return configured content analysis engine instance."""
    return ContentAnalysisEngine()