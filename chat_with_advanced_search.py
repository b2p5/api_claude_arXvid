"""
Advanced Chat System with Enhanced Search Capabilities
Integrates the advanced search engine with the chat interface for better paper retrieval.
"""

import os
import argparse
import json
from datetime import date
from typing import List, Dict, Any, Optional, Tuple

import dotenv
from langchain_deepseek import ChatDeepSeek
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

from config import get_config
from logger import get_logger, log_info, log_warning, log_error
from advanced_search import AdvancedSearchEngine, SearchMode, SearchResult
from search_filters import AdvancedSearchFilters, DateRangePreset, SmartFilterEngine
from search_ranking import RankingStrategy, RankingWeights, RankingConfig


class AdvancedChatRAG:
    """Enhanced RAG chat system with advanced search capabilities."""
    
    def __init__(self, db_path: str = None, model_name: str = None):
        """Initialize the advanced chat RAG system."""
        self.config = get_config()
        self.logger = get_logger()
        
        # Initialize search components
        self.search_engine = AdvancedSearchEngine(db_path, model_name)
        self.filter_engine = SmartFilterEngine()
        
        # Initialize LLM
        self.llm = ChatDeepSeek(model=self.config.models.llm_model)
        
        # Chat history and context
        self.chat_history = []
        self.user_preferences = {
            'search_focus': 'balanced',  # 'semantic', 'keyword', 'recent', 'authoritative'
            'preferred_categories': [],
            'preferred_authors': [],
            'max_results': 5,
            'min_relevance': 0.3
        }
        
        log_info("Advanced chat RAG system initialized")
    
    def chat(self, user_query: str, search_options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process user query with advanced search and generate response.
        
        Args:
            user_query: User's question or query
            search_options: Advanced search options and filters
            
        Returns:
            Dictionary containing response and search metadata
        """
        log_info("Processing chat query", query=user_query[:100])
        
        search_options = search_options or {}
        
        try:
            # Analyze query to determine search strategy
            search_strategy = self._analyze_query_intent(user_query)
            
            # Extract search parameters from query
            extracted_params = self._extract_search_parameters(user_query)
            
            # Combine with explicit search options
            final_search_options = {**extracted_params, **search_options}
            
            # Perform advanced search
            search_results = self._perform_advanced_search(
                user_query, 
                search_strategy, 
                final_search_options
            )
            
            # Generate context from search results
            context = self._format_search_context(search_results)
            
            # Generate response using LLM
            response = self._generate_llm_response(user_query, context, search_results)
            
            # Update chat history
            self._update_chat_history(user_query, response, search_results)
            
            # Return comprehensive result
            return {
                'response': response,
                'search_results': search_results,
                'search_strategy': search_strategy,
                'context_used': len(search_results) > 0,
                'total_results_found': len(search_results)
            }
            
        except Exception as e:
            log_error("Chat processing failed", error=str(e))
            return {
                'response': f"I encountered an error while processing your query: {str(e)}",
                'search_results': [],
                'search_strategy': None,
                'context_used': False,
                'total_results_found': 0
            }
    
    def _analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Analyze query to determine optimal search strategy."""
        query_lower = query.lower()
        
        strategy = {
            'search_mode': SearchMode.HYBRID,
            'ranking_strategy': RankingStrategy.HYBRID_SCORE,
            'focus_areas': []
        }
        
        # Detect specific search intents
        if any(word in query_lower for word in ['recent', 'latest', 'new', '2024', '2023']):
            strategy['ranking_strategy'] = RankingStrategy.RECENCY
            strategy['focus_areas'].append('recency')
        
        if any(word in query_lower for word in ['author:', 'by ', 'written by', 'from ']):
            strategy['focus_areas'].append('author')
        
        if any(word in query_lower for word in ['similar to', 'like', 'semantic', 'meaning']):
            strategy['search_mode'] = SearchMode.SEMANTIC
            strategy['focus_areas'].append('semantic')
        
        if any(word in query_lower for word in ['keyword', 'contains', 'mentions', 'about']):
            strategy['search_mode'] = SearchMode.KEYWORD
            strategy['focus_areas'].append('keyword')
        
        if any(word in query_lower for word in ['popular', 'cited', 'influential', 'important']):
            strategy['ranking_strategy'] = RankingStrategy.CITATION_COUNT
            strategy['focus_areas'].append('citations')
        
        log_info("Query intent analyzed", strategy=strategy)
        return strategy
    
    def _extract_search_parameters(self, query: str) -> Dict[str, Any]:
        """Extract search parameters from natural language query."""
        params = {}
        query_lower = query.lower()
        
        # Extract author information
        author_patterns = [
            r'by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'author:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'from\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
        ]
        
        import re
        authors = []
        for pattern in author_patterns:
            matches = re.findall(pattern, query)
            authors.extend(matches)
        
        if authors:
            params['authors'] = authors
        
        # Extract date/time filters
        if 'last year' in query_lower or '2023' in query:
            params['date_preset'] = DateRangePreset.LAST_YEAR
        elif 'last month' in query_lower:
            params['date_preset'] = DateRangePreset.LAST_MONTH
        elif 'recent' in query_lower or 'latest' in query_lower:
            params['date_preset'] = DateRangePreset.LAST_6_MONTHS
        
        # Extract category hints
        category_hints = {
            'machine learning': ['cs.lg', 'stat.ml'],
            'computer vision': ['cs.cv'],
            'nlp': ['cs.cl'],
            'natural language': ['cs.cl'],
            'ai': ['cs.ai'],
            'artificial intelligence': ['cs.ai'],
            'statistics': ['stat.ml', 'math.st'],
            'physics': ['physics'],
            'mathematics': ['math']
        }
        
        categories = []
        for hint, cats in category_hints.items():
            if hint in query_lower:
                categories.extend(cats)
        
        if categories:
            params['arxiv_categories'] = list(set(categories))
        
        # Extract result preferences
        if 'few' in query_lower or 'brief' in query_lower:
            params['max_results'] = 3
        elif 'detailed' in query_lower or 'comprehensive' in query_lower:
            params['max_results'] = 10
        
        log_info("Search parameters extracted", params=params)
        return params
    
    def _perform_advanced_search(
        self, 
        query: str, 
        strategy: Dict[str, Any], 
        search_options: Dict[str, Any]
    ) -> List[SearchResult]:
        """Perform advanced search with extracted parameters."""
        
        # Create search filters
        filters = AdvancedSearchFilters(
            authors=search_options.get('authors'),
            date_preset=search_options.get('date_preset'),
            arxiv_categories=search_options.get('arxiv_categories'),
            max_results=search_options.get('max_results', self.user_preferences['max_results']),
            min_relevance_score=search_options.get('min_relevance', self.user_preferences['min_relevance'])
        )
        
        # Create search config based on strategy
        search_config = self._create_search_config(strategy)
        
        # Perform search
        search_results = self.search_engine.search(
            query=query,
            mode=strategy['search_mode'],
            filters=filters,
            ranking=strategy['ranking_strategy'],
            search_config=search_config
        )
        
        log_info("Advanced search completed", 
                results_count=len(search_results),
                search_mode=strategy['search_mode'].value)
        
        return search_results
    
    def _create_search_config(self, strategy: Dict[str, Any]):
        """Create search configuration based on search strategy."""
        from advanced_search import SearchConfig
        
        # Create basic config
        config = SearchConfig()
        
        # Adjust weights based on focus areas
        focus_areas = strategy.get('focus_areas', [])
        
        if 'semantic' in focus_areas:
            config.semantic_weight = 0.8
            config.keyword_weight = 0.2
        elif 'keyword' in focus_areas:
            config.keyword_weight = 0.7
            config.semantic_weight = 0.3
        elif 'recency' in focus_areas:
            config.boost_recent_papers = 1.5
        
        # User preference adjustments
        if self.user_preferences['search_focus'] == 'semantic':
            config.semantic_weight = min(config.semantic_weight + 0.1, 0.9)
            config.keyword_weight = 1.0 - config.semantic_weight
        elif self.user_preferences['search_focus'] == 'recent':
            config.boost_recent_papers = 2.0
        
        return config
    
    def _format_search_context(self, search_results: List[SearchResult]) -> str:
        """Format search results into context for LLM."""
        if not search_results:
            return "No relevant papers were found in the database."
        
        context_parts = []
        
        for i, result in enumerate(search_results[:self.user_preferences['max_results']], 1):
            paper_context = f"[Paper {i}]"
            
            if result.title:
                paper_context += f" Title: {result.title}"
            
            if result.authors:
                paper_context += f" | Authors: {', '.join(result.authors[:3])}"
                if len(result.authors) > 3:
                    paper_context += " et al."
            
            if result.publication_date:
                paper_context += f" | Date: {result.publication_date}"
            
            if result.arxiv_id:
                paper_context += f" | arXiv: {result.arxiv_id}"
            
            paper_context += f" | Relevance: {result.final_score:.2f}"
            paper_context += f"\nContent: {result.content[:500]}..."
            
            if hasattr(result, 'ranking_explanation') and result.ranking_explanation:
                paper_context += f"\nRanking factors: {result.ranking_explanation}"
            
            context_parts.append(paper_context)
        
        context = "\n\n".join(context_parts)
        
        # Add search metadata
        context += f"\n\n[Search Summary: Found {len(search_results)} relevant papers, "
        context += f"showing top {min(len(search_results), self.user_preferences['max_results'])}]"
        
        return context
    
    def _generate_llm_response(
        self, 
        query: str, 
        context: str, 
        search_results: List[SearchResult]
    ) -> str:
        """Generate response using LLM with enhanced context."""
        
        # Choose prompt template based on query type
        if self._is_comparison_query(query):
            template = self._get_comparison_prompt_template()
        elif self._is_summary_query(query):
            template = self._get_summary_prompt_template()
        elif self._is_recommendation_query(query):
            template = self._get_recommendation_prompt_template()
        else:
            template = self._get_general_prompt_template()
        
        prompt = PromptTemplate.from_template(template)
        
        # Create RAG chain
        rag_chain = (
            {"context": lambda x: context, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        try:
            response = rag_chain.invoke(query)
            
            # Add search metadata to response
            if search_results:
                response += f"\n\n---\nSearch Info: Found {len(search_results)} papers"
                if search_results[0].final_score > 0.8:
                    response += " (high relevance)"
                elif search_results[0].final_score > 0.5:
                    response += " (moderate relevance)"
                
                # Add paper references
                response += "\nKey Papers:"
                for i, result in enumerate(search_results[:3], 1):
                    response += f"\n{i}. "
                    if result.title:
                        response += f"{result.title}"
                    if result.authors:
                        response += f" - {result.authors[0]}"
                        if len(result.authors) > 1:
                            response += " et al."
                    if result.arxiv_id:
                        response += f" (arXiv:{result.arxiv_id})"
            
            return response
            
        except Exception as e:
            log_error("LLM response generation failed", error=str(e))
            return f"I found relevant information but encountered an error generating the response: {str(e)}"
    
    def _is_comparison_query(self, query: str) -> bool:
        """Check if query is asking for comparison."""
        comparison_keywords = ['compare', 'difference', 'versus', 'vs', 'better', 'advantage']
        return any(keyword in query.lower() for keyword in comparison_keywords)
    
    def _is_summary_query(self, query: str) -> bool:
        """Check if query is asking for summary."""
        summary_keywords = ['summarize', 'overview', 'what is', 'explain', 'describe']
        return any(keyword in query.lower() for keyword in summary_keywords)
    
    def _is_recommendation_query(self, query: str) -> bool:
        """Check if query is asking for recommendations."""
        recommendation_keywords = ['recommend', 'suggest', 'best', 'should I read', 'good papers']
        return any(keyword in query.lower() for keyword in recommendation_keywords)
    
    def _get_general_prompt_template(self) -> str:
        """Get general purpose prompt template."""
        return """
You are a knowledgeable research assistant specializing in academic papers. Answer the question based on the provided research papers context.

CONTEXT FROM RESEARCH PAPERS:
{context}

QUESTION:
{question}

Please provide a comprehensive answer that:
1. Directly addresses the question
2. References specific papers when relevant
3. Synthesizes information from multiple sources if applicable
4. Indicates confidence level in your answer
5. Suggests related topics if helpful

ANSWER:
"""
    
    def _get_comparison_prompt_template(self) -> str:
        """Get comparison-focused prompt template."""
        return """
You are a research assistant expert at comparing academic approaches. Based on the provided papers, create a detailed comparison.

CONTEXT FROM RESEARCH PAPERS:
{context}

COMPARISON REQUEST:
{question}

Please provide a structured comparison that includes:
1. Key similarities between the approaches/papers
2. Main differences and their implications
3. Strengths and weaknesses of each approach
4. Use cases where each might be preferred
5. Overall assessment with evidence from the papers

COMPARISON ANALYSIS:
"""
    
    def _get_summary_prompt_template(self) -> str:
        """Get summary-focused prompt template."""
        return """
You are a research assistant skilled at synthesizing academic literature. Provide a comprehensive summary based on the research papers.

CONTEXT FROM RESEARCH PAPERS:
{context}

SUMMARY REQUEST:
{question}

Please provide a well-structured summary that includes:
1. Main concepts and key findings
2. Methodological approaches used
3. Important results and their significance
4. Current state of the field based on these papers
5. Future directions or open questions mentioned

SUMMARY:
"""
    
    def _get_recommendation_prompt_template(self) -> str:
        """Get recommendation-focused prompt template."""
        return """
You are a research advisor helping with paper recommendations. Based on the search results, provide thoughtful recommendations.

CONTEXT FROM RESEARCH PAPERS:
{context}

RECOMMENDATION REQUEST:
{question}

Please provide recommendations that include:
1. Top recommended papers with clear justification
2. Reading order if multiple papers are suggested
3. What to focus on in each paper
4. How these papers relate to the user's interests
5. Additional search suggestions for deeper exploration

RECOMMENDATIONS:
"""
    
    def _update_chat_history(
        self, 
        query: str, 
        response: str, 
        search_results: List[SearchResult]
    ):
        """Update chat history with current interaction."""
        history_entry = {
            'timestamp': date.today().isoformat(),
            'query': query,
            'response': response[:500] + "..." if len(response) > 500 else response,
            'results_count': len(search_results),
            'top_relevance': search_results[0].final_score if search_results else 0
        }
        
        self.chat_history.append(history_entry)
        
        # Keep only last 10 interactions
        if len(self.chat_history) > 10:
            self.chat_history = self.chat_history[-10:]
    
    def get_search_suggestions(self, partial_query: str) -> List[str]:
        """Get search suggestions for the user."""
        return self.search_engine.get_search_suggestions(partial_query)
    
    def update_user_preferences(self, preferences: Dict[str, Any]):
        """Update user preferences for search behavior."""
        self.user_preferences.update(preferences)
        log_info("User preferences updated", preferences=preferences)
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get search engine statistics."""
        return self.search_engine.get_search_statistics()
    
    def get_chat_statistics(self) -> Dict[str, Any]:
        """Get chat session statistics."""
        if not self.chat_history:
            return {'total_queries': 0, 'avg_relevance': 0, 'session_duration': 0}
        
        total_queries = len(self.chat_history)
        avg_relevance = sum(entry['top_relevance'] for entry in self.chat_history) / total_queries
        
        return {
            'total_queries': total_queries,
            'avg_relevance': round(avg_relevance, 3),
            'search_engine_stats': self.search_engine.get_search_statistics(),
            'recent_queries': [entry['query'] for entry in self.chat_history[-3:]]
        }


def main():
    """Main function for command-line interface."""
    dotenv.load_dotenv()
    
    # Check API key
    if "DEEPSEEK_API_KEY" not in os.environ:
        print("Error: DEEPSEEK_API_KEY environment variable not set.")
        print("Please get your API key from DeepSeek (https://platform.deepseek.com/api_keys)")
        print("Then, add it to the .env file: DEEPSEEK_API_KEY=YOUR_API_KEY_HERE")
        return
    
    parser = argparse.ArgumentParser(
        description="Chat with your papers using advanced search capabilities."
    )
    parser.add_argument("query", type=str, nargs='?',
                       help="The question you want to ask (optional for interactive mode)")
    parser.add_argument("--interactive", action="store_true",
                       help="Start interactive chat mode")
    parser.add_argument("--search-mode", choices=['semantic', 'keyword', 'hybrid'],
                       default='hybrid', help="Search mode to use")
    parser.add_argument("--max-results", type=int, default=5,
                       help="Maximum number of results to consider")
    parser.add_argument("--author", type=str,
                       help="Filter by specific author")
    parser.add_argument("--category", type=str,
                       help="Filter by arXiv category")
    parser.add_argument("--recent", action="store_true",
                       help="Focus on recent papers")
    parser.add_argument("--stats", action="store_true",
                       help="Show search engine statistics")
    
    args = parser.parse_args()
    
    try:
        print("Initializing Advanced Chat RAG System...")
        chat_system = AdvancedChatRAG()
        
        if args.stats:
            stats = chat_system.get_search_statistics()
            print(f"\nSearch Engine Statistics:")
            print(f"Vector Database Size: {stats.get('vector_db_size', 0)} documents")
            print(f"Knowledge Graph Papers: {stats.get('kg_papers', 0)}")
            print(f"Knowledge Graph Authors: {stats.get('kg_authors', 0)}")
            return
        
        # Prepare search options
        search_options = {
            'max_results': args.max_results
        }
        
        if args.author:
            search_options['authors'] = [args.author]
        if args.category:
            search_options['arxiv_categories'] = [args.category.lower()]
        if args.recent:
            search_options['date_preset'] = DateRangePreset.LAST_6_MONTHS
        
        # Update user preferences
        chat_system.update_user_preferences({
            'search_focus': 'recent' if args.recent else 'balanced',
            'max_results': args.max_results
        })
        
        if args.interactive:
            print("\nStarting Interactive Chat Mode")
            print("Type 'quit', 'exit', or press Ctrl+C to stop")
            print("Type 'stats' to see session statistics")
            print("Type 'help' for available commands\n")
            
            while True:
                try:
                    user_input = input("Your question: ").strip()
                    
                    if user_input.lower() in ['quit', 'exit']:
                        break
                    elif user_input.lower() == 'stats':
                        stats = chat_system.get_chat_statistics()
                        print(f"\nSession Statistics:")
                        for key, value in stats.items():
                            print(f"  {key}: {value}")
                        print()
                        continue
                    elif user_input.lower() == 'help':
                        print("\nAvailable Commands:")
                        print("  help - Show this help message")
                        print("  stats - Show session statistics")
                        print("  quit/exit - Exit the chat")
                        print("  Any other text - Ask a question about your papers\n")
                        continue
                    elif not user_input:
                        continue
                    
                    print("\nSearching...")
                    result = chat_system.chat(user_input, search_options)
                    
                    print(f"\nAnswer:")
                    print(result['response'])
                    print()
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"\nError: {e}\n")
            
            print("\nChat session ended. Thank you!")
            
        elif args.query:
            print(f"\nSearching for: {args.query}")
            result = chat_system.chat(args.query, search_options)
            
            print(f"\nAnswer:")
            print(result['response'])
            
            if result['search_results']:
                print(f"\nFound {result['total_results_found']} relevant papers")
                print(f"Search strategy: {result['search_strategy']}")
        
        else:
            print("Please provide a query or use --interactive mode")
            
    except FileNotFoundError as e:
        print(f"Database not found: {e}")
        print("Please create the database first using rag_bbdd_vector_optimized.py")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()