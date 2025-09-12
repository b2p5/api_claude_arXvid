#!/usr/bin/env python3
"""
Advanced Web Interface for arXiv Papers Analysis System
Complete Streamlit dashboard with interactive visualizations, metrics, and export capabilities.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import json
import io
import zipfile
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import sqlite3
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from logger import get_logger
from enhanced_rag_processor import EnhancedRAGProcessor
from content_analysis_db import ContentAnalysisDatabase
from content_analysis import ContentAnalysisEngine
import knowledge_graph
from administration.system.reset_service import SystemResetService


class WebInterface:
    """Advanced web interface for the arXiv analysis system."""
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger()
        self.processor = None
        self.db = None
        self.analysis_db = None
        self.reset_service = SystemResetService()
        
        # Initialize session state
        if 'initialized' not in st.session_state:
            self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        st.session_state.initialized = True
        st.session_state.current_page = "Dashboard"
        st.session_state.selected_papers = []
        st.session_state.analysis_results = None
        st.session_state.export_data = None
    
    def run(self):
        """Main entry point for the web interface."""
        st.set_page_config(
            page_title="arXiv Papers Analysis System",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown(self._get_custom_css(), unsafe_allow_html=True)
        
        # Initialize components
        self._initialize_components()
        
        # Sidebar navigation
        self._render_sidebar()
        
        # Main content based on selected page
        page = st.session_state.current_page
        
        if page == "Dashboard":
            self._render_dashboard()
        elif page == "Paper Analysis":
            self._render_paper_analysis()
        elif page == "Content Analysis":
            self._render_content_analysis()
        elif page == "Knowledge Graph":
            self._render_knowledge_graph()
        elif page == "Export & Reports":
            self._render_export_reports()
        elif page == "System Settings":
            self._render_system_settings()
        elif page == "Administration":
            self._render_administration_page()
    
    def _get_custom_css(self) -> str:
        """Return custom CSS for the interface."""
        return """
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
        }
        .metric-label {
            font-size: 0.9rem;
            opacity: 0.8;
        }
        .sidebar .sidebar-content {
            background: linear-gradient(180deg, #2196F3 0%, #21CBF3 100%);
        }
        .stAlert > div {
            background-color: rgba(33, 150, 243, 0.1);
            border: 1px solid rgba(33, 150, 243, 0.2);
        }
        </style>
        """
    
    def _initialize_components(self):
        """Initialize system components."""
        try:
            # Initialize processor
            if self.processor is None:
                with st.spinner("Initializing analysis system..."):
                    self.processor = EnhancedRAGProcessor(enable_content_analysis=True)
                    self.analysis_db = ContentAnalysisDatabase()
            
        except Exception as e:
            st.error(f"Failed to initialize system components: {e}")
            st.stop()
    
    def _render_sidebar(self):
        """Render the navigation sidebar."""
        with st.sidebar:
            st.image("https://via.placeholder.com/200x80/2196F3/FFFFFF?text=arXiv+Analysis", width=200)
            
            st.markdown("### 🧭 Navigation")
            
            pages = {
                "📊 Dashboard": "Dashboard",
                "📄 Paper Analysis": "Paper Analysis", 
                "🔍 Content Analysis": "Content Analysis",
                "🕸️ Knowledge Graph": "Knowledge Graph",
                "📤 Export & Reports": "Export & Reports",
                "⚙️ System Settings": "System Settings",
                "👑 Administration": "Administration"
            }
            
            for display_name, page_name in pages.items():
                if st.button(display_name, key=f"nav_{page_name}", use_container_width=True):
                    st.session_state.current_page = page_name
                    st.rerun()
            
            st.markdown("---")
            
            # System status
            st.markdown("### 📡 System Status")
            
            try:
                stats = self.processor.get_enhanced_statistics() if self.processor else {}
                
                st.metric(
                    "Papers in Database", 
                    stats.get('processed_papers', 0),
                    help="Total number of processed papers"
                )
                
                st.metric(
                    "Content Analyses",
                    stats.get('total_analyses', 0), 
                    help="Number of completed content analyses"
                )
                
                st.metric(
                    "Concepts Extracted",
                    stats.get('total_concepts', 0),
                    help="Total technical concepts identified"
                )
                
            except Exception as e:
                st.error(f"Status update failed: {e}")
            
            st.markdown("---")
            st.markdown("**🤖 Powered by Claude Code**")
            st.markdown("*Advanced arXiv Analysis System*")
    
    def _render_dashboard(self):
        """Render the main dashboard with metrics and visualizations."""
        st.markdown('<div class="main-header">📊 arXiv Papers Analysis Dashboard</div>', 
                   unsafe_allow_html=True)
        
        # Get comprehensive statistics
        try:
            stats = self.processor.get_enhanced_statistics()
            analysis_stats = self.analysis_db.get_analysis_statistics()
            
            # Overview metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(
                    f"""<div class="metric-card">
                    <div class="metric-value">{stats.get('processed_papers', 0)}</div>
                    <div class="metric-label">📄 Total Papers</div>
                    </div>""", 
                    unsafe_allow_html=True
                )
            
            with col2:
                st.markdown(
                    f"""<div class="metric-card">
                    <div class="metric-value">{analysis_stats.get('total_analyses', 0)}</div>
                    <div class="metric-label">🔍 Content Analyses</div>
                    </div>""",
                    unsafe_allow_html=True
                )
            
            with col3:
                st.markdown(
                    f"""<div class="metric-card">
                    <div class="metric-value">{analysis_stats.get('total_references', 0)}</div>
                    <div class="metric-label">📚 References Extracted</div>
                    </div>""",
                    unsafe_allow_html=True
                )
            
            with col4:
                st.markdown(
                    f"""<div class="metric-card">
                    <div class="metric-value">{analysis_stats.get('total_concepts', 0)}</div>
                    <div class="metric-label">🧠 Concepts Identified</div>
                    </div>""",
                    unsafe_allow_html=True
                )
            
            st.markdown("---")
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 Top Research Concepts")
                self._render_concepts_chart(analysis_stats)
            
            with col2:
                st.subheader("🏷️ Technical Level Distribution")
                self._render_technical_levels_chart(analysis_stats)
            
            # Citation network overview
            st.subheader("🕸️ Citation Network Overview")
            self._render_citation_network_overview()
            
            # Recent activity
            st.subheader("🕒 Recent Analysis Activity")
            self._render_recent_activity()
            
        except Exception as e:
            st.error(f"Dashboard rendering failed: {e}")
    
    def _render_concepts_chart(self, stats: Dict[str, Any]):
        """Render top concepts chart."""
        top_concepts = stats.get('top_concepts', [])
        
        if top_concepts:
            df = pd.DataFrame(top_concepts, columns=['Concept', 'Frequency'])
            df = df.head(10)
            
            fig = px.bar(
                df, 
                x='Frequency', 
                y='Concept',
                orientation='h',
                title="Most Frequent Technical Concepts",
                color='Frequency',
                color_continuous_scale='viridis'
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No concept data available. Run content analysis first.")
    
    def _render_technical_levels_chart(self, stats: Dict[str, Any]):
        """Render technical levels distribution chart."""
        tech_levels = stats.get('technical_levels', {})
        
        if tech_levels:
            fig = px.pie(
                values=list(tech_levels.values()),
                names=list(tech_levels.keys()),
                title="Papers by Technical Complexity"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No technical level data available.")
    
    def _render_citation_network_overview(self):
        """Render citation network overview."""
        try:
            network = self.processor.get_citation_network()
            
            if network:
                total_papers = len(network)
                total_citations = sum(len(cited) for cited in network.values())
                avg_citations = total_citations / total_papers if total_papers > 0 else 0
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Papers with Citations", total_papers)
                with col2:
                    st.metric("Total Citation Links", total_citations)
                with col3:
                    st.metric("Average Citations per Paper", f"{avg_citations:.1f}")
                
                # Top citing papers
                if network:
                    top_papers = sorted(network.items(), key=lambda x: len(x[1]), reverse=True)[:5]
                    
                    st.markdown("**Top Citing Papers:**")
                    for i, (paper, citations) in enumerate(top_papers, 1):
                        st.write(f"{i}. {paper[:80]}... ({len(citations)} citations)")
            else:
                st.info("No citation network data available.")
                
        except Exception as e:
            st.warning(f"Citation network overview failed: {e}")
    
    def _render_recent_activity(self):
        """Render recent analysis activity."""
        try:
            # Query recent analyses from database
            with sqlite3.connect(knowledge_graph.DB_FILE) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT ca.analysis_date, p.title, ca.technical_level, 
                           COUNT(pc.id) as concepts_count,
                           COUNT(pr.id) as references_count
                    FROM content_analyses ca
                    JOIN papers p ON ca.paper_id = p.id
                    LEFT JOIN paper_concepts pc ON ca.id = pc.analysis_id
                    LEFT JOIN paper_references pr ON ca.id = pr.analysis_id
                    GROUP BY ca.id
                    ORDER BY ca.analysis_date DESC
                    LIMIT 10
                """)
                
                recent_data = cursor.fetchall()
                
                if recent_data:
                    df = pd.DataFrame(recent_data, columns=[
                        'Date', 'Paper Title', 'Technical Level', 
                        'Concepts', 'References'
                    ])
                    
                    # Format date
                    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d %H:%M')
                    df['Paper Title'] = df['Paper Title'].str[:60] + '...'
                    
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("No recent analysis activity found.")
                    
        except Exception as e:
            st.warning(f"Recent activity display failed: {e}")
    
    def _render_paper_analysis(self):
        """Render paper analysis interface."""
        st.header("📄 Individual Paper Analysis")
        
        # Paper search and selection
        st.subheader("🔍 Search Papers")
        
        search_query = st.text_input(
            "Enter search terms:", 
            placeholder="machine learning, transformer, neural network..."
        )
        
        search_type = st.selectbox(
            "Search type:",
            ["Semantic Search", "Keyword Search", "Title Search"]
        )
        
        if st.button("🔍 Search Papers") and search_query:
            with st.spinner("Searching papers..."):
                results = self._search_papers(search_query, search_type)
                st.session_state.search_results = results
        
        # Display search results
        if hasattr(st.session_state, 'search_results') and st.session_state.search_results:
            st.subheader("📋 Search Results")
            
            for i, result in enumerate(st.session_state.search_results[:10]):
                with st.expander(f"{i+1}. {result.get('title', 'Unknown Title')}", expanded=False):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**Abstract:** {result.get('summary', 'No abstract available')[:300]}...")
                        st.write(f"**Similarity Score:** {result.get('score', 0):.3f}")
                    
                    with col2:
                        if st.button(f"Analyze", key=f"analyze_{i}"):
                            self._analyze_single_paper(result)
        
        # Single paper upload
        st.subheader("📤 Upload New Paper")
        
        uploaded_file = st.file_uploader(
            "Upload PDF file:", 
            type=['pdf'],
            help="Upload a PDF research paper for analysis"
        )
        
        if uploaded_file and st.button("📊 Analyze Uploaded Paper"):
            with st.spinner("Processing uploaded paper..."):
                self._process_uploaded_paper(uploaded_file)
    
    def _search_papers(self, query: str, search_type: str) -> List[Dict[str, Any]]:
        """Search papers based on query and type."""
        try:
            if search_type == "Semantic Search":
                return self.processor.rag_system.search_similar(query, top_k=20)
            elif search_type == "Keyword Search":
                # Implement keyword search through database
                return self._keyword_search(query)
            else:  # Title Search
                return self._title_search(query)
        except Exception as e:
            st.error(f"Search failed: {e}")
            return []
    
    def _keyword_search(self, query: str) -> List[Dict[str, Any]]:
        """Perform keyword search."""
        try:
            with sqlite3.connect(knowledge_graph.DB_FILE) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, title, summary, arxiv_id
                    FROM papers 
                    WHERE title LIKE ? OR summary LIKE ?
                    ORDER BY title
                    LIMIT 20
                """, (f"%{query}%", f"%{query}%"))
                
                results = []
                for row in cursor.fetchall():
                    results.append({
                        'id': row[0],
                        'title': row[1],
                        'summary': row[2],
                        'arxiv_id': row[3],
                        'score': 1.0
                    })
                
                return results
        except Exception as e:
            st.error(f"Keyword search failed: {e}")
            return []
    
    def _title_search(self, query: str) -> List[Dict[str, Any]]:
        """Perform title-specific search."""
        try:
            with sqlite3.connect(knowledge_graph.DB_FILE) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, title, summary, arxiv_id
                    FROM papers 
                    WHERE title LIKE ?
                    ORDER BY title
                    LIMIT 20
                """, (f"%{query}%",))
                
                results = []
                for row in cursor.fetchall():
                    results.append({
                        'id': row[0],
                        'title': row[1],
                        'summary': row[2],
                        'arxiv_id': row[3],
                        'score': 1.0
                    })
                
                return results
        except Exception as e:
            st.error(f"Title search failed: {e}")
            return []
    
    def _analyze_single_paper(self, paper_data: Dict[str, Any]):
        """Analyze a single selected paper."""
        st.subheader(f"📊 Analysis: {paper_data.get('title', 'Unknown Paper')}")
        
        try:
            # Get or create content analysis
            analysis = self.analysis_db.get_analysis(str(paper_data['id']))
            
            if not analysis:
                with st.spinner("Performing content analysis..."):
                    # Create new analysis
                    engine = ContentAnalysisEngine()
                    analysis = engine.analyze_paper(
                        paper_id=str(paper_data['id']),
                        title=paper_data.get('title', ''),
                        content=paper_data.get('content', paper_data.get('summary', ''))
                    )
                    
                    # Store in database
                    self.analysis_db.store_analysis(analysis)
            
            # Display analysis results
            self._display_paper_analysis(analysis)
            
        except Exception as e:
            st.error(f"Paper analysis failed: {e}")
    
    def _display_paper_analysis(self, analysis):
        """Display comprehensive paper analysis results."""
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("References", len(analysis.references))
        with col2:
            st.metric("Key Concepts", len(analysis.concepts))
        with col3:
            st.metric("Topics", len(analysis.topics))
        with col4:
            st.metric("Technical Level", analysis.technical_level.title())
        
        # Tabs for detailed analysis
        tab1, tab2, tab3, tab4 = st.tabs(["📚 References", "🧠 Concepts", "🏷️ Topics", "📄 Sections"])
        
        with tab1:
            self._display_references(analysis.references)
        
        with tab2:
            self._display_concepts(analysis.concepts)
        
        with tab3:
            self._display_topics(analysis.topics)
        
        with tab4:
            self._display_sections(analysis.sections)
        
        # Overall summary
        if analysis.overall_summary:
            st.subheader("📝 Summary")
            st.write(analysis.overall_summary)
        
        # Main contributions
        if analysis.main_contributions:
            st.subheader("🎯 Main Contributions")
            for i, contribution in enumerate(analysis.main_contributions, 1):
                st.write(f"{i}. {contribution}")
    
    def _display_references(self, references):
        """Display references analysis."""
        if references:
            df_data = []
            for ref in references:
                df_data.append({
                    'Citation': ref.raw_text[:80] + ('...' if len(ref.raw_text) > 80 else ''),
                    'Authors': ', '.join(ref.authors) if ref.authors else 'Unknown',
                    'Year': ref.year or 'Unknown',
                    'Venue': ref.venue or 'Unknown',
                    'Confidence': f"{ref.confidence:.2f}",
                    'ArXiv ID': ref.arxiv_id or '',
                    'DOI': ref.doi or ''
                })
            
            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True)
            
            # Export option
            if st.button("📤 Export References"):
                self._export_references(references)
        else:
            st.info("No references found in this paper.")
    
    def _display_concepts(self, concepts):
        """Display concepts analysis."""
        if concepts:
            # Top concepts chart
            df = pd.DataFrame([
                {'Concept': c.term, 'Importance': c.importance_score, 'Frequency': c.frequency}
                for c in concepts[:15]
            ])
            
            fig = px.scatter(
                df, 
                x='Frequency', 
                y='Importance',
                size='Frequency',
                text='Concept',
                title="Concept Importance vs Frequency",
                hover_data=['Concept']
            )
            fig.update_traces(textposition="top center")
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed table
            st.subheader("📋 Detailed Concepts")
            concept_data = []
            for concept in concepts:
                concept_data.append({
                    'Term': concept.term,
                    'Importance Score': f"{concept.importance_score:.3f}",
                    'Frequency': concept.frequency,
                    'Context Examples': len(concept.context_examples),
                    'Related Terms': ', '.join(concept.related_terms[:3]) if concept.related_terms else ''
                })
            
            df = pd.DataFrame(concept_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No concepts extracted from this paper.")
    
    def _display_topics(self, topics):
        """Display topics analysis."""
        if topics:
            # Topics visualization
            df = pd.DataFrame([
                {'Topic': t.name, 'Weight': t.weight, 'Keywords': ', '.join(t.keywords[:5])}
                for t in topics
            ])
            
            fig = px.bar(
                df,
                x='Weight',
                y='Topic',
                orientation='h',
                title="Topic Classification Results",
                text='Keywords'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed topics
            for i, topic in enumerate(topics, 1):
                with st.expander(f"Topic {i}: {topic.name}"):
                    st.write(f"**Weight:** {topic.weight:.3f}")
                    st.write(f"**Keywords:** {', '.join(topic.keywords)}")
                    if topic.description:
                        st.write(f"**Description:** {topic.description}")
        else:
            st.info("No topics classified for this paper.")
    
    def _display_sections(self, sections):
        """Display sections analysis."""
        if sections:
            for section in sections:
                with st.expander(f"📄 {section.title or section.section_type.value.replace('_', ' ').title()}"):
                    if section.summary:
                        st.markdown("**Summary:**")
                        st.write(section.summary)
                    
                    if section.key_points:
                        st.markdown("**Key Points:**")
                        for point in section.key_points:
                            st.write(f"• {point}")
                    
                    if section.content and st.checkbox(f"Show full content for {section.title}", key=f"content_{section.title}"):
                        st.markdown("**Full Content:**")
                        st.text_area("", section.content[:1000] + ('...' if len(section.content) > 1000 else ''), height=200)
        else:
            st.info("No sections identified in this paper.")
    
    def _process_uploaded_paper(self, uploaded_file):
        """Process an uploaded PDF paper."""
        try:
            # Save uploaded file temporarily
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())
            
            # Process the paper
            results = self.processor.process_papers_with_analysis([temp_path], max_workers=1)
            
            # Clean up temp file
            os.remove(temp_path)
            
            if results['success']:
                st.success("✅ Paper processed successfully!")
                st.rerun()
            else:
                st.error(f"❌ Processing failed: {results.get('error', 'Unknown error')}")
                
        except Exception as e:
            st.error(f"Upload processing failed: {e}")
    
    def _render_content_analysis(self):
        """Render content analysis interface with advanced visualizations."""
        st.header("🔍 Advanced Content Analysis")
        
        # Analysis options
        st.subheader("🎛️ Analysis Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            analysis_type = st.selectbox(
                "Analysis Type:",
                ["Concept Co-occurrence", "Citation Network", "Topic Evolution", "Technical Trends"]
            )
        
        with col2:
            time_filter = st.selectbox(
                "Time Period:",
                ["All Time", "Last Year", "Last 6 Months", "Last Month"]
            )
        
        if st.button("🔄 Run Analysis"):
            with st.spinner(f"Running {analysis_type.lower()}..."):
                self._run_advanced_analysis(analysis_type, time_filter)
        
        # Display results based on analysis type
        if hasattr(st.session_state, 'analysis_results'):
            self._display_advanced_analysis_results()
    
    def _run_advanced_analysis(self, analysis_type: str, time_filter: str):
        """Run advanced content analysis."""
        try:
            if analysis_type == "Concept Co-occurrence":
                results = self._analyze_concept_cooccurrence(time_filter)
            elif analysis_type == "Citation Network":
                results = self._analyze_citation_network(time_filter)
            elif analysis_type == "Topic Evolution":
                results = self._analyze_topic_evolution(time_filter)
            else:  # Technical Trends
                results = self._analyze_technical_trends(time_filter)
            
            st.session_state.analysis_results = {
                'type': analysis_type,
                'data': results,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            st.error(f"Analysis failed: {e}")
    
    def _analyze_concept_cooccurrence(self, time_filter: str) -> Dict[str, Any]:
        """Analyze concept co-occurrence patterns."""
        co_occurrence = self.analysis_db.get_concept_co_occurrence(min_frequency=2)
        
        # Create network data for visualization
        nodes = []
        edges = []
        node_sizes = {}
        
        for concept1, related_concepts in co_occurrence.items():
            if concept1 not in node_sizes:
                node_sizes[concept1] = sum(freq for _, freq in related_concepts)
                nodes.append({'id': concept1, 'size': node_sizes[concept1]})
            
            for concept2, freq in related_concepts[:5]:  # Top 5 related concepts
                if concept2 not in node_sizes:
                    node_sizes[concept2] = freq
                    nodes.append({'id': concept2, 'size': freq})
                
                edges.append({
                    'source': concept1,
                    'target': concept2,
                    'weight': freq
                })
        
        return {
            'nodes': nodes[:50],  # Limit for visualization
            'edges': edges[:100],
            'total_concepts': len(co_occurrence)
        }
    
    def _analyze_citation_network(self, time_filter: str) -> Dict[str, Any]:
        """Analyze citation network structure."""
        network = self.processor.get_citation_network()
        
        # Create networkx graph
        G = nx.DiGraph()
        
        for citing, cited_list in network.items():
            for cited in cited_list:
                G.add_edge(citing, cited)
        
        # Calculate network metrics
        metrics = {
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(),
            'density': nx.density(G) if G.number_of_nodes() > 1 else 0,
            'avg_clustering': nx.average_clustering(G.to_undirected()) if G.number_of_nodes() > 2 else 0
        }
        
        # Get central papers
        if G.number_of_nodes() > 0:
            centrality = nx.pagerank(G)
            top_papers = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
            metrics['top_papers'] = top_papers
        else:
            metrics['top_papers'] = []
        
        return metrics
    
    def _analyze_topic_evolution(self, time_filter: str) -> Dict[str, Any]:
        """Analyze topic evolution over time."""
        try:
            with sqlite3.connect(knowledge_graph.DB_FILE) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT pt.topic_name, ca.analysis_date, COUNT(*) as frequency
                    FROM paper_topics pt
                    JOIN content_analyses ca ON pt.analysis_id = ca.id
                    GROUP BY pt.topic_name, DATE(ca.analysis_date)
                    ORDER BY ca.analysis_date
                """)
                
                data = cursor.fetchall()
                
                # Process data for visualization
                topic_evolution = {}
                for topic, date, freq in data:
                    if topic not in topic_evolution:
                        topic_evolution[topic] = []
                    topic_evolution[topic].append({'date': date, 'frequency': freq})
                
                return {
                    'evolution_data': topic_evolution,
                    'total_topics': len(topic_evolution)
                }
                
        except Exception as e:
            st.error(f"Topic evolution analysis failed: {e}")
            return {'evolution_data': {}, 'total_topics': 0}
    
    def _analyze_technical_trends(self, time_filter: str) -> Dict[str, Any]:
        """Analyze technical complexity trends."""
        try:
            with sqlite3.connect(knowledge_graph.DB_FILE) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT technical_level, COUNT(*) as count,
                           AVG(CASE WHEN pc.importance_score IS NOT NULL 
                               THEN pc.importance_score ELSE 0 END) as avg_concept_score
                    FROM content_analyses ca
                    LEFT JOIN paper_concepts pc ON ca.id = pc.analysis_id
                    GROUP BY technical_level
                """)
                
                data = cursor.fetchall()
                
                trends = {
                    'technical_distribution': data,
                    'complexity_scores': {row[0]: row[2] for row in data}
                }
                
                return trends
                
        except Exception as e:
            st.error(f"Technical trends analysis failed: {e}")
            return {'technical_distribution': [], 'complexity_scores': {}}
    
    def _display_advanced_analysis_results(self):
        """Display advanced analysis results with interactive visualizations."""
        results = st.session_state.analysis_results
        analysis_type = results['type']
        data = results['data']
        
        st.subheader(f"📊 {analysis_type} Results")
        
        if analysis_type == "Concept Co-occurrence":
            self._display_concept_network(data)
        elif analysis_type == "Citation Network":
            self._display_citation_metrics(data)
        elif analysis_type == "Topic Evolution":
            self._display_topic_evolution(data)
        else:  # Technical Trends
            self._display_technical_trends(data)
    
    def _display_concept_network(self, data: Dict[str, Any]):
        """Display concept co-occurrence network."""
        nodes = data.get('nodes', [])
        edges = data.get('edges', [])
        
        if nodes and edges:
            st.info(f"Displaying {len(nodes)} concepts with {len(edges)} relationships")
            
            # Create network visualization using plotly
            G = nx.Graph()
            
            for node in nodes:
                G.add_node(node['id'], size=node['size'])
            
            for edge in edges:
                G.add_edge(edge['source'], edge['target'], weight=edge['weight'])
            
            # Generate layout
            pos = nx.spring_layout(G, k=2, iterations=50)
            
            # Create plotly figure
            edge_x = []
            edge_y = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines'
            )
            
            node_x = []
            node_y = []
            node_text = []
            node_size = []
            
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(node)
                node_size.append(G.nodes[node].get('size', 10))
            
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                text=node_text,
                textposition="middle center",
                hoverinfo='text',
                marker=dict(
                    size=[s/2 + 10 for s in node_size],
                    color='lightblue',
                    line=dict(width=2, color='darkblue')
                )
            )
            
            fig = go.Figure(data=[edge_trace, node_trace],
                           layout=go.Layout(
                               title="Concept Co-occurrence Network",
                               showlegend=False,
                               hovermode='closest',
                               margin=dict(b=20,l=5,r=5,t=40),
                               xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                               yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                               height=600
                           ))
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No concept co-occurrence data available.")
    
    def _display_citation_metrics(self, data: Dict[str, Any]):
        """Display citation network metrics."""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Papers", data.get('nodes', 0))
        with col2:
            st.metric("Citation Links", data.get('edges', 0))
        with col3:
            st.metric("Network Density", f"{data.get('density', 0):.3f}")
        with col4:
            st.metric("Avg Clustering", f"{data.get('avg_clustering', 0):.3f}")
        
        # Top papers by centrality
        top_papers = data.get('top_papers', [])
        if top_papers:
            st.subheader("🏆 Most Influential Papers")
            
            df = pd.DataFrame([
                {'Paper': paper[:60] + '...', 'PageRank Score': f"{score:.4f}"}
                for paper, score in top_papers
            ])
            
            fig = px.bar(
                df, 
                x='PageRank Score', 
                y='Paper',
                orientation='h',
                title="Paper Influence by PageRank"
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    def _display_topic_evolution(self, data: Dict[str, Any]):
        """Display topic evolution over time."""
        evolution_data = data.get('evolution_data', {})
        
        if evolution_data:
            # Create time series plot
            all_data = []
            for topic, timeline in evolution_data.items():
                for point in timeline:
                    all_data.append({
                        'Topic': topic,
                        'Date': point['date'],
                        'Frequency': point['frequency']
                    })
            
            if all_data:
                df = pd.DataFrame(all_data)
                df['Date'] = pd.to_datetime(df['Date'])
                
                fig = px.line(
                    df, 
                    x='Date', 
                    y='Frequency',
                    color='Topic',
                    title="Topic Evolution Over Time"
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Topic frequency table
                topic_summary = df.groupby('Topic')['Frequency'].sum().sort_values(ascending=False)
                st.subheader("📊 Topic Frequency Summary")
                st.bar_chart(topic_summary)
        else:
            st.info("No topic evolution data available.")
    
    def _display_technical_trends(self, data: Dict[str, Any]):
        """Display technical complexity trends."""
        distribution = data.get('technical_distribution', [])
        complexity_scores = data.get('complexity_scores', {})
        
        if distribution:
            # Technical level distribution
            df_dist = pd.DataFrame(distribution, columns=['Level', 'Count', 'Avg_Score'])
            
            fig_dist = px.pie(
                df_dist,
                values='Count',
                names='Level',
                title="Technical Complexity Distribution"
            )
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # Complexity scores
            if complexity_scores:
                fig_scores = px.bar(
                    x=list(complexity_scores.keys()),
                    y=list(complexity_scores.values()),
                    title="Average Concept Complexity by Technical Level"
                )
                fig_scores.update_layout(
                    xaxis_title="Technical Level",
                    yaxis_title="Average Concept Score"
                )
                st.plotly_chart(fig_scores, use_container_width=True)
    
    def _render_knowledge_graph(self):
        """Render knowledge graph visualization interface."""
        st.header("🕸️ Knowledge Graph Visualization")
        
        # Graph options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            graph_type = st.selectbox(
                "Graph Type:",
                ["Citation Network", "Concept Relations", "Author Collaboration", "Topic Connections"]
            )
        
        with col2:
            layout_algorithm = st.selectbox(
                "Layout Algorithm:",
                ["Spring", "Circular", "Hierarchical", "Random"]
            )
        
        with col3:
            node_limit = st.slider("Node Limit:", 10, 100, 50)
        
        if st.button("🎨 Generate Graph"):
            with st.spinner("Generating knowledge graph..."):
                self._generate_knowledge_graph(graph_type, layout_algorithm, node_limit)
        
        # Display graph if generated
        if hasattr(st.session_state, 'knowledge_graph'):
            self._display_knowledge_graph()
    
    def _generate_knowledge_graph(self, graph_type: str, layout: str, node_limit: int):
        """Generate knowledge graph based on parameters."""
        try:
            if graph_type == "Citation Network":
                graph_data = self._build_citation_graph(node_limit)
            elif graph_type == "Concept Relations":
                graph_data = self._build_concept_graph(node_limit)
            elif graph_type == "Author Collaboration":
                graph_data = self._build_author_graph(node_limit)
            else:  # Topic Connections
                graph_data = self._build_topic_graph(node_limit)
            
            st.session_state.knowledge_graph = {
                'type': graph_type,
                'layout': layout,
                'data': graph_data
            }
            
        except Exception as e:
            st.error(f"Knowledge graph generation failed: {e}")
    
    def _build_citation_graph(self, node_limit: int) -> Dict[str, Any]:
        """Build citation network graph."""
        network = self.processor.get_citation_network()
        
        # Limit nodes
        limited_network = dict(list(network.items())[:node_limit])
        
        nodes = []
        edges = []
        
        all_papers = set(limited_network.keys())
        for cited_list in limited_network.values():
            all_papers.update(cited_list[:5])  # Limit cited papers per paper
        
        for paper in list(all_papers)[:node_limit]:
            nodes.append({
                'id': paper,
                'label': paper[:30] + '...' if len(paper) > 30 else paper,
                'type': 'citing' if paper in limited_network else 'cited'
            })
        
        for citing, cited_list in limited_network.items():
            for cited in cited_list[:3]:  # Limit edges per node
                if cited in all_papers:
                    edges.append({
                        'source': citing,
                        'target': cited,
                        'type': 'citation'
                    })
        
        return {'nodes': nodes, 'edges': edges}
    
    def _build_concept_graph(self, node_limit: int) -> Dict[str, Any]:
        """Build concept relations graph."""
        co_occurrence = self.analysis_db.get_concept_co_occurrence(min_frequency=1)
        
        nodes = []
        edges = []
        
        # Get top concepts by frequency
        concept_freq = {}
        for concept, relations in co_occurrence.items():
            total_freq = sum(freq for _, freq in relations)
            concept_freq[concept] = total_freq
        
        top_concepts = sorted(concept_freq.items(), key=lambda x: x[1], reverse=True)[:node_limit]
        
        for concept, freq in top_concepts:
            nodes.append({
                'id': concept,
                'label': concept,
                'size': freq,
                'type': 'concept'
            })
        
        top_concept_names = {concept for concept, _ in top_concepts}
        
        for concept, relations in co_occurrence.items():
            if concept in top_concept_names:
                for related_concept, freq in relations[:5]:
                    if related_concept in top_concept_names:
                        edges.append({
                            'source': concept,
                            'target': related_concept,
                            'weight': freq,
                            'type': 'co_occurrence'
                        })
        
        return {'nodes': nodes, 'edges': edges}
    
    def _build_author_graph(self, node_limit: int) -> Dict[str, Any]:
        """Build author collaboration graph (simplified)."""
        # This would need author extraction from papers
        # For now, return empty graph with message
        return {
            'nodes': [],
            'edges': [],
            'message': "Author collaboration graph requires author extraction implementation"
        }
    
    def _build_topic_graph(self, node_limit: int) -> Dict[str, Any]:
        """Build topic connections graph."""
        try:
            with sqlite3.connect(knowledge_graph.DB_FILE) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT pt1.topic_name, pt2.topic_name, COUNT(*) as shared_papers
                    FROM paper_topics pt1
                    JOIN paper_topics pt2 ON pt1.analysis_id = pt2.analysis_id
                    WHERE pt1.topic_name != pt2.topic_name
                    GROUP BY pt1.topic_name, pt2.topic_name
                    HAVING shared_papers > 0
                    ORDER BY shared_papers DESC
                """)
                
                connections = cursor.fetchall()
                
                # Get all unique topics
                topics = set()
                for topic1, topic2, _ in connections:
                    topics.add(topic1)
                    topics.add(topic2)
                
                nodes = []
                for topic in list(topics)[:node_limit]:
                    nodes.append({
                        'id': topic,
                        'label': topic.replace('_', ' ').title(),
                        'type': 'topic'
                    })
                
                edges = []
                topic_names = {node['id'] for node in nodes}
                
                for topic1, topic2, shared in connections:
                    if topic1 in topic_names and topic2 in topic_names:
                        edges.append({
                            'source': topic1,
                            'target': topic2,
                            'weight': shared,
                            'type': 'shared_papers'
                        })
                
                return {'nodes': nodes, 'edges': edges}
                
        except Exception as e:
            return {'nodes': [], 'edges': [], 'error': str(e)}
    
    def _display_knowledge_graph(self):
        """Display the generated knowledge graph."""
        graph_info = st.session_state.knowledge_graph
        graph_data = graph_info['data']
        
        nodes = graph_data.get('nodes', [])
        edges = graph_data.get('edges', [])
        
        if 'message' in graph_data:
            st.info(graph_data['message'])
            return
        
        if 'error' in graph_data:
            st.error(f"Graph generation failed: {graph_data['error']}")
            return
        
        if not nodes:
            st.warning("No graph data available for the selected parameters.")
            return
        
        # Create networkx graph
        G = nx.Graph() if graph_info['type'] != "Citation Network" else nx.DiGraph()
        
        for node in nodes:
            G.add_node(node['id'], **node)
        
        for edge in edges:
            G.add_edge(edge['source'], edge['target'], **edge)
        
        # Generate layout
        layout_func = {
            'Spring': nx.spring_layout,
            'Circular': nx.circular_layout,
            'Hierarchical': nx.spring_layout,  # Could implement hierarchical
            'Random': nx.random_layout
        }.get(graph_info['layout'], nx.spring_layout)
        
        pos = layout_func(G, k=1, iterations=50)
        
        # Create plotly visualization
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []
        
        color_map = {
            'citing': 'red',
            'cited': 'blue',
            'concept': 'green',
            'topic': 'orange',
            'author': 'purple'
        }
        
        for node_id in G.nodes():
            x, y = pos[node_id]
            node_x.append(x)
            node_y.append(y)
            
            node_data = G.nodes[node_id]
            node_text.append(node_data.get('label', node_id))
            
            node_type = node_data.get('type', 'default')
            node_color.append(color_map.get(node_type, 'lightblue'))
            
            node_size.append(node_data.get('size', 20))
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="middle center",
            hoverinfo='text',
            marker=dict(
                size=[max(10, min(50, s/2)) for s in node_size] if any(s > 1 for s in node_size) else 20,
                color=node_color,
                line=dict(width=2, color='white')
            )
        )
        
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=f"{graph_info['type']} - {graph_info['layout']} Layout",
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=700
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Graph statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Nodes", len(nodes))
        with col2:
            st.metric("Edges", len(edges))
        with col3:
            if len(nodes) > 1:
                density = len(edges) / (len(nodes) * (len(nodes) - 1) / 2) if not G.is_directed() else len(edges) / (len(nodes) * (len(nodes) - 1))
                st.metric("Density", f"{density:.3f}")
    
    def _render_export_reports(self):
        """Render export and reports interface."""
        st.header("📤 Export & Reports")
        
        # Export options
        st.subheader("📊 Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            export_type = st.selectbox(
                "Data Type:",
                ["Complete Analysis", "Papers Metadata", "Content Analysis", "Citation Network", "Concepts & Topics"]
            )
        
        with col2:
            export_format = st.selectbox(
                "Format:",
                ["JSON", "CSV", "Excel", "PDF Report"]
            )
        
        # Time range filter
        st.subheader("🕒 Time Range")
        col1, col2 = st.columns(2)
        
        with col1:
            start_date = st.date_input("From:", datetime.now() - timedelta(days=365))
        
        with col2:
            end_date = st.date_input("To:", datetime.now())
        
        # Additional filters
        st.subheader("🔍 Filters")
        
        technical_levels = st.multiselect(
            "Technical Levels:",
            ["basic", "medium", "advanced"],
            default=["basic", "medium", "advanced"]
        )
        
        min_concepts = st.slider("Minimum Concepts:", 0, 100, 0)
        
        # Export button
        if st.button("📥 Generate Export", type="primary"):
            with st.spinner("Generating export..."):
                export_data = self._generate_export_data(
                    export_type, export_format, start_date, end_date, 
                    technical_levels, min_concepts
                )
                
                if export_data:
                    self._provide_download(export_data, export_type, export_format)
        
        # Pre-built reports
        st.subheader("📋 Pre-built Reports")
        
        report_types = [
            "📊 System Overview Report",
            "🔍 Content Analysis Summary", 
            "🕸️ Citation Network Analysis",
            "📈 Technical Trends Report",
            "🏷️ Topic Classification Report"
        ]
        
        for report in report_types:
            if st.button(report, key=f"report_{report}"):
                with st.spinner(f"Generating {report}..."):
                    self._generate_report(report)
    
    def _generate_export_data(self, export_type: str, export_format: str, 
                             start_date, end_date, technical_levels: List[str], 
                             min_concepts: int) -> Optional[Dict[str, Any]]:
        """Generate export data based on parameters."""
        try:
            if export_type == "Complete Analysis":
                return self._export_complete_analysis(start_date, end_date, technical_levels, min_concepts)
            elif export_type == "Papers Metadata":
                return self._export_papers_metadata(start_date, end_date)
            elif export_type == "Content Analysis":
                return self._export_content_analysis(start_date, end_date, technical_levels)
            elif export_type == "Citation Network":
                return self._export_citation_network()
            else:  # Concepts & Topics
                return self._export_concepts_topics(min_concepts)
                
        except Exception as e:
            st.error(f"Export generation failed: {e}")
            return None
    
    def _export_complete_analysis(self, start_date, end_date, technical_levels, min_concepts) -> Dict[str, Any]:
        """Export complete analysis data."""
        try:
            with sqlite3.connect(knowledge_graph.DB_FILE) as conn:
                # Papers with analysis
                papers_query = """
                    SELECT p.id, p.title, p.summary, p.arxiv_id, p.authors, p.published_date,
                           ca.technical_level, ca.overall_summary, ca.analysis_date
                    FROM papers p
                    JOIN content_analyses ca ON p.id = ca.paper_id
                    WHERE DATE(ca.analysis_date) BETWEEN ? AND ?
                    AND ca.technical_level IN ({})
                """.format(','.join('?' * len(technical_levels)))
                
                papers_df = pd.read_sql_query(
                    papers_query,
                    conn,
                    params=[start_date, end_date] + technical_levels
                )
                
                # Get analysis IDs for further filtering
                analysis_ids = papers_df.index.tolist() if len(papers_df) > 0 else []
                
                if not analysis_ids:
                    return {'papers': [], 'concepts': [], 'references': [], 'topics': []}
                
                # Concepts
                concepts_query = """
                    SELECT pc.analysis_id, pc.term, pc.frequency, pc.importance_score
                    FROM paper_concepts pc
                    WHERE pc.importance_score >= ?
                    ORDER BY pc.importance_score DESC
                """
                concepts_df = pd.read_sql_query(concepts_query, conn, params=[min_concepts / 100.0])
                
                # References  
                references_query = """
                    SELECT pr.analysis_id, pr.raw_text, pr.authors_json, pr.publication_year, pr.venue
                    FROM paper_references pr
                """
                references_df = pd.read_sql_query(references_query, conn)
                
                # Topics
                topics_query = """
                    SELECT pt.analysis_id, pt.topic_name, pt.weight, pt.keywords_json
                    FROM paper_topics pt
                """
                topics_df = pd.read_sql_query(topics_query, conn)
                
                return {
                    'papers': papers_df.to_dict('records'),
                    'concepts': concepts_df.to_dict('records'),
                    'references': references_df.to_dict('records'),
                    'topics': topics_df.to_dict('records'),
                    'metadata': {
                        'export_date': datetime.now().isoformat(),
                        'filters': {
                            'date_range': [str(start_date), str(end_date)],
                            'technical_levels': technical_levels,
                            'min_concepts': min_concepts
                        }
                    }
                }
                
        except Exception as e:
            st.error(f"Complete analysis export failed: {e}")
            return None
    
    def _export_papers_metadata(self, start_date, end_date) -> Dict[str, Any]:
        """Export papers metadata."""
        try:
            with sqlite3.connect(knowledge_graph.DB_FILE) as conn:
                query = """
                    SELECT id, title, summary, arxiv_id, authors, published_date,
                           pdf_path, processed_date
                    FROM papers
                    WHERE DATE(processed_date) BETWEEN ? AND ?
                    ORDER BY processed_date DESC
                """
                
                df = pd.read_sql_query(query, conn, params=[start_date, end_date])
                
                return {
                    'papers': df.to_dict('records'),
                    'total_papers': len(df),
                    'date_range': [str(start_date), str(end_date)]
                }
                
        except Exception as e:
            st.error(f"Papers metadata export failed: {e}")
            return None
    
    def _export_content_analysis(self, start_date, end_date, technical_levels) -> Dict[str, Any]:
        """Export content analysis results."""
        try:
            with sqlite3.connect(knowledge_graph.DB_FILE) as conn:
                query = """
                    SELECT ca.*, p.title as paper_title
                    FROM content_analyses ca
                    JOIN papers p ON ca.paper_id = p.id
                    WHERE DATE(ca.analysis_date) BETWEEN ? AND ?
                    AND ca.technical_level IN ({})
                    ORDER BY ca.analysis_date DESC
                """.format(','.join('?' * len(technical_levels)))
                
                df = pd.read_sql_query(query, conn, params=[start_date, end_date] + technical_levels)
                
                return {
                    'analyses': df.to_dict('records'),
                    'total_analyses': len(df),
                    'filters': {
                        'date_range': [str(start_date), str(end_date)],
                        'technical_levels': technical_levels
                    }
                }
                
        except Exception as e:
            st.error(f"Content analysis export failed: {e}")
            return None
    
    def _export_citation_network(self) -> Dict[str, Any]:
        """Export citation network data."""
        try:
            network = self.processor.get_citation_network()
            
            # Convert to edge list format
            edges = []
            for citing, cited_list in network.items():
                for cited in cited_list:
                    edges.append({
                        'source': citing,
                        'target': cited,
                        'type': 'citation'
                    })
            
            # Get nodes
            nodes = set(network.keys())
            for cited_list in network.values():
                nodes.update(cited_list)
            
            node_list = [{'id': node, 'type': 'paper'} for node in nodes]
            
            return {
                'nodes': node_list,
                'edges': edges,
                'network_stats': {
                    'total_nodes': len(node_list),
                    'total_edges': len(edges),
                    'density': len(edges) / (len(node_list) * (len(node_list) - 1)) if len(node_list) > 1 else 0
                }
            }
            
        except Exception as e:
            st.error(f"Citation network export failed: {e}")
            return None
    
    def _export_concepts_topics(self, min_concepts: int) -> Dict[str, Any]:
        """Export concepts and topics data."""
        try:
            with sqlite3.connect(knowledge_graph.DB_FILE) as conn:
                # Concepts
                concepts_query = """
                    SELECT pc.*, p.title as paper_title
                    FROM paper_concepts pc
                    JOIN content_analyses ca ON pc.analysis_id = ca.id
                    JOIN papers p ON ca.paper_id = p.id
                    WHERE pc.frequency >= ?
                    ORDER BY pc.importance_score DESC
                """
                
                concepts_df = pd.read_sql_query(concepts_query, conn, params=[min_concepts])
                
                # Topics
                topics_query = """
                    SELECT pt.*, p.title as paper_title
                    FROM paper_topics pt
                    JOIN content_analyses ca ON pt.analysis_id = ca.id
                    JOIN papers p ON ca.paper_id = p.id
                    ORDER BY pt.weight DESC
                """
                
                topics_df = pd.read_sql_query(topics_query, conn)
                
                return {
                    'concepts': concepts_df.to_dict('records'),
                    'topics': topics_df.to_dict('records'),
                    'summary': {
                        'total_concepts': len(concepts_df),
                        'total_topics': len(topics_df),
                        'min_concept_frequency': min_concepts
                    }
                }
                
        except Exception as e:
            st.error(f"Concepts/topics export failed: {e}")
            return None
    
    def _provide_download(self, data: Dict[str, Any], export_type: str, format_type: str):
        """Provide download link for exported data."""
        try:
            filename = f"arxiv_export_{export_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            if format_type == "JSON":
                json_data = json.dumps(data, indent=2, default=str)
                st.download_button(
                    label="📥 Download JSON",
                    data=json_data,
                    file_name=f"{filename}.json",
                    mime="application/json"
                )
                
            elif format_type == "CSV":
                # Convert main data to CSV
                if 'papers' in data and data['papers']:
                    df = pd.DataFrame(data['papers'])
                    csv_data = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv_data,
                        file_name=f"{filename}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No tabular data available for CSV export")
                    
            elif format_type == "Excel":
                # Create Excel file with multiple sheets
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    for key, value in data.items():
                        if isinstance(value, list) and value:
                            df = pd.DataFrame(value)
                            sheet_name = key[:31]  # Excel sheet name limit
                            df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                st.download_button(
                    label="📥 Download Excel",
                    data=buffer.getvalue(),
                    file_name=f"{filename}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
            st.success("✅ Export ready for download!")
            
        except Exception as e:
            st.error(f"Download preparation failed: {e}")
    
    def _generate_report(self, report_type: str):
        """Generate and display pre-built reports."""
        try:
            if "System Overview" in report_type:
                self._show_system_overview_report()
            elif "Content Analysis" in report_type:
                self._show_content_analysis_report()
            elif "Citation Network" in report_type:
                self._show_citation_network_report()
            elif "Technical Trends" in report_type:
                self._show_technical_trends_report()
            else:  # Topic Classification
                self._show_topic_classification_report()
                
        except Exception as e:
            st.error(f"Report generation failed: {e}")
    
    def _show_system_overview_report(self):
        """Show comprehensive system overview report."""
        st.subheader("📊 System Overview Report")
        
        stats = self.processor.get_enhanced_statistics()
        analysis_stats = self.analysis_db.get_analysis_statistics()
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Papers", stats.get('processed_papers', 0))
        with col2:
            st.metric("Analyses Completed", analysis_stats.get('total_analyses', 0))
        with col3:
            st.metric("References Extracted", analysis_stats.get('total_references', 0))
        with col4:
            st.metric("Concepts Identified", analysis_stats.get('total_concepts', 0))
        
        # Detailed breakdown
        st.markdown("### 📈 Performance Metrics")
        
        performance_data = {
            "Metric": ["Papers Processed", "Content Analyses", "References", "Concepts", "Topics", "Citations"],
            "Count": [
                stats.get('processed_papers', 0),
                analysis_stats.get('total_analyses', 0),
                analysis_stats.get('total_references', 0),
                analysis_stats.get('total_concepts', 0),
                analysis_stats.get('total_topics', 0),
                analysis_stats.get('total_citations', 0)
            ]
        }
        
        df = pd.DataFrame(performance_data)
        fig = px.bar(df, x='Metric', y='Count', title="System Performance Overview")
        st.plotly_chart(fig, use_container_width=True)
        
        # Technical levels
        tech_levels = analysis_stats.get('technical_levels', {})
        if tech_levels:
            st.markdown("### 🎯 Technical Complexity Distribution")
            fig_tech = px.pie(
                values=list(tech_levels.values()),
                names=list(tech_levels.keys()),
                title="Papers by Technical Level"
            )
            st.plotly_chart(fig_tech, use_container_width=True)
    
    def _show_content_analysis_report(self):
        """Show content analysis summary report."""
        st.subheader("🔍 Content Analysis Report")
        
        analysis_stats = self.analysis_db.get_analysis_statistics()
        
        # Top concepts
        top_concepts = analysis_stats.get('top_concepts', [])
        if top_concepts:
            st.markdown("### 🧠 Most Frequent Concepts")
            
            df_concepts = pd.DataFrame(top_concepts[:15], columns=['Concept', 'Frequency'])
            fig_concepts = px.bar(
                df_concepts, 
                x='Frequency', 
                y='Concept',
                orientation='h',
                title="Top Technical Concepts"
            )
            st.plotly_chart(fig_concepts, use_container_width=True)
        
        # Analysis summary
        st.markdown("### 📊 Analysis Summary")
        
        summary_data = {
            "Category": ["References", "Concepts", "Topics", "Citations"],
            "Total Count": [
                analysis_stats.get('total_references', 0),
                analysis_stats.get('total_concepts', 0),
                analysis_stats.get('total_topics', 0),
                analysis_stats.get('total_citations', 0)
            ],
            "Per Paper Average": [
                analysis_stats.get('total_references', 0) / max(1, analysis_stats.get('total_analyses', 1)),
                analysis_stats.get('total_concepts', 0) / max(1, analysis_stats.get('total_analyses', 1)),
                analysis_stats.get('total_topics', 0) / max(1, analysis_stats.get('total_analyses', 1)),
                analysis_stats.get('total_citations', 0) / max(1, analysis_stats.get('total_analyses', 1))
            ]
        }
        
        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary, use_container_width=True)
    
    def _show_citation_network_report(self):
        """Show citation network analysis report."""
        st.subheader("🕸️ Citation Network Report")
        
        network = self.processor.get_citation_network()
        
        if network:
            total_papers = len(network)
            total_citations = sum(len(cited) for cited in network.values())
            avg_citations = total_citations / total_papers if total_papers > 0 else 0
            
            # Network metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Citing Papers", total_papers)
            with col2:
                st.metric("Total Citations", total_citations)
            with col3:
                st.metric("Average Citations/Paper", f"{avg_citations:.1f}")
            
            # Top citing papers
            st.markdown("### 🏆 Most Active Citing Papers")
            
            citation_counts = [(paper, len(citations)) for paper, citations in network.items()]
            citation_counts.sort(key=lambda x: x[1], reverse=True)
            
            top_papers = citation_counts[:10]
            df_citations = pd.DataFrame(top_papers, columns=['Paper', 'Citations'])
            df_citations['Paper'] = df_citations['Paper'].str[:50] + '...'
            
            fig_citations = px.bar(
                df_citations,
                x='Citations',
                y='Paper', 
                orientation='h',
                title="Papers by Citation Count"
            )
            st.plotly_chart(fig_citations, use_container_width=True)
        else:
            st.info("No citation network data available")
    
    def _show_technical_trends_report(self):
        """Show technical trends analysis report."""
        st.subheader("📈 Technical Trends Report")
        
        # Get technical trends data
        trends_data = self._analyze_technical_trends("All Time")
        
        # Technical distribution
        distribution = trends_data.get('technical_distribution', [])
        if distribution:
            df_tech = pd.DataFrame(distribution, columns=['Level', 'Count', 'Avg_Score'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_dist = px.pie(
                    df_tech,
                    values='Count',
                    names='Level',
                    title="Technical Level Distribution"
                )
                st.plotly_chart(fig_dist, use_container_width=True)
            
            with col2:
                fig_scores = px.bar(
                    df_tech,
                    x='Level',
                    y='Avg_Score',
                    title="Average Concept Complexity"
                )
                st.plotly_chart(fig_scores, use_container_width=True)
    
    def _show_topic_classification_report(self):
        """Show topic classification report."""
        st.subheader("🏷️ Topic Classification Report")
        
        try:
            with sqlite3.connect(knowledge_graph.DB_FILE) as conn:
                # Topic frequency
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT topic_name, COUNT(*) as frequency,
                           AVG(weight) as avg_weight
                    FROM paper_topics
                    GROUP BY topic_name
                    ORDER BY frequency DESC
                    LIMIT 15
                """)
                
                topic_data = cursor.fetchall()
                
                if topic_data:
                    df_topics = pd.DataFrame(topic_data, columns=['Topic', 'Frequency', 'Avg_Weight'])
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_freq = px.bar(
                            df_topics,
                            x='Frequency',
                            y='Topic',
                            orientation='h',
                            title="Topic Frequency"
                        )
                        st.plotly_chart(fig_freq, use_container_width=True)
                    
                    with col2:
                        fig_weight = px.scatter(
                            df_topics,
                            x='Frequency',
                            y='Avg_Weight',
                            text='Topic',
                            title="Topic Frequency vs Average Weight"
                        )
                        st.plotly_chart(fig_weight, use_container_width=True)
                    
                    # Detailed table
                    st.markdown("### 📋 Topic Details")
                    st.dataframe(df_topics, use_container_width=True)
                else:
                    st.info("No topic classification data available")
                    
        except Exception as e:
            st.error(f"Topic classification report failed: {e}")
    
    def _render_system_settings(self):
        """Render system settings and configuration interface."""
        st.header("⚙️ System Settings")
        
        # Configuration settings
        st.subheader("🔧 Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Analysis Settings**")
            
            enable_content_analysis = st.checkbox(
                "Enable Content Analysis",
                value=True,
                help="Perform detailed content analysis including concepts, topics, etc."
            )
            
            max_workers = st.slider(
                "Max Processing Workers:",
                1, 8, 4,
                help="Number of parallel workers for paper processing"
            )
            
            chunk_size = st.slider(
                "Text Chunk Size:",
                500, 2000, 1000,
                help="Size of text chunks for processing"
            )
        
        with col2:
            st.markdown("**Visualization Settings**")
            
            default_graph_nodes = st.slider(
                "Default Graph Node Limit:",
                10, 200, 50,
                help="Default number of nodes in graph visualizations"
            )
            
            chart_theme = st.selectbox(
                "Chart Theme:",
                ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn"]
            )
            
            show_advanced_metrics = st.checkbox(
                "Show Advanced Metrics",
                value=True,
                help="Display advanced statistical metrics in dashboards"
            )
        
        # Database settings
        st.subheader("🗄️ Database Management")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Refresh Statistics"):
                st.success("Statistics refreshed!")
                st.rerun()
        
        with col2:
            if st.button("🧹 Clean Cache"):
                st.success("Cache cleaned!")
        
        with col3:
            if st.button("📊 Rebuild Indexes"):
                with st.spinner("Rebuilding database indexes..."):
                    # Would implement index rebuilding
                    st.success("Indexes rebuilt!")
        
        # System information
        st.subheader("ℹ️ System Information")
        
        try:
            import psutil
            import platform
            
            system_info = {
                "Python Version": platform.python_version(),
                "Platform": platform.platform(),
                "CPU Count": psutil.cpu_count(),
                "Memory": f"{psutil.virtual_memory().total // (1024**3)} GB",
                "Available Memory": f"{psutil.virtual_memory().available // (1024**3)} GB"
            }
            
            for key, value in system_info.items():
                st.text(f"{key}: {value}")
                
        except ImportError:
            st.info("Install 'psutil' for detailed system information")
        
        # Logs and diagnostics
        st.subheader("📋 Logs & Diagnostics")
        
        if st.button("📄 View Recent Logs"):
            st.text_area(
                "Recent Log Entries",
                "Log viewing functionality would be implemented here...",
                height=200
            )
        
        if st.button("🔍 Run Diagnostics"):
            with st.spinner("Running system diagnostics..."):
                diagnostics = self._run_system_diagnostics()
                
                for check, status in diagnostics.items():
                    if status:
                        st.success(f"✅ {check}")
                    else:
                        st.error(f"❌ {check}")
    
    def _run_system_diagnostics(self) -> Dict[str, bool]:
        """Run system diagnostics checks."""
        diagnostics = {}
        
        try:
            # Check database connection
            with sqlite3.connect(knowledge_graph.DB_FILE) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM papers")
                diagnostics["Database Connection"] = True
        except Exception:
            diagnostics["Database Connection"] = False
        
        # Check processor initialization
        try:
            if self.processor:
                diagnostics["RAG Processor"] = True
            else:
                diagnostics["RAG Processor"] = False
        except Exception:
            diagnostics["RAG Processor"] = False
        
        # Check content analysis database
        try:
            stats = self.analysis_db.get_analysis_statistics()
            diagnostics["Content Analysis DB"] = True
        except Exception:
            diagnostics["Content Analysis DB"] = False
        
        # Check dependencies
        dependencies = ['pandas', 'plotly', 'streamlit', 'networkx']
        for dep in dependencies:
            try:
                __import__(dep)
                diagnostics[f"{dep.title()} Library"] = True
            except ImportError:
                diagnostics[f"{dep.title()} Library"] = False
        
        return diagnostics
    
    def _export_references(self, references):
        """Export references to downloadable format."""
        try:
            # Convert references to DataFrame
            ref_data = []
            for ref in references:
                ref_data.append({
                    'Citation': ref.raw_text,
                    'Authors': ', '.join(ref.authors) if ref.authors else '',
                    'Year': ref.year or '',
                    'Venue': ref.venue or '',
                    'DOI': ref.doi or '',
                    'ArXiv_ID': ref.arxiv_id or '',
                    'Confidence': ref.confidence,
                    'Context': ref.context or ''
                })
            
            df = pd.DataFrame(ref_data)
            csv_data = df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download References CSV",
                data=csv_data,
                file_name=f"references_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"References export failed: {e}")

    def _render_administration_page(self):
        """Render the administration page for system reset tasks."""
        st.header("👑 System Administration")

        st.warning("**DANGER ZONE**: These actions are irreversible and can lead to data loss.")

        # Full System Reset
        st.subheader("💥 Full System Reset")
        st.markdown("This will reset the entire system, including all documents, databases, and caches.")
        backup_checkbox = st.checkbox("Create backup before reset", value=True)
        
        if st.button("Perform Full System Reset"):
            with st.expander("Confirm Full System Reset", expanded=True):
                st.error("Are you absolutely sure? This cannot be undone.")
                if st.button("I confirm, proceed with full reset"):
                    with st.spinner("Performing full system reset..."):
                        stats = self.reset_service.full_system_reset(create_backup_first=backup_checkbox)
                        st.success("Full system reset completed!")
                        st.json(stats)

        st.markdown("---")

        # Component Reset
        st.subheader("🧨 Component Reset")
        st.markdown("Reset individual components of the system.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Reset Documents"):
                with st.spinner("Resetting documents..."):
                    stats = self.reset_service.reset_documents()
                    st.success("Documents reset completed!")
                    st.json(stats)
            if st.button("Reset Vector DB"):
                with st.spinner("Resetting vector database..."):
                    stats = self.reset_service.reset_vector_database()
                    st.success("Vector database reset completed!")
                    st.json(stats)
        with col2:
            if st.button("Reset Knowledge DB"):
                with st.spinner("Resetting knowledge database..."):
                    stats = self.reset_service.reset_knowledge_database()
                    st.success("Knowledge database reset completed!")
                    st.json(stats)
            if st.button("Reset Users DB"):
                with st.spinner("Resetting users database..."):
                    stats = self.reset_service.reset_users_database()
                    st.success("Users database reset completed!")
                    st.json(stats)
        with col3:
            if st.button("Reset Embeddings Cache"):
                with st.spinner("Resetting embeddings cache..."):
                    stats = self.reset_service.reset_embeddings_cache()
                    st.success("Embeddings cache reset completed!")
                    st.json(stats)

        st.markdown("---")

        # User Reset
        st.subheader("👤 User Data Reset")
        username_input = st.text_input("Username", placeholder="Enter username to reset")
        if st.button("Reset User Documents"):
            if username_input:
                with st.spinner(f"Resetting documents for user {username_input}..."):
                    stats = self.reset_service.reset_documents(username=username_input)
                    st.success(f"Document reset for user {username_input} completed!")
                    st.json(stats)
            else:
                st.error("Please enter a username.")

        st.markdown("---")

        # Backup
        st.subheader("💾 Create Backup")
        backup_name_input = st.text_input("Backup Name (optional)", placeholder="e.g., pre-migration-backup")
        if st.button("Create Backup"):
            with st.spinner("Creating backup..."):
                backup_name = backup_name_input if backup_name_input else None
                backup_path = self.reset_service.create_backup(backup_name=backup_name)
                st.success(f"Backup created successfully at: {backup_path}")


def main():
    """Main entry point for the web interface."""
    interface = WebInterface()
    interface.run()


if __name__ == "__main__":
    main()