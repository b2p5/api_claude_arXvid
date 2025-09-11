"""
Embedding cache system for storing and retrieving computed embeddings.
Provides persistent storage with deduplication and efficient retrieval.
"""

import os
import pickle
import hashlib
import sqlite3
import json
import time
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from config import get_config
from logger import get_logger, log_info, log_warning, log_error


@dataclass
class EmbeddingEntry:
    """Single embedding cache entry."""
    text_hash: str
    embedding: List[float]
    model_name: str
    embedding_dim: int
    created_at: float
    access_count: int = 0
    last_accessed: float = 0.0


class EmbeddingCache:
    """Persistent cache for text embeddings with deduplication."""
    
    def __init__(
        self, 
        cache_dir: str = "cache", 
        db_name: str = "embeddings.db",
        max_cache_size_mb: int = 500
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.db_path = self.cache_dir / db_name
        self.max_cache_size_mb = max_cache_size_mb
        self.logger = get_logger()
        
        # Initialize database
        self._init_database()
        
        # Load configuration
        config = get_config()
        self.default_model = config.models.embedding_model_name
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'total_queries': 0,
            'embeddings_cached': 0,
            'cache_size_mb': 0.0
        }
        
        self._update_stats()
    
    def _init_database(self):
        """Initialize the SQLite database for embedding cache."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create embeddings table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    text_hash TEXT PRIMARY KEY,
                    embedding_blob BLOB NOT NULL,
                    model_name TEXT NOT NULL,
                    embedding_dim INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    last_accessed REAL DEFAULT 0,
                    text_preview TEXT
                )
            """)
            
            # Create indices for performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_model_hash 
                ON embeddings(model_name, text_hash)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_last_accessed 
                ON embeddings(last_accessed)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_access_count 
                ON embeddings(access_count DESC)
            """)
            
            # Create metadata table for cache management
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cache_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)
            
            conn.commit()
    
    def _text_to_hash(self, text: str, model_name: str) -> str:
        """Convert text and model to a unique hash."""
        # Include model name in hash to avoid conflicts between different models
        content = f"{model_name}:{text}"
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def _serialize_embedding(self, embedding: List[float]) -> bytes:
        """Serialize embedding to bytes for storage."""
        return pickle.dumps(embedding, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _deserialize_embedding(self, blob: bytes) -> List[float]:
        """Deserialize embedding from bytes."""
        return pickle.loads(blob)
    
    def _update_stats(self):
        """Update cache statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Count total embeddings
                cursor.execute("SELECT COUNT(*) FROM embeddings")
                self.stats['embeddings_cached'] = cursor.fetchone()[0]
                
                # Calculate cache size
                cursor.execute("SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size()")
                cache_size_bytes = cursor.fetchone()[0]
                self.stats['cache_size_mb'] = cache_size_bytes / (1024 * 1024)
                
        except Exception as e:
            log_warning("Could not update cache stats", error=str(e))
    
    def get_embedding(
        self, 
        text: str, 
        model_name: str = None
    ) -> Optional[List[float]]:
        """
        Get cached embedding for text.
        
        Args:
            text: Input text
            model_name: Model name (uses default if None)
            
        Returns:
            Cached embedding or None if not found
        """
        if model_name is None:
            model_name = self.default_model
        
        text_hash = self._text_to_hash(text, model_name)
        self.stats['total_queries'] += 1
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT embedding_blob, embedding_dim 
                    FROM embeddings 
                    WHERE text_hash = ? AND model_name = ?
                """, (text_hash, model_name))
                
                result = cursor.fetchone()
                
                if result:
                    # Found in cache - update access stats
                    embedding_blob, embedding_dim = result
                    embedding = self._deserialize_embedding(embedding_blob)
                    
                    # Update access statistics
                    current_time = time.time()
                    cursor.execute("""
                        UPDATE embeddings 
                        SET access_count = access_count + 1, last_accessed = ?
                        WHERE text_hash = ? AND model_name = ?
                    """, (current_time, text_hash, model_name))
                    conn.commit()
                    
                    self.stats['hits'] += 1
                    return embedding
                else:
                    # Not found in cache
                    self.stats['misses'] += 1
                    return None
                    
        except Exception as e:
            log_error("Cache retrieval failed", text_hash=text_hash[:16], error=str(e))
            self.stats['misses'] += 1
            return None
    
    def store_embedding(
        self, 
        text: str, 
        embedding: List[float], 
        model_name: str = None
    ) -> bool:
        """
        Store embedding in cache.
        
        Args:
            text: Input text
            embedding: Computed embedding
            model_name: Model name (uses default if None)
            
        Returns:
            True if stored successfully
        """
        if model_name is None:
            model_name = self.default_model
        
        text_hash = self._text_to_hash(text, model_name)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Serialize embedding
                embedding_blob = self._serialize_embedding(embedding)
                current_time = time.time()
                
                # Create preview of text (first 100 chars)
                text_preview = text[:100] if len(text) > 100 else text
                
                # Insert or replace embedding
                cursor.execute("""
                    INSERT OR REPLACE INTO embeddings 
                    (text_hash, embedding_blob, model_name, embedding_dim, 
                     created_at, last_accessed, text_preview)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    text_hash, embedding_blob, model_name, len(embedding),
                    current_time, current_time, text_preview
                ))
                
                conn.commit()
                
                log_info("Embedding cached", 
                       text_hash=text_hash[:16], 
                       model=model_name,
                       dim=len(embedding))
                
                return True
                
        except Exception as e:
            log_error("Cache storage failed", 
                    text_hash=text_hash[:16], 
                    error=str(e))
            return False
    
    def get_batch_embeddings(
        self, 
        texts: List[str], 
        model_name: str = None
    ) -> Tuple[List[Optional[List[float]]], List[str]]:
        """
        Get cached embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            model_name: Model name (uses default if None)
            
        Returns:
            Tuple of (embeddings_list, missing_texts)
            embeddings_list contains None for missing entries
            missing_texts contains texts that need to be computed
        """
        if model_name is None:
            model_name = self.default_model
        
        embeddings = [None] * len(texts)
        missing_texts = []
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for i, text in enumerate(texts):
                    text_hash = self._text_to_hash(text, model_name)
                    
                    cursor.execute("""
                        SELECT embedding_blob 
                        FROM embeddings 
                        WHERE text_hash = ? AND model_name = ?
                    """, (text_hash, model_name))
                    
                    result = cursor.fetchone()
                    
                    if result:
                        embedding_blob = result[0]
                        embedding = self._deserialize_embedding(embedding_blob)
                        embeddings[i] = embedding
                        self.stats['hits'] += 1
                    else:
                        missing_texts.append(text)
                        self.stats['misses'] += 1
                    
                    self.stats['total_queries'] += 1
                
                # Batch update access stats for found embeddings
                found_hashes = []
                for i, text in enumerate(texts):
                    if embeddings[i] is not None:
                        found_hashes.append(self._text_to_hash(text, model_name))
                
                if found_hashes:
                    current_time = time.time()
                    placeholders = ','.join(['?' for _ in found_hashes])
                    cursor.execute(f"""
                        UPDATE embeddings 
                        SET access_count = access_count + 1, last_accessed = ?
                        WHERE text_hash IN ({placeholders}) AND model_name = ?
                    """, [current_time] + found_hashes + [model_name])
                    conn.commit()
        
        except Exception as e:
            log_error("Batch cache retrieval failed", error=str(e))
            # Return all as missing on error
            embeddings = [None] * len(texts)
            missing_texts = texts.copy()
        
        return embeddings, missing_texts
    
    def store_batch_embeddings(
        self, 
        texts: List[str], 
        embeddings: List[List[float]], 
        model_name: str = None
    ) -> int:
        """
        Store multiple embeddings in cache.
        
        Args:
            texts: List of input texts
            embeddings: List of computed embeddings
            model_name: Model name (uses default if None)
            
        Returns:
            Number of embeddings successfully stored
        """
        if model_name is None:
            model_name = self.default_model
        
        if len(texts) != len(embeddings):
            log_error("Text and embedding count mismatch", 
                    texts=len(texts), 
                    embeddings=len(embeddings))
            return 0
        
        stored_count = 0
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                current_time = time.time()
                
                batch_data = []
                for text, embedding in zip(texts, embeddings):
                    text_hash = self._text_to_hash(text, model_name)
                    embedding_blob = self._serialize_embedding(embedding)
                    text_preview = text[:100] if len(text) > 100 else text
                    
                    batch_data.append((
                        text_hash, embedding_blob, model_name, len(embedding),
                        current_time, current_time, text_preview
                    ))
                
                # Batch insert
                cursor.executemany("""
                    INSERT OR REPLACE INTO embeddings 
                    (text_hash, embedding_blob, model_name, embedding_dim, 
                     created_at, last_accessed, text_preview)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, batch_data)
                
                conn.commit()
                stored_count = len(batch_data)
                
                log_info("Batch embeddings cached", 
                       count=stored_count, 
                       model=model_name)
        
        except Exception as e:
            log_error("Batch cache storage failed", error=str(e))
        
        return stored_count
    
    def cleanup_old_entries(self, max_age_days: int = 30) -> int:
        """
        Remove old cache entries.
        
        Args:
            max_age_days: Maximum age in days for cache entries
            
        Returns:
            Number of entries removed
        """
        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Remove old entries
                cursor.execute("""
                    DELETE FROM embeddings 
                    WHERE last_accessed < ?
                """, (cutoff_time,))
                
                removed_count = cursor.rowcount
                conn.commit()
                
                log_info("Cache cleanup completed", 
                       removed=removed_count, 
                       max_age_days=max_age_days)
                
                # Update stats
                self._update_stats()
                
                return removed_count
        
        except Exception as e:
            log_error("Cache cleanup failed", error=str(e))
            return 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get detailed cache statistics."""
        self._update_stats()
        
        hit_rate = (self.stats['hits'] / self.stats['total_queries'] * 100) if self.stats['total_queries'] > 0 else 0
        
        detailed_stats = self.stats.copy()
        detailed_stats['hit_rate_percent'] = hit_rate
        detailed_stats['miss_rate_percent'] = 100 - hit_rate
        
        # Get model distribution
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT model_name, COUNT(*) as count, AVG(embedding_dim) as avg_dim
                    FROM embeddings 
                    GROUP BY model_name
                """)
                
                model_stats = {}
                for row in cursor.fetchall():
                    model_name, count, avg_dim = row
                    model_stats[model_name] = {
                        'count': count,
                        'avg_dimension': int(avg_dim) if avg_dim else 0
                    }
                
                detailed_stats['models'] = model_stats
        
        except Exception as e:
            log_warning("Could not get model statistics", error=str(e))
        
        return detailed_stats
    
    def vacuum_database(self):
        """Optimize database by running VACUUM."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                log_info("Starting database vacuum operation")
                conn.execute("VACUUM")
                log_info("Database vacuum completed")
                self._update_stats()
        except Exception as e:
            log_error("Database vacuum failed", error=str(e))


# Global cache instance
_cache_instance = None

def get_embedding_cache() -> EmbeddingCache:
    """Get the global embedding cache instance."""
    global _cache_instance
    if _cache_instance is None:
        config = get_config()
        _cache_instance = EmbeddingCache()
    return _cache_instance


class CachedEmbeddingModel:
    """Wrapper around embedding model with automatic caching."""
    
    def __init__(self, model_name: str = None, cache: EmbeddingCache = None):
        config = get_config()
        self.model_name = model_name or config.models.embedding_model_name
        self.cache = cache or get_embedding_cache()
        self._model = None
        self.logger = get_logger()
    
    def _get_model(self):
        """Lazy loading of the embedding model."""
        if self._model is None:
            from langchain_huggingface import HuggingFaceEmbeddings
            self._model = HuggingFaceEmbeddings(model_name=self.model_name)
        return self._model
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents with caching."""
        # Check cache for existing embeddings
        cached_embeddings, missing_texts = self.cache.get_batch_embeddings(texts, self.model_name)
        
        if not missing_texts:
            # All embeddings found in cache
            log_info("All embeddings found in cache", count=len(texts))
            return [emb for emb in cached_embeddings if emb is not None]
        
        # Compute missing embeddings
        log_info("Computing missing embeddings", 
               total=len(texts), 
               cached=len(texts) - len(missing_texts), 
               missing=len(missing_texts))
        
        model = self._get_model()
        new_embeddings = model.embed_documents(missing_texts)
        
        # Store new embeddings in cache
        self.cache.store_batch_embeddings(missing_texts, new_embeddings, self.model_name)
        
        # Merge cached and new embeddings
        result_embeddings = []
        missing_idx = 0
        
        for cached_emb in cached_embeddings:
            if cached_emb is not None:
                result_embeddings.append(cached_emb)
            else:
                result_embeddings.append(new_embeddings[missing_idx])
                missing_idx += 1
        
        return result_embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query with caching."""
        # Check cache first
        cached_embedding = self.cache.get_embedding(text, self.model_name)
        
        if cached_embedding is not None:
            return cached_embedding
        
        # Compute new embedding
        model = self._get_model()
        embedding = model.embed_query(text)
        
        # Store in cache
        self.cache.store_embedding(text, embedding, self.model_name)
        
        return embedding