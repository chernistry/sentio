#!/usr/bin/env python3
"""
Advanced text chunking with multiple strategies and optimization.

This module provides intelligent text chunking capabilities with support for
different splitting strategies, semantic preservation, and performance optimization.
"""

import logging
import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, TextNode

logger = logging.getLogger(__name__)


class ChunkingStrategy(Enum):
    """Available chunking strategies."""
    SENTENCE = "sentence"           # Split by sentences (default)
    SEMANTIC = "semantic"           # Semantic-aware splitting
    FIXED = "fixed"                # Fixed-size chunks
    PARAGRAPH = "paragraph"         # Split by paragraphs
    HYBRID = "hybrid"              # Combination of strategies


class ChunkingError(Exception):
    """Custom exception for chunking-related errors."""
    pass


class BaseChunker(ABC):
    """Abstract base class for text chunkers."""
    
    @abstractmethod
    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[TextNode]:
        """Chunk text into nodes."""
        pass


class SentenceChunker(BaseChunker):
    """Sentence-based chunking with intelligent boundary detection."""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=" ",
            paragraph_separator="\n\n\n",
            chunking_tokenizer_fn=self._smart_tokenizer
        )
    
    def _smart_tokenizer(self, text: str) -> List[str]:
        """Smart tokenizer that preserves semantic units."""
        # Simple word-based tokenization with punctuation handling
        return re.findall(r'\b\w+\b|[^\w\s]', text)
    
    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[TextNode]:
        """Chunk text using sentence boundaries."""
        try:
            nodes = self.splitter.get_nodes_from_documents([Document(text=text, metadata=metadata or {})])
            return nodes
        except Exception as e:
            raise ChunkingError(f"Sentence chunking failed: {e}")


class SemanticChunker(BaseChunker):
    """Semantic-aware chunking that preserves meaning."""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Patterns for semantic boundaries
        self.paragraph_pattern = re.compile(r'\n\s*\n')
        self.section_pattern = re.compile(r'\n#+\s+.*?\n|^\s*\d+\.\s+.*?$', re.MULTILINE)
        self.list_pattern = re.compile(r'^\s*[-*•]\s+|^\s*\d+\.\s+', re.MULTILINE)
    
    def _find_semantic_boundaries(self, text: str) -> List[int]:
        """Find semantic boundaries in text."""
        boundaries = [0]
        
        # Add paragraph boundaries
        for match in self.paragraph_pattern.finditer(text):
            boundaries.append(match.end())
        
        # Add section boundaries
        for match in self.section_pattern.finditer(text):
            boundaries.append(match.start())
        
        # Add list item boundaries
        for match in self.list_pattern.finditer(text):
            boundaries.append(match.start())
        
        boundaries.append(len(text))
        return sorted(set(boundaries))
    
    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[TextNode]:
        """Chunk text using semantic boundaries."""
        try:
            boundaries = self._find_semantic_boundaries(text)
            chunks = []
            current_chunk = ""
            current_start = 0
            
            for boundary in boundaries[1:]:
                segment = text[current_start:boundary].strip()
                
                if len(current_chunk) + len(segment) <= self.chunk_size:
                    current_chunk += segment
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        # Handle overlap
                        overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                        current_chunk = current_chunk[overlap_start:]
                    current_chunk += segment
                
                current_start = boundary
            
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            
            # Convert to TextNodes
            nodes = []
            for i, chunk in enumerate(chunks):
                if chunk:  # Only create nodes for non-empty chunks
                    node_metadata = (metadata or {}).copy()
                    node_metadata.update({
                        'chunk_index': i,
                        'chunk_type': 'semantic',
                        'chunk_size': len(chunk)
                    })
                    nodes.append(TextNode(text=chunk, metadata=node_metadata))
            
            return nodes
            
        except Exception as e:
            raise ChunkingError(f"Semantic chunking failed: {e}")


class FixedChunker(BaseChunker):
    """Fixed-size chunking with word boundary preservation."""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[TextNode]:
        """Chunk text into fixed-size pieces."""
        try:
            words = text.split()
            chunks = []
            current_chunk = []
            current_size = 0
            
            for word in words:
                word_size = len(word) + 1  # +1 for space
                
                if current_size + word_size > self.chunk_size and current_chunk:
                    # Create chunk
                    chunk_text = ' '.join(current_chunk)
                    chunks.append(chunk_text)
                    
                    # Handle overlap
                    overlap_words = []
                    overlap_size = 0
                    for w in reversed(current_chunk):
                        w_size = len(w) + 1
                        if overlap_size + w_size <= self.chunk_overlap:
                            overlap_words.insert(0, w)
                            overlap_size += w_size
                        else:
                            break
                    
                    current_chunk = overlap_words
                    current_size = overlap_size
                
                current_chunk.append(word)
                current_size += word_size
            
            # Add final chunk
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            
            # Convert to TextNodes
            nodes = []
            for i, chunk in enumerate(chunks):
                node_metadata = (metadata or {}).copy()
                node_metadata.update({
                    'chunk_index': i,
                    'chunk_type': 'fixed',
                    'chunk_size': len(chunk)
                })
                nodes.append(TextNode(text=chunk, metadata=node_metadata))
            
            return nodes
            
        except Exception as e:
            raise ChunkingError(f"Fixed chunking failed: {e}")


class ParagraphChunker(BaseChunker):
    """Paragraph-based chunking with size constraints."""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[TextNode]:
        """Chunk text by paragraphs with size limits."""
        try:
            paragraphs = re.split(r'\n\s*\n', text)
            chunks = []
            current_chunk = ""
            
            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if not paragraph:
                    continue
                
                if len(current_chunk) + len(paragraph) <= self.chunk_size:
                    if current_chunk:
                        current_chunk += "\n\n" + paragraph
                    else:
                        current_chunk = paragraph
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                        # Handle overlap at word level
                        words = current_chunk.split()
                        overlap_words = []
                        overlap_size = 0
                        for word in reversed(words):
                            word_size = len(word) + 1
                            if overlap_size + word_size <= self.chunk_overlap:
                                overlap_words.insert(0, word)
                                overlap_size += word_size
                            else:
                                break
                        current_chunk = ' '.join(overlap_words)
                        if current_chunk:
                            current_chunk += "\n\n" + paragraph
                        else:
                            current_chunk = paragraph
                    else:
                        current_chunk = paragraph
            
            if current_chunk:
                chunks.append(current_chunk)
            
            # Convert to TextNodes
            nodes = []
            for i, chunk in enumerate(chunks):
                node_metadata = (metadata or {}).copy()
                node_metadata.update({
                    'chunk_index': i,
                    'chunk_type': 'paragraph',
                    'chunk_size': len(chunk)
                })
                nodes.append(TextNode(text=chunk, metadata=node_metadata))
            
            return nodes
            
        except Exception as e:
            raise ChunkingError(f"Paragraph chunking failed: {e}")


class TextChunker:
    """
    Advanced text chunker with multiple strategies and optimization.
    
    Features:
    - Multiple chunking strategies
    - Automatic strategy selection
    - Quality validation
    - Performance monitoring
    - Content-aware chunking
    """
    
    def __init__(
        self, 
        chunk_size: int = 512, 
        chunk_overlap: int = 64,
        strategy: ChunkingStrategy = ChunkingStrategy.SENTENCE,
        min_chunk_size: int = 50,
        max_chunk_size: Optional[int] = None,
        preserve_code_blocks: bool = True,
        preserve_tables: bool = True
    ):
        """
        Initialize text chunker with configuration.
        
        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks in characters
            strategy: Chunking strategy to use
            min_chunk_size: Minimum acceptable chunk size
            max_chunk_size: Maximum acceptable chunk size (None for 2x chunk_size)
            preserve_code_blocks: Whether to preserve code block integrity
            preserve_tables: Whether to preserve table structure
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size or (chunk_size * 2)
        self.preserve_code_blocks = preserve_code_blocks
        self.preserve_tables = preserve_tables
        
        # Initialize strategy-specific chunkers
        self.chunkers = {
            ChunkingStrategy.SENTENCE: SentenceChunker(chunk_size, chunk_overlap),
            ChunkingStrategy.SEMANTIC: SemanticChunker(chunk_size, chunk_overlap),
            ChunkingStrategy.FIXED: FixedChunker(chunk_size, chunk_overlap),
            ChunkingStrategy.PARAGRAPH: ParagraphChunker(chunk_size, chunk_overlap),
        }
        
        # Performance tracking
        self.stats = {
            'documents_processed': 0,
            'total_chunks': 0,
            'avg_chunk_size': 0.0,
            'strategy_usage': {strategy.value: 0 for strategy in ChunkingStrategy}
        }
        
        logger.info(f"✓ Text chunker initialized with strategy: {strategy.value}")
    
    def _preprocess_text(self, text: str) -> Tuple[str, Dict]:
        """Preprocess text and extract special regions."""
        special_regions = {
            'code_blocks': [],
            'tables': [],
            'equations': []
        }
        
        processed_text = text
        
        if self.preserve_code_blocks:
            # Find code blocks
            code_pattern = re.compile(r'```[\s\S]*?```|`[^`\n]+`', re.MULTILINE)
            for match in code_pattern.finditer(text):
                special_regions['code_blocks'].append({
                    'start': match.start(),
                    'end': match.end(),
                    'content': match.group()
                })
        
        if self.preserve_tables:
            # Find markdown tables
            table_pattern = re.compile(r'\|.*?\|.*?\n(?:\|[-:]+\|.*?\n)?(?:\|.*?\|.*?\n)*', re.MULTILINE)
            for match in table_pattern.finditer(text):
                special_regions['tables'].append({
                    'start': match.start(),
                    'end': match.end(),
                    'content': match.group()
                })
        
        return processed_text, special_regions
    
    def _select_strategy(self, text: str, metadata: Optional[Dict] = None) -> ChunkingStrategy:
        """Automatically select the best chunking strategy for the text."""
        if self.strategy != ChunkingStrategy.HYBRID:
            return self.strategy
        
        # Analyze text characteristics
        has_sections = bool(re.search(r'\n#+\s+.*?\n|^\s*\d+\.\s+.*?$', text, re.MULTILINE))
        has_paragraphs = '\n\n' in text
        has_lists = bool(re.search(r'^\s*[-*•]\s+|^\s*\d+\.\s+', text, re.MULTILINE))
        avg_sentence_length = len(text) / max(1, text.count('.') + text.count('!') + text.count('?'))
        
        # Strategy selection logic
        if has_sections and has_paragraphs:
            return ChunkingStrategy.SEMANTIC
        elif has_paragraphs and avg_sentence_length > 100:
            return ChunkingStrategy.PARAGRAPH
        elif avg_sentence_length < 200:
            return ChunkingStrategy.SENTENCE
        else:
            return ChunkingStrategy.FIXED
    
    def _validate_chunks(self, chunks: List[TextNode]) -> List[TextNode]:
        """Validate, filter, and (if needed) iteratively split oversized chunks.

        The previous recursive approach occasionally hit Python's recursion
        limit on very large documents. We now use an explicit stack to keep the
        call-depth constant while still enforcing `max_chunk_size`.
        """

        valid: List[TextNode] = []
        stack: List[TextNode] = list(chunks)  # LIFO stack for iterative processing

        while stack:
            node = stack.pop()
            text = node.text.strip()

            # 1. Basic size check – too small → skip
            if len(text) < self.min_chunk_size:
                logger.debug("Skipping chunk below minimum size: %d < %d", len(text), self.min_chunk_size)
                continue

            # 2. Empty / whitespace only
            if not text or text.isspace():
                continue

            # 3. Oversized – split once and push sub-nodes onto stack (no recursion)
            if len(text) > self.max_chunk_size:
                logger.warning("Chunk exceeds max size (%d > %d). Splitting.", len(text), self.max_chunk_size)

                sub_splitter = SentenceSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    separator=" ",
                    paragraph_separator="\n\n",
                )
                sub_nodes = sub_splitter.get_nodes_from_documents([
                    Document(text=text, metadata=node.metadata or {})
                ])

                # Fallback: if splitter failed (returned no nodes or returned the same chunk)
                if not sub_nodes or all(len(n.text) >= len(text) for n in sub_nodes):
                    logger.debug("SentenceSplitter fallback – hard slicing oversized chunk (%d chars)", len(text))

                    step = self.max_chunk_size - self.chunk_overlap
                    hard_chunks: List[str] = [text[i : i + self.max_chunk_size] for i in range(0, len(text), step)]

                    sub_nodes = [
                        TextNode(
                            text=c,
                            metadata={**(node.metadata or {}), "chunk_type": "hard_slice"},
                        )
                        for c in hard_chunks
                    ]
 
                # Push new nodes onto stack for further validation
                stack.extend(sub_nodes)
                continue  # Skip original oversized node

            # 4. Alpha-ratio filter – skip junk
            alpha_ratio = sum(c.isalpha() for c in text) / max(1, len(text))
            if alpha_ratio < 0.1:
                logger.debug("Skipping chunk with low alpha ratio: %.2f", alpha_ratio)
                continue

            valid.append(node)

        return valid
    
    def split(self, documents: List[Document]) -> List[TextNode]:
        """
        Split documents into chunks using the configured strategy.
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            List of text nodes (chunks)
        """
        all_nodes = []
        
        for doc in documents:
            try:
                # Preprocess text
                text, special_regions = self._preprocess_text(doc.text)
                
                # Select strategy
                strategy = self._select_strategy(text, doc.metadata)
                chunker = self.chunkers[strategy]
                
                # Chunk the text
                nodes = chunker.chunk_text(text, doc.metadata)
                
                # Validate chunks
                valid_nodes = self._validate_chunks(nodes)
                
                # Update statistics
                self.stats['strategy_usage'][strategy.value] += 1
                self.stats['total_chunks'] += len(valid_nodes)
                
                # Add document-level metadata
                for node in valid_nodes:
                    if node.metadata is None:
                        node.metadata = {}
                    node.metadata.update({
                        'document_id': doc.doc_id,
                        'chunking_strategy': strategy.value,
                        'source_length': len(doc.text)
                    })
                
                all_nodes.extend(valid_nodes)
                
                logger.debug(f"Chunked document into {len(valid_nodes)} chunks using {strategy.value} strategy")
                
            except Exception as e:
                logger.error(f"Failed to chunk document: {e}")
                # Continue with other documents
                continue
        
        # Update global statistics
        self.stats['documents_processed'] += len(documents)
        if all_nodes:
            total_size = sum(len(node.text) for node in all_nodes)
            self.stats['avg_chunk_size'] = total_size / len(all_nodes)
        
        logger.info(f"✓ Processed {len(documents)} documents into {len(all_nodes)} chunks")
        return all_nodes
    
    def get_stats(self) -> Dict:
        """Get chunking performance statistics."""
        stats = self.stats.copy()
        stats['config'] = {
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'strategy': self.strategy.value,
            'min_chunk_size': self.min_chunk_size,
            'max_chunk_size': self.max_chunk_size
        }
        return stats
    
    def reset_stats(self) -> None:
        """Reset performance statistics."""
        self.stats = {
            'documents_processed': 0,
            'total_chunks': 0,
            'avg_chunk_size': 0.0,
            'strategy_usage': {strategy.value: 0 for strategy in ChunkingStrategy}
        }
        logger.info("Chunking statistics reset")
    
    def __repr__(self) -> str:
        return (f"TextChunker(size={self.chunk_size}, overlap={self.chunk_overlap}, "
                f"strategy={self.strategy.value})") 