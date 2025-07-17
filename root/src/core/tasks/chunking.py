#!/usr/bin/env python3
"""
Advanced text chunking with multiple strategies and optimization.
"""

import logging
import re
import asyncio

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from collections import deque
from functools import lru_cache

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, TextNode

 # Pre‑compiled token regex for fast reuse
TOKEN_RE: re.Pattern[str] = re.compile(r'\b\w+\b|[^\w\s]')

logger = logging.getLogger(__name__)

# ==== DEFINITIONS & CONSTANTS ==== #
# --► ENUMS & EXCEPTIONS

class ChunkingStrategy(Enum):
    """Available chunking strategies for text chunking."""

    SENTENCE = "sentence"
    SEMANTIC = "semantic"
    FIXED = "fixed"
    PARAGRAPH = "paragraph"
    HYBRID = "hybrid"


class ChunkingError(Exception):
    """Exception raised for errors during text chunking."""


# ==== CORE PROCESSING MODULE ==== #
# --► DATA EXTRACTION & TRANSFORMATION
# ⚠️ POTENTIALLY ERROR-PRONE LOGIC

class BaseChunker(ABC):
    """Abstract base class for all chunker implementations."""
    __slots__: Tuple = ()

    @abstractmethod
    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[TextNode]:
        """
        Chunk text into TextNode instances.

        Args:
            text: The input text to be chunked.
            metadata: Optional metadata to attach to each node.

        Returns:
            A list of TextNode instances.

        Raises:
            ChunkingError: If chunking fails.
        """
        ...


class SentenceChunker(BaseChunker):
    """Sentence-based chunking with intelligent boundary detection."""
    __slots__: Tuple = ("splitter",)

    def __init__(
        self,
        splitter: SentenceSplitter
    ) -> None:
        """
        Initialize SentenceChunker with a pre-configured splitter.
        
        Args:
            splitter: An initialized SentenceSplitter instance.
        """
        self.splitter = splitter

    @classmethod
    async def create(
        cls,
        chunk_size: int = 512,
        chunk_overlap: int = 64
    ) -> "SentenceChunker":
        """
        Asynchronously create a SentenceChunker instance.

        This method handles the potentially blocking initialization of the
        tokenizer in an async-friendly way.

        Args:
            chunk_size: Maximum tokens per chunk.
            chunk_overlap: Overlap tokens between chunks.

        Returns:
            A new instance of SentenceChunker.
        """
        splitter = await asyncio.to_thread(
            SentenceSplitter,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=" ",
            paragraph_separator="\n\n\n",
            chunking_tokenizer_fn=cls._smart_tokenizer,
        )
        return cls(splitter)

    @staticmethod
    @lru_cache(maxsize=1024)
    def _smart_tokenizer(text: str) -> List[str]:
        """
        Tokenize text preserving semantic units (cached).

        Args:
            text: Raw input text.

        Returns:
            A list of tokens including punctuation.
        """
        return TOKEN_RE.findall(text)


    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[TextNode]:
        """
        Chunk text using sentence boundaries.

        Args:
            text: Text to chunk.
            metadata: Optional metadata for nodes.

        Returns:
            List of TextNode objects.

        Raises:
            ChunkingError: On failure during chunking.
        """
        try:
            nodes = self.splitter.get_nodes_from_documents([
                Document(text=text, metadata=metadata or {})
            ])
            return nodes

        except Exception as exc:
            raise ChunkingError(f"Sentence chunking failed: {exc}") from exc



class SemanticChunker(BaseChunker):
    """Semantic-aware chunking that preserves meaning."""
    __slots__: Tuple = ("chunk_size", "chunk_overlap",)

    _paragraph_pattern: re.Pattern[str] = re.compile(r'\n\s*\n')
    _section_pattern: re.Pattern[str] = re.compile(
        r'\n#+\s+.*?\n|^\s*\d+\.\s+.*?$',
        re.MULTILINE,
    )
    _list_pattern: re.Pattern[str] = re.compile(
        r'^\s*[-*•]\s+|^\s*\d+\.\s+',
        re.MULTILINE,
    )

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64
    ) -> None:
        """
        Initialize SemanticChunker.

        Args:
            chunk_size: Target chunk size in characters.
            chunk_overlap: Character overlap between chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap


    def _find_semantic_boundaries(self, text: str) -> List[int]:
        """
        Identify semantic boundaries in text.

        Args:
            text: Input text string.

        Returns:
            Sorted list of boundary indices.
        """
        boundaries: List[int] = [0]

        for match in self._paragraph_pattern.finditer(text):
            boundaries.append(match.end())

        for match in self._section_pattern.finditer(text):
            boundaries.append(match.start())

        for match in self._list_pattern.finditer(text):
            boundaries.append(match.start())

        boundaries.append(len(text))
        return sorted(set(boundaries))


    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[TextNode]:
        """
        Chunk text using semantic content boundaries.

        Args:
            text: Raw input text.
            metadata: Optional metadata dict.

        Returns:
            List of semantic TextNode objects.

        Raises:
            ChunkingError: On failure to chunk.
        """
        try:
            boundaries = self._find_semantic_boundaries(text)
            chunks: List[str] = []
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
                        overlap_start = max(
                            0,
                            len(current_chunk) - self.chunk_overlap
                        )
                        current_chunk = current_chunk[overlap_start:]

                    current_chunk += segment

                current_start = boundary

            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            nodes: List[TextNode] = []
            for idx, chunk in enumerate(chunks):
                metadata_copy = (metadata or {}).copy()
                metadata_copy.update({
                    "chunk_index": idx,
                    "chunk_type": "semantic",
                    "chunk_size": len(chunk),
                })
                nodes.append(TextNode(text=chunk, metadata=metadata_copy))

            return nodes

        except Exception as exc:
            raise ChunkingError(f"Semantic chunking failed: {exc}") from exc



class FixedChunker(BaseChunker):
    """Fixed-size chunking with word boundary preservation."""
    __slots__: Tuple = ("chunk_size", "chunk_overlap",)

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64
    ) -> None:
        """
        Initialize FixedChunker.

        Args:
            chunk_size: Target chunk size in characters.
            chunk_overlap: Character overlap for chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap


    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[TextNode]:
        """
        Chunk text into fixed-size pieces preserving words.

        Args:
            text: Raw input text.
            metadata: Optional metadata.

        Returns:
            List of TextNode instances.

        Raises:
            ChunkingError: On failure during fixed chunking.
        """
        try:
            words = text.split()
            chunks: List[str] = []
            current_chunk: List[str] = []
            current_size = 0

            for word in words:
                word_size = len(word) + 1
                if current_size + word_size > self.chunk_size and current_chunk:
                    chunks.append(" ".join(current_chunk))

                    # Overlap handling
                    overlap_words: List[str] = []
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

            if current_chunk:
                chunks.append(" ".join(current_chunk))

            nodes: List[TextNode] = []
            for idx, chunk in enumerate(chunks):
                meta = (metadata or {}).copy()
                meta.update({
                    "chunk_index": idx,
                    "chunk_type": "fixed",
                    "chunk_size": len(chunk),
                })
                nodes.append(TextNode(text=chunk, metadata=meta))

            return nodes

        except Exception as exc:
            raise ChunkingError(f"Fixed chunking failed: {exc}") from exc



class ParagraphChunker(BaseChunker):
    """Paragraph-based chunking with size constraints."""
    __slots__: Tuple = ("chunk_size", "chunk_overlap",)

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64
    ) -> None:
        """
        Initialize ParagraphChunker.

        Args:
            chunk_size: Target chunk size in characters.
            chunk_overlap: Character overlap between chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap


    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[TextNode]:
        """
        Chunk text by paragraphs respecting size limits.

        Args:
            text: Raw input text.
            metadata: Optional metadata.

        Returns:
            List of paragraph TextNode instances.

        Raises:
            ChunkingError: On failure during paragraph chunking.
        """
        try:
            paragraphs = re.split(r'\n\s*\n', text)
            chunks: List[str] = []
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

                        # Overlap at word-level
                        words = current_chunk.split()
                        overlap_words: List[str] = []
                        overlap_size = 0

                        for w in reversed(words):
                            w_size = len(w) + 1
                            if overlap_size + w_size <= self.chunk_overlap:
                                overlap_words.insert(0, w)
                                overlap_size += w_size
                            else:
                                break

                        current_chunk = (
                            " ".join(overlap_words) + "\n\n"
                            + paragraph
                            if overlap_words else paragraph
                        )
                    else:
                        current_chunk = paragraph

            if current_chunk:
                chunks.append(current_chunk)

            nodes: List[TextNode] = []
            for idx, chunk in enumerate(chunks):
                meta = (metadata or {}).copy()
                meta.update({
                    "chunk_index": idx,
                    "chunk_type": "paragraph",
                    "chunk_size": len(chunk),
                })
                nodes.append(TextNode(text=chunk, metadata=meta))

            return nodes

        except Exception as exc:
            raise ChunkingError(f"Paragraph chunking failed: {exc}") from exc



# ==== ORCHESTRATION & VALIDATION MODULE ==== #
# --► STRATEGY SELECTION & STATS TRACKING

class TextChunker:
    """
    Facade for various text chunking strategies.
    
    This class orchestrates the chunking process, including text preprocessing,
    strategy selection, and post-processing validation. It is the primary
    entry point for chunking text content.
    """
    __slots__: Tuple = (
        "chunk_size", "chunk_overlap", "strategy",
        "min_chunk_size", "max_chunk_size",
        "preserve_code_blocks", "preserve_tables",
        "_chunkers", "_stats", "_code_placeholder_pattern",
        "_table_placeholder_pattern"
    )

    def __init__(
        self,
        chunkers: Dict[ChunkingStrategy, BaseChunker],
        chunk_size: int,
        chunk_overlap: int,
        strategy: ChunkingStrategy,
        min_chunk_size: int,
        max_chunk_size: Optional[int],
        preserve_code_blocks: bool,
        preserve_tables: bool,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.preserve_code_blocks = preserve_code_blocks
        self.preserve_tables = preserve_tables
        
        self._chunkers = chunkers
        self._stats: Dict[str, Any] = {"chunks_created": 0, "errors": 0}
        
        self._code_placeholder_pattern = re.compile(r"(__CODE_BLOCK_\d+__)")
        self._table_placeholder_pattern = re.compile(r"(__TABLE_\d+__)")

    @classmethod
    async def create(
        cls,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        strategy: ChunkingStrategy = ChunkingStrategy.SENTENCE,
        min_chunk_size: int = 50,
        max_chunk_size: Optional[int] = None,
        preserve_code_blocks: bool = True,
        preserve_tables: bool = True,
    ) -> "TextChunker":
        """
        Asynchronously create a TextChunker instance.

        This factory method ensures that all underlying chunker strategies
        are initialized in a non-blocking way.
        """
        sentence_chunker = await SentenceChunker.create(chunk_size, chunk_overlap)
        
        chunkers = {
            ChunkingStrategy.SENTENCE: sentence_chunker,
            ChunkingStrategy.SEMANTIC: SemanticChunker(chunk_size, chunk_overlap),
            ChunkingStrategy.FIXED: FixedChunker(chunk_size, chunk_overlap),
            ChunkingStrategy.PARAGRAPH: ParagraphChunker(chunk_size, chunk_overlap),
        }
        chunkers[ChunkingStrategy.HYBRID] = chunkers[strategy]

        return cls(
            chunkers=chunkers,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strategy=strategy,
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size,
            preserve_code_blocks=preserve_code_blocks,
            preserve_tables=preserve_tables,
        )

    def _preprocess_text(
        self,
        text: str
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Preprocess text and extract special regions (code, tables).

        Args:
            text: Raw document text.

        Returns:
            Tuple of processed text and regions metadata.

        Assumptions:
            preserve_code_blocks and preserve_tables flags are honored.
        """
        special_regions: Dict[str, Any] = {
            "code_blocks": [],
            "tables": [],
            "equations": []
        }

        processed_text = text

        if self.preserve_code_blocks:
            code_pattern = re.compile(r'```[\s\S]*?```|`[^`\n]+`',
                                      re.MULTILINE)
            for match in code_pattern.finditer(text):
                special_regions["code_blocks"].append({
                    "start": match.start(),
                    "end": match.end(),
                    "content": match.group(),
                })

        if self.preserve_tables:
            table_pattern = re.compile(
                r'\|.*?\|.*?\n(?:\|[-:]+\|.*?\n)?(?:\|.*?\|.*?\n)*',
                re.MULTILINE,
            )
            for match in table_pattern.finditer(text):
                special_regions["tables"].append({
                    "start": match.start(),
                    "end": match.end(),
                    "content": match.group(),
                })

        return processed_text, special_regions



    def _select_strategy(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ChunkingStrategy:
        """
        Automatically select the best chunking strategy.

        Args:
            text: Preprocessed document text.
            metadata: Optional metadata dict.

        Returns:
            Selected ChunkingStrategy.
        """
        if self.strategy != ChunkingStrategy.HYBRID:
            return self.strategy

        has_sections = bool(
            re.search(r'\n#+\s+.*?\n|^\s*\d+\.\s+.*?$',
                      text, re.MULTILINE)
        )
        has_paragraphs = '\n\n' in text
        has_lists = bool(
            re.search(r'^\s*[-*•]\s+|^\s*\d+\.\s+',
                      text, re.MULTILINE)
        )
        avg_sentence_length = (
            len(text) /
            max(1, text.count('.') + text.count('!') + text.count('?'))
        )

        if has_sections and has_paragraphs:
            return ChunkingStrategy.SEMANTIC
        elif has_paragraphs and avg_sentence_length > 100:
            return ChunkingStrategy.PARAGRAPH
        elif avg_sentence_length < 200:
            return ChunkingStrategy.SENTENCE

        return ChunkingStrategy.FIXED



    def _validate_chunks(
        self,
        chunks: List[TextNode]
    ) -> List[TextNode]:
        """
        Validate, filter, and split oversized chunks iteratively.

        Args:
            chunks: Initial list of TextNode objects.

        Returns:
            List of validated TextNode objects.
        """
        valid: List[TextNode] = []
        stack = deque(chunks)

        while stack:
            node = stack.pop()
            text = node.text.strip()

            if len(text) < self.min_chunk_size:
                continue

            if not text or text.isspace():
                continue

            if len(text) > self.max_chunk_size:
                sub_splitter = SentenceSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    separator=" ",
                    paragraph_separator="\n\n",
                )
                sub_nodes = sub_splitter.get_nodes_from_documents([
                    Document(text=text, metadata=node.metadata or {})
                ])

                if not sub_nodes or all(
                    len(n.text) >= len(text) for n in sub_nodes
                ):
                    step = self.max_chunk_size - self.chunk_overlap
                    hard_chunks = [
                        text[i : i + self.max_chunk_size]
                        for i in range(0, len(text), step)
                    ]
                    sub_nodes = [
                        TextNode(
                            text=c,
                            metadata={**(node.metadata or {}),
                                      "chunk_type": "hard_slice"},
                        )
                        for c in hard_chunks
                    ]

                stack.extend(sub_nodes)
                continue

            alpha_ratio = sum(c.isalpha() for c in text) / len(text)
            if alpha_ratio < 0.1:
                continue

            valid.append(node)

        return valid



    def split(
        self,
        documents: List[Document]
    ) -> List[TextNode]:
        """
        Split documents into chunks using the configured strategy.

        Args:
            documents: List of Document instances to chunk.

        Returns:
            List of validated TextNode chunks.
        """
        all_nodes: List[TextNode] = []

        for doc in documents:
            try:
                text, regions = self._preprocess_text(doc.text)

                strategy = self._select_strategy(text, doc.metadata)
                chunker = self._chunkers[strategy]

                nodes = chunker.chunk_text(text, doc.metadata)
                valid_nodes = self._validate_chunks(nodes)

                self._stats["strategy_usage"][strategy.value] += 1
                self._stats["total_chunks"] += len(valid_nodes)

                for node in valid_nodes:
                    node.metadata = node.metadata or {}
                    node.metadata.update({
                        "document_id": doc.doc_id,
                        "chunking_strategy": strategy.value,
                        "source_length": len(doc.text),
                    })

                all_nodes.extend(valid_nodes)
                logger.debug(
                    "Document %s produced %d valid chunks via %s strategy",
                    doc.doc_id,
                    len(valid_nodes),
                    strategy.value,
                )

            except Exception as exc:
                logging.error(f"Failed to chunk document: {exc}")
                continue

        self.stats["documents_processed"] += len(documents)

        if all_nodes:
            total_size = sum(len(n.text) for n in all_nodes)
            self.stats["avg_chunk_size"] = total_size / len(all_nodes)

        logger.info(f"Processed {len(documents)} docs into "
                    f"{len(all_nodes)} chunks")
        return all_nodes



    def get_stats(self) -> Dict[str, Any]:
        """
        Retrieve chunking performance statistics.

        Returns:
            Dictionary containing processing stats and config.
        """
        stats_copy = self.stats.copy()
        stats_copy["config"] = {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "strategy": self.strategy.value,
            "min_chunk_size": self.min_chunk_size,
            "max_chunk_size": self.max_chunk_size,
        }
        return stats_copy



    def reset_stats(self) -> None:
        """
        Reset performance statistics to initial state.
        """
        self.stats = {
            "documents_processed": 0,
            "total_chunks": 0,
            "avg_chunk_size": 0.0,
            "strategy_usage": {
                strat.value: 0 for strat in ChunkingStrategy
            },
        }
        logger.info("Chunking statistics reset")



    def __repr__(self) -> str:
        """
        Human-readable representation of TextChunker instance.

        Returns:
            String representation including size, overlap, and strategy.
        """
        return (
            f"TextChunker(size={self.chunk_size}, "
            f"overlap={self.chunk_overlap}, "
            f"strategy={self.strategy.value})"
        )