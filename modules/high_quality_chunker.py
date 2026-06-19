# High Quality Chunker Module
# Implements paragraph-based chunking for improved audiobook quality
# Activated when CHUNKING_QUALITY = "High"

from config.config import MAX_CHUNK_WORDS, MIN_CHUNK_WORDS
import re

def paragraph_chunk_text(text):
    """
    High-quality paragraph-based text chunking.

    Splits text at paragraph boundaries (double newlines) and breaks long
    paragraphs at word boundaries to maintain optimal TTS chunk sizes.

    Returns: List of (chunk_text, boundary_type) tuples
    """
    chunks = []

    # Split into paragraphs by double newlines
    paragraphs = text.split('\n\n')

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        # Detect chapter headers
        para_lower = paragraph.lower().strip()
        is_chapter = (
            any(word in para_lower for word in ['chapter', 'section', 'part', 'prologue', 'epilogue']) and
            len(paragraph.split()) <= 10
        )

        if is_chapter:
            # Chapter headers are single chunks with chapter_end boundary
            chunks.append((paragraph, "chapter_end"))
            continue

        # Process regular paragraphs
        words = paragraph.split()
        if len(words) <= MAX_CHUNK_WORDS:
            # Paragraph fits, single chunk
            chunks.append((paragraph, "paragraph_end"))
        else:
            # Break long paragraph at word boundaries
            start = 0
            while start < len(words):
                end = min(start + MAX_CHUNK_WORDS, len(words))
                chunk_words = words[start:end]
                chunk_text = ' '.join(chunk_words)

                # Last chunk of paragraph gets paragraph_end, others get paragraph_continue
                if end == len(words):
                    boundary = "paragraph_end"
                else:
                    boundary = "paragraph_continue"

                chunks.append((chunk_text, boundary))
                start = end

    return chunks