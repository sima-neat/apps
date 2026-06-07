import re

def _split_sentences(text):
    """Split text into sentences based on punctuation."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s]

def update_buffer_and_detect_repetition(chunk, buffer_text, seen_sentences):
    """
    Update buffer, check for sentence repetition.

    Returns:
        - is_repetitive (bool)
        - new_sentences (list)
        - updated buffer_text (str)
    """
    buffer_text += chunk
    is_repetitive = False
    new_sentences = []

    if any(p in buffer_text for p in ['.', '?', '!']):
        sentences = _split_sentences(buffer_text)

        for sent in sentences[:-1]:  # Completed sentences
            cleaned_sent = sent.strip()
            if cleaned_sent:
                if cleaned_sent in seen_sentences:
                    is_repetitive = True
                    break
                else:
                    new_sentences.append(cleaned_sent)

        # Update buffer with last partial sentence
        buffer_text = sentences[-1] if sentences else ''

    return is_repetitive, new_sentences, buffer_text