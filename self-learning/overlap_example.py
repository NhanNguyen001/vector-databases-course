def show_text_splitting(text, chunk_size, chunk_overlap):
    print(f"\nSplitting text with chunk_size={chunk_size}, overlap={chunk_overlap}")
    print(f"Original text: '{text}'\n")

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]

        # Show the current chunk with overlap highlighted
        display = list("." * len(text))
        # Mark the chunk
        for i in range(start, end):
            display[i] = "^"
        # Mark the overlap
        if start > 0:
            for i in range(start, min(start + chunk_overlap, end)):
                display[i] = "O"

        print(f"Chunk: '{chunk}'")
        print(f"Text:  {text}")
        print(f"       {''.join(display)}")
        print(
            f"       {''.join([' ' if i < start else str(i % 10) for i in range(len(text))])}"
        )
        print()

        chunks.append(chunk)
        start = end - chunk_overlap
        if end == len(text):
            break

    return chunks


# Example 1: Small text with words that might be split
text = "machine learning"
print("Example 1: Splitting 'machine learning'")
print("With overlap=0 (problematic - splits words):")
chunks = show_text_splitting(text, chunk_size=7, chunk_overlap=0)

print("\nWith overlap=3 (better - keeps word context):")
chunks = show_text_splitting(text, chunk_size=7, chunk_overlap=3)

# Example 2: Sentence with important phrase
text = "The quick brown fox jumps"
print("\nExample 2: Splitting sentence with important phrase 'brown fox'")
print("With overlap=1 (might split phrases):")
chunks = show_text_splitting(text, chunk_size=10, chunk_overlap=1)

print("\nWith overlap=3 (keeps phrases together):")
chunks = show_text_splitting(text, chunk_size=10, chunk_overlap=3)
