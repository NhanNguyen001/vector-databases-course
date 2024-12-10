def split_text_with_visualization(text, chunk_size=10, chunk_overlap=3):
    chunks = []
    start = 0

    print(f"\nOriginal text: '{text}'")
    print(f"Chunk size: {chunk_size}")
    print(f"Overlap size: {chunk_overlap}\n")

    while start < len(text):
        end = start + chunk_size
        if end > len(text):
            end = len(text)

        chunk = text[start:end]
        chunks.append(chunk)

        # Visualize the chunk and its position
        print(f"Chunk {len(chunks)}:")
        print(f"Position: {start:2d} to {end:2d}")
        print(f"Content: '{chunk}'")
        if start > 0:
            print(f"Overlap: '{text[start:start+chunk_overlap]}'")
        print()

        start = end - chunk_overlap

    return chunks


# Example 1: Small overlap
print("Example 1: Small overlap")
text = "The quick brown fox jumps over the lazy dog"
chunks = split_text_with_visualization(text, chunk_size=10, chunk_overlap=3)

# Example 2: Larger overlap
print("\nExample 2: Larger overlap")
chunks = split_text_with_visualization(text, chunk_size=10, chunk_overlap=5)

# Example 3: No overlap
print("\nExample 3: No overlap")
chunks = split_text_with_visualization(text, chunk_size=10, chunk_overlap=0)
