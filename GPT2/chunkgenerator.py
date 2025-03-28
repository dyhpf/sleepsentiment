from transformers import AutoTokenizer

def split_text_by_token_lengths(text, tokenizer, chunk_sizes=[510, 700, 900, 1100]):
    """
    Splits a long text into chunks with given token lengths.

    Args:
        text (str): The input long text.
        tokenizer: A HuggingFace tokenizer.
        chunk_sizes (list): List of token sizes to split by in order.

    Returns:
        List of decoded text chunks.
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    chunks = []
    idx = 0
    for size in chunk_sizes:
        if idx + size <= len(tokens):
            chunk = tokens[idx:idx + size]
            chunks.append(tokenizer.decode(chunk, skip_special_tokens=True))
            idx += size
        else:
            break

    if idx < len(tokens):
        final_chunk = tokens[idx:]
        chunks.append(tokenizer.decode(final_chunk, skip_special_tokens=True))
    
    return chunks


def main():
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")

    # Example long text (repeat to simulate long content)
    long_text = "Dies ist ein sehr langer Text, der viele Informationen enthält. " * 1000

    # Split the text into chunks
    chunks = split_text_by_token_lengths(long_text, tokenizer)

    # Display results
    print(f"Total chunks created: {len(chunks)}\n")
    for i, chunk in enumerate(chunks):
        token_count = len(tokenizer.encode(chunk, add_special_tokens=False))
        print(f"Chunk {i+1}: {token_count} tokens")


if __name__ == "__main__":
    main()
