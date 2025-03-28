def separate_by_token_length(sequences, buckets=[510, 700, 900, 1100]):
    """
    Separates sequences into buckets based on token length.

    Parameters:
    - sequences: List of tokenized sequences (each a list of tokens or a string if already tokenized).
    - buckets: List of max token lengths to separate into.

    Returns:
    - A dictionary with bucket max length as keys and lists of sequences as values.
    """
    separated = {bucket: [] for bucket in buckets}
    for seq in sequences:
        length = len(seq)
        for bucket in buckets:
            if length <= bucket:
                separated[bucket].append(seq)
                break  # Put in the first matching bucket only
    return separated
