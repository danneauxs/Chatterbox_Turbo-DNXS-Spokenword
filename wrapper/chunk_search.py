def search_chunks(chunks, query):
    """Searches through a list of chunks for those containing a specified query.
    Args: chunks (list): A list of dictionaries, each with a 'text' key.
    query (str): The text to search for within the chunks.
    Returns: list: A list of chunks that contain the query text.
    """
    results = []
    query_lower = query.lower()

    for chunk in chunks:
        if query_lower in chunk['text'].lower():
            results.append(chunk)

    return results
