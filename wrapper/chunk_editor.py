def update_chunk(chunk, boundary_type=None, pause_duration=None, sentiment_score=None):
    """Updates a chunk dictionary with optional boundary type, pause duration, and sentiment score.
    Args:
    boundary_type (str): The type of boundary to set.
    pause_duration (float): The duration of the pause.
    sentiment_score (float): The sentiment score of the chunk.
    Returns:
    dict: The updated chunk dictionary.
    """
    if boundary_type is not None:
        chunk['boundary_type'] = boundary_type
    if pause_duration is not None:
        chunk['pause_duration'] = pause_duration
    if sentiment_score is not None:
        chunk['sentiment_score'] = sentiment_score
    return chunk
