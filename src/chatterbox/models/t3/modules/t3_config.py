from ..llama_configs import LLAMA_CONFIGS


class T3Config:
    """Class representing configuration settings for T3 models. Includes parameters for tokenization and model architecture details."""
    def __init__(self, text_tokens_dict_size=704):
        """Initializes a configuration class for handling text and speech tokens.
        Args:
        text_tokens_dict_size (int): The size of the dictionary for text tokens. Default is 704.
        Returns:
        None
        """
        self.start_text_token = 255
        self.stop_text_token = 0
        self.text_tokens_dict_size = text_tokens_dict_size
        self.max_text_tokens = 2048

        self.start_speech_token = 6561
        self.stop_speech_token = 6562
        self.speech_tokens_dict_size = 8194
        self.max_speech_tokens = 4096

        self.llama_config_name = "Llama_520M"
        self.input_pos_emb = "learned"
        self.speech_cond_prompt_len = 150

        self.encoder_type = "voice_encoder"
        self.speaker_embed_size = 256
        self.use_perceiver_resampler = True
        self.emotion_adv = True

    @property
    def n_channels(self):
        """Retrieves the number of channels in the current configuration.
        Returns:
        The number of hidden units as specified in the configuration dictionary.
        ---
        Indicates whether the text token dictionary size is 2454, suggesting a multilingual model.
        Args:
        None
        Returns:
        True if the text token dictionary size is 2454, False otherwise.
        ---
        Creates a configuration for an English-only TTS model.
        Args:
        text_tokens_dict_size (int): The size of the text tokens dictionary, defaults to 704.
        Returns:
        A new instance of the class with the specified text tokens dictionary size.
        ---
        Creates a configuration for a multilingual TTS model.
        """
        return LLAMA_CONFIGS[self.llama_config_name]["hidden_size"]
    
    @property
    def is_multilingual(self):
        """```python
        Determines if the TTS model is configured for multilingual speech.
        Args:
        None
        Returns:
        bool: True if the model supports multiple languages, False otherwise.
        ```
        """
        return self.text_tokens_dict_size == 2454

    @classmethod
    def english_only(cls):
        """Create configuration for English-only TTS model."""
        return cls(text_tokens_dict_size=704)
    
    @classmethod 
    def multilingual(cls):
        """Create configuration for multilingual TTS model."""
        return cls(text_tokens_dict_size=2454)
