from functools import lru_cache

from scipy import signal
import numpy as np
import librosa


@lru_cache()
def mel_basis(hp):
    """### mel_basis(hp)
    Computes the Mel filterbank.
    Args:
    - hp (dict): Hyperparameters containing sample_rate, n_fft, num_mels, fmin, and fmax.
    Returns:
    - ndarray: Mel filterbank matrix of shape (nmel, nfreq).
    ### preemphasis(wav, hp)
    Applies preemphasis to a waveform.
    Args:
    - wav (ndarray): Input audio waveform.
    - hp (dict): Hyperparameters containing preemphasis coefficient.
    Returns:
    - ndarray: Preemphasized audio waveform.
    """
    assert hp.fmax <= hp.sample_rate // 2
    return librosa.filters.mel(
        sr=hp.sample_rate,
        n_fft=hp.n_fft,
        n_mels=hp.num_mels,
        fmin=hp.fmin,
        fmax=hp.fmax)  # -> (nmel, nfreq)


def preemphasis(wav, hp):
    """Applies preemphasis to the input waveform.
    Args:
    wav (np.ndarray): Input audio waveform.
    hp (Hparams): Hyperparameters including preemphasis factor.
    Returns:
    np.ndarray: Preemphasized audio waveform.
    Note: The function assumes hp.preemphasis is not zero and applies a linear filter to the waveform.
    """
    assert hp.preemphasis != 0
    wav = signal.lfilter([1, -hp.preemphasis], [1], wav)
    wav = np.clip(wav, -1, 1)
    return wav


def melspectrogram(wav, hp, pad=True):
    """Calculates the Mel-spectrogram of a given audio waveform.
    Args:
    wav (np.ndarray): Input audio waveform.
    hp (dict): Hyperparameters including preemphasis, STFT settings, and Mel power.
    pad (bool, optional): Whether to pad the input waveform before processing. Defaults to True.
    Returns:
    np.ndarray: Mel-spectrogram of the audio waveform.
    """
    # Run through pre-emphasis
    if hp.preemphasis > 0:
        wav = preemphasis(wav, hp)
        assert np.abs(wav).max() - 1 < 1e-07

    # Do the stft
    spec_complex = _stft(wav, hp, pad=pad)

    # Get the magnitudes
    spec_magnitudes = np.abs(spec_complex)

    if hp.mel_power != 1.0:
        spec_magnitudes **= hp.mel_power

    # Get the mel and convert magnitudes->db
    mel = np.dot(mel_basis(hp), spec_magnitudes)
    if hp.mel_type == "db":
        mel = _amp_to_db(mel, hp)

    # Normalise the mel from db to 0,1
    if hp.normalized_mels:
        mel = _normalize(mel, hp).astype(np.float32)

    assert not pad or mel.shape[1] == 1 + len(wav) // hp.hop_size   # Sanity check
    return mel   # (M, T)


def _stft(y, hp, pad=True):
    """Computes the Short-Time Fourier Transform (STFT) of an audio signal.
    Args:
    y (numpy.ndarray): The input audio signal.
    hp (dict): Dictionary containing hyperparameters including n_fft, hop_size, and win_size.
    pad (bool, optional): Whether to pad the signal before computing STFT. Defaults to True.
    Returns:
    numpy.ndarray: The computed STFT of the audio signal.
    """
    # NOTE: after 0.8, pad mode defaults to constant, setting this to reflect for
    #   historical consistency and streaming-version consistency
    return librosa.stft(
        y,
        n_fft=hp.n_fft,
        hop_length=hp.hop_size,
        win_length=hp.win_size,
        center=pad,
        pad_mode="reflect",
    )


def _amp_to_db(x, hp):
    """Convert amplitude to decibels.
    Args:
    x: Amplitude value(s).
    hp: Hyperparameters containing magnitude minimum.
    Returns:
    Decibel representation of input amplitudes.
    Convert decibels back to amplitude.
    Args:
    x: Decibel value(s).
    Returns:
    Amplitude representation of input decibels.
    Normalize signal by converting it to a range between 0 and 1.
    Args:
    s: Signal values.
    hp: Hyperparameters containing magnitude minimum.
    headroom_db: Headroom in decibels for normalization.
    Returns:
    Normalized signal.
    """
    return 20 * np.log10(np.maximum(hp.stft_magnitude_min, x))


def _db_to_amp(x):
    """Converts a value from decibels to amplitude.
    Args:
    x (float): The decibel value.
    Returns:
    float: The corresponding amplitude value.
    Normalizes an input signal based on the given hyperparameters and headroom in decibels.
    Args:
    s (numpy.ndarray): The input signal.
    hp (object): Hyperparameter object containing stft_magnitude_min.
    headroom_db (int, optional): Headroom in decibels. Defaults to 15.
    Returns:
    numpy.ndarray: The normalized signal.
    """
    return np.power(10.0, x * 0.05)


def _normalize(s, hp, headroom_db=15):
    """Normalizes a signal `s` based on its minimum level and a headroom in decibels.
    Args:
    s (numpy.ndarray): Input signal to normalize.
    hp (object): Hyperparameters containing the minimum STFT magnitude.
    headroom_db (float, optional): Headroom in decibels. Default is 15.
    Returns:
    numpy.ndarray: Normalized signal.
    """
    min_level_db = 20 * np.log10(hp.stft_magnitude_min)
    s = (s - min_level_db) / (-min_level_db + headroom_db)
    return s
