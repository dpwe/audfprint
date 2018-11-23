"""Provide stft to avoid librosa dependency. 

This implementation is based on routines from 
https://github.com/tensorflow/models/blob/master/research/audioset/mel_features.py
"""

from __future__ import division

import numpy as np


def frame(data, window_length, hop_length):
  """Convert array into a sequence of successive possibly overlapping frames.

  An n-dimensional array of shape (num_samples, ...) is converted into an
  (n+1)-D array of shape (num_frames, window_length, ...), where each frame
  starts hop_length points after the preceding one.

  This is accomplished using stride_tricks, so the original data is not
  copied.  However, there is no zero-padding, so any incomplete frames at the
  end are not included.

  Args:
    data: np.array of dimension N >= 1.
    window_length: Number of samples in each frame.
    hop_length: Advance (in samples) between each window.

  Returns:
    (N+1)-D np.array with as many rows as there are complete frames that can be
    extracted.
  """
  num_samples = data.shape[0]
  num_frames = 1 + ((num_samples - window_length) // hop_length)
  shape = (num_frames, window_length) + data.shape[1:]
  strides = (data.strides[0] * hop_length,) + data.strides
  return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)


def periodic_hann(window_length):
  """Calculate a "periodic" Hann window.

  The classic Hann window is defined as a raised cosine that starts and
  ends on zero, and where every value appears twice, except the middle
  point for an odd-length window.  Matlab calls this a "symmetric" window
  and np.hanning() returns it.  However, for Fourier analysis, this
  actually represents just over one cycle of a period N-1 cosine, and
  thus is not compactly expressed on a length-N Fourier basis.  Instead,
  it's better to use a raised cosine that ends just before the final
  zero value - i.e. a complete cycle of a period-N cosine.  Matlab
  calls this a "periodic" window. This routine calculates it.

  Args:
    window_length: The number of points in the returned window.

  Returns:
    A 1D np.array containing the periodic hann window.
  """
  return 0.5 - (0.5 * np.cos(2 * np.pi / window_length *
                             np.arange(window_length)))


def stft(signal, n_fft, hop_length=None, window=None):
  """Calculate the short-time Fourier transform.

  Args:
    signal: 1D np.array of the input time-domain signal.
    n_fft: Size of the FFT to apply.
    hop_length: Advance (in samples) between each frame passed to FFT. Defaults
      to half the window length.
    window: Length of each block of samples to pass to FFT, or vector of window
      values.  Defaults to n_fft.

  Returns:
    2D np.array where each column contains the complex values of the 
    fft_length/2+1 unique values of the FFT for the corresponding frame of 
    input samples ("spectrogram transposition").
  """
  if window is None:
    window = n_fft
  if isinstance(window, (int, float)):
    # window holds the window length, need to make the actual window.
    window = periodic_hann(int(window))
  window_length = len(window)
  if not hop_length:
    hop_length = window_length // 2
  # Default librosa STFT behavior.
  pad_mode = 'reflect'
  signal = np.pad(signal, (n_fft // 2), mode=pad_mode)
  frames = frame(signal, window_length, hop_length)
  # Apply frame window to each frame. We use a periodic Hann (cosine of period
  # window_length) instead of the symmetric Hann of np.hanning (period
  # window_length-1).
  windowed_frames = frames * window
  return np.fft.rfft(windowed_frames, n_fft).transpose()
