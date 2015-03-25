#import librosa

#def audio_read(filename, sr=11025, channels=1):
#    """Read a soundfile, return (d, sr)."""
#    d, sr = librosa.load(filename, sr=sr)
#    return d, sr


import os
import numpy as np

def audio_read(filename, sr=11025, channels=1):
    """Read a soundfile, return (d, sr)."""
    #d, sr = librosa.load(filename, sr=sr)
    #return d, sr
    # Hacked version of librosa.load and audioread/ff.
    mono = True
    offset = 0.0
    duration = None
    dtype = np.float32
    y = []
    with FFmpegAudioFile(os.path.realpath(filename), sample_rate=sr, channels=channels) as input_file:
        sr_native = sr

        s_start = int(np.floor(sr_native * offset) * channels)

        if duration is None:
            s_end = np.inf
        else:
            s_end = s_start + int(np.ceil(sr_native * duration)
                                  * channels)

        n = 0

        for frame in input_file:
            frame = buf_to_float(frame, dtype=dtype)
            n_prev = n
            n = n + len(frame)

            if n < s_start:
                # offset is after the current frame
                # keep reading
                continue

            if s_end < n_prev:
                # we're off the end.  stop reading
                break

            if s_end < n:
                # the end is in this frame.  crop.
                frame = frame[:s_end - n_prev]

            if n_prev <= s_start < n:
                # beginning is in this frame
                frame = frame[(s_start - n_prev):]

            # tack on the current frame
            y.append(frame)

        if not len(y):
            # Zero-length read
            y = np.zeros(0, dtype=dtype)

        else:
            y = np.concatenate(y)

            if channels > 1:
                y = y.reshape((-1, 2)).T

            sr = sr_native

    # Final cleanup for dtype and contiguity
    y = np.ascontiguousarray(y, dtype=dtype)

    return (y, sr)


def buf_to_float(x, n_bytes=2, dtype=np.float32):
    """Convert an integer buffer to floating point values.
    This is primarily useful when loading integer-valued wav data
    into numpy arrays.
    .. seealso:: :func:`librosa.util.buf_to_float`
    :parameters:
        - x : np.ndarray [dtype=int]
            The integer-valued data buffer
        - n_bytes : int [1, 2, 4]
            The number of bytes per sample in ``x``
        - dtype : numeric type
            The target output type (default: 32-bit float)
    :return:
        - x_float : np.ndarray [dtype=float]
            The input data buffer cast to floating point
    """

    # Invert the scale of the data
    scale = 1./float(1 << ((8 * n_bytes) - 1))

    # Construct the format string
    fmt = '<i{:d}'.format(n_bytes)

    # Rescale and format the data buffer
    return scale * np.frombuffer(x, fmt).astype(dtype)


import subprocess
import re
import threading
import time
try:
    import queue
except ImportError:
    import Queue as queue



class QueueReaderThread(threading.Thread):
    """A thread that consumes data from a filehandle and sends the data
    over a Queue.
    """
    def __init__(self, fh, blocksize=1024, discard=False):
        super(QueueReaderThread, self).__init__()
        self.fh = fh
        self.blocksize = blocksize
        self.daemon = True
        self.discard = discard
        self.queue = None if discard else queue.Queue()

    def run(self):
        while True:
            data = self.fh.read(self.blocksize)
            if not self.discard:
                self.queue.put(data)
            if not data:
                # Stream closed (EOF).
                break



class FFmpegAudioFile(object):
    """An audio file decoded by the ffmpeg command-line utility."""
    def __init__(self, filename, channels=1, sample_rate=11025, block_size=4096):
        self.proc = subprocess.Popen(
            ['ffmpeg', '-i', filename, '-f', 's16le', '-ac', str(channels), '-ar', str(sample_rate), '-'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        # Start another thread to consume the standard output of the
        # process, which contains raw audio data.
        self.stdout_reader = QueueReaderThread(self.proc.stdout, block_size)
        self.stdout_reader.start()

        # Read relevant information from stderr.
        self._get_info()

        # Start a separate thread to read the rest of the data from
        # stderr. This (a) avoids filling up the OS buffer and (b)
        # collects the error output for diagnosis.
        self.stderr_reader = QueueReaderThread(self.proc.stderr)
        self.stderr_reader.start()

    def read_data(self, timeout=10.0):
        """Read blocks of raw PCM data from the file."""
        # Read from stdout in a separate thread and consume data from
        # the queue.
        start_time = time.time()
        while True:
            # Wait for data to be available or a timeout.
            data = None
            try:
                data = self.stdout_reader.queue.get(timeout=timeout)
                if data:
                    yield data
                else:
                    # End of file.
                    break
            except queue.Empty:
                # Queue read timed out.
                end_time = time.time()
                if not data:
                    if end_time - start_time >= timeout:
                        # Nothing interesting has happened for a while --
                        # FFmpeg is probably hanging.
                        raise ValueError('ffmpeg output: {}'.format(
                            ''.join(self.stderr_reader.queue.queue)
                        ))
                    else:
                        start_time = end_time
                        # Keep waiting.
                        continue

    def _get_info(self):
        """Reads the tool's output from its stderr stream, extracts the
        relevant information, and parses it.
        """
        out_parts = []
        while True:
            line = self.proc.stderr.readline()
            if not line:
                # EOF and data not found.
                raise ValueError("stream info not found")

            # In Python 3, result of reading from stderr is bytes.
            if isinstance(line, bytes):
                line = line.decode('utf8', 'ignore')

            line = line.strip().lower()

            if 'no such file' in line:
                raise IOError('file not found')
            elif 'invalid data found' in line:
                raise UnsupportedError()
            elif 'duration:' in line:
                out_parts.append(line)
            elif 'audio:' in line:
                out_parts.append(line)
                self._parse_info(''.join(out_parts))
                break

    def _parse_info(self, s):
        """Given relevant data from the ffmpeg output, set audio
        parameter fields on this object.
        """
        # Sample rate.
        match = re.search(r'(\d+) hz', s)
        if match:
            self.samplerate = int(match.group(1))
        else:
            self.samplerate = 0

        # Channel count.
        match = re.search(r'hz, ([^,]+),', s)
        if match:
            mode = match.group(1)
            if mode == 'stereo':
                self.channels = 2
            else:
                match = re.match(r'(\d+) ', mode)
                if match:
                    self.channels = int(match.group(1))
                else:
                    self.channels = 1
        else:
            self.channels = 0

        # Duration.
        match = re.search(
            r'duration: (\d+):(\d+):(\d+).(\d)', s
        )
        if match:
            durparts = list(map(int, match.groups()))
            duration = (
                durparts[0] * 60 * 60 +
                durparts[1] * 60 +
                durparts[2] +
                float(durparts[3]) / 10
            )
            self.duration = duration
        else:
            # No duration found.
            self.duration = 0

    def close(self):
        """Close the ffmpeg process used to perform the decoding."""
        # Kill the process if it is still running.
        if hasattr(self, 'proc') and self.proc.returncode is None:
            self.proc.kill()
            self.proc.wait()

    def __del__(self):
        self.close()

    # Iteration.
    def __iter__(self):
        return self.read_data()

    # Context manager.
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
