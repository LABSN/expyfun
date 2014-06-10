# -*- coding: utf-8 -*-

try:
    from pyglet.media import NdarraySource
except ImportError:
    from pyglet.media import StaticMemorySource, AudioFormat

    class NdarraySource(StaticMemorySource):
        """Play sound from numpy array

        :Parameters:
            `data` : ndarray
                float data with shape n_channels x n_samples. If ``data`` is
                1D, then the sound is assumed to be mono. Note that data
                will be clipped between +/- 1.
            `fs` : int
                Sample rate for the data.
        """
        def __init__(self, data, fs):
            fs = int(fs)
            if data.ndim not in (1, 2):
                raise ValueError('Data must have one or two dimensions')
            n_ch = data.shape[0] if data.ndim == 2 else 1
            data = data.T.ravel('C')
            data[data < -1] = -1
            data[data > 1] = 1
            data = (data * (2 ** 15)).astype('int16').tostring()
            audio_format = AudioFormat(channels=n_ch, sample_size=16,
                                       sample_rate=fs)
            super(NdarraySource, self).__init__(data, audio_format)

        def _get_queue_source(self):
            return self
