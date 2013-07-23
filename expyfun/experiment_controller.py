import logging
from functools import partial

logger = logging.getLogger('expyfun')


class ExperimentController(object):
    def __init__(self, sound_type=None):
        """Interface for hardware control

        Parameters
        ----------
        sound_type : str | None
            Can be 'psychopy' or a TDT model (e.g., 'RM1' or 'RP2'). If None,
            the type will be read from the machine preferences file.

        Returns
        -------
        exp_controller : instance of ExperimentController
            The experiment control interface.

        Notes
        -----
        Blah blah blah.
        """
        # some things
        self._fp_function = None
        self._fs = 24414

    def load_buffer(self, data):
        """XXX ADD DOCSTRING
        """
        logger.info('Loading buffers with %s samples of data' % data.size)
        # XXX

    def clear_buffer(self):
        """XXX ADD DOCSTRING
        """
        logger.info('Clearing buffers')
        # XXX

    def _stop(self):
        # XXX
        logger.debug('stopping')

    def _reset(self):
        # XXX
        logger.debug('reset')

    def stop_reset(self):
        """XXX ADD DOCSTRING
        """
        logger.info('Stopping and resetting')
        self._stop()
        self._reset()

    def close(self):
        """XXX ADD DOCSTRING
        """
        self.__exit__()

    def __exit__(self, type, value, traceback):
        # XXX stop the TDT circuit, etc.  (for use with "with" syntax)
        logger.debug('exiting cleanly')

    def __enter__(self):
        # wrap to init?
        # XXX for with statement (for use with "with" syntax)
        logger.debug('entering')
        return self

    def flip_and_play(self):
        # XXX flip screen
        # XXX play buffer
        logger.info('Flipping and playing audio')
        if self._fp_function is not None:
            self._fp_function()

    def call_on_flip_and_play(self, function, *args, **kwargs):
        if function is not None:
            self._fp_function = partial(function, *args, **kwargs)
        else:
            self._fp_function = None

    @property
    def fs(self):
        # do it this way so people can't set it
        return self._fs
