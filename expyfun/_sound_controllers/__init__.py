from ._sound_controller import (SoundCardController, SoundPlayer, _BACKENDS,
                                _import_backend)

_AUTO_BACKENDS = ('auto',) + _BACKENDS
_SOUND_CARD_ACS = tuple({'TYPE': 'sound_card', 'SOUND_CARD_BACKEND': backend}
                        for backend in _AUTO_BACKENDS)
