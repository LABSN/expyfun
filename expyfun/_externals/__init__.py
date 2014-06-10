# -*- coding: utf-8 -*-

from .decorator import decorator  # noqa
# do NOT import `ndarraysource` here, since it could impact pyglet.media
# driver priority (which currently resides in _sound_controllers.py)

