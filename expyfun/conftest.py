# -*- coding: utf-8 -*-
# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import os
import pytest
from expyfun._sound_controllers import _AUTO_BACKENDS


@pytest.mark.timeout(0)  # importing plt will build font cache, slow on Azure
@pytest.fixture(scope='session')
def matplotlib_config():
    """Configure matplotlib for viz tests."""
    import matplotlib
    matplotlib.use('agg')  # don't pop up windows
    import matplotlib.pyplot as plt
    assert plt.get_backend() == 'agg'
    # overwrite some params that can horribly slow down tests that
    # users might have changed locally (but should not otherwise affect
    # functionality)
    plt.ioff()
    plt.rcParams['figure.dpi'] = 100
    os.environ['_EXPYFUN_WIN_INVISIBLE'] = 'true'


@pytest.fixture(scope='function')
def hide_window():
    """Hide the expyfun window."""
    import pyglet
    try:
        pyglet.window.get_platform().get_default_display()
    except Exception as exp:
        pytest.skip('Pyglet windowing unavailable (%s)' % exp)


_SOUND_CARD_ACS = tuple({'TYPE': 'sound_card', 'SOUND_CARD_BACKEND': backend}
                        for backend in _AUTO_BACKENDS)
for val in _SOUND_CARD_ACS:
    if val['SOUND_CARD_BACKEND'] == 'pyglet':
        val.update(SOUND_CARD_API=None, SOUND_CARD_NAME=None,
                   SOUND_CARD_FIXED_DELAY=None)


@pytest.fixture(scope="module", params=('tdt',) + _SOUND_CARD_ACS)
def ac(request):
    """Get the backend name."""
    yield request.param
