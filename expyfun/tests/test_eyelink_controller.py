from expyfun import EyelinkController
from expyfun.utils import _TempDir

std_args = ['test']
temp_dir = _TempDir()
std_kwargs = dict(output_dir=temp_dir, full_screen=False, window_size=(1, 1),
                  participant='foo', session='01')


def test_eyelink_init():
    """Test eyelink overridden and new experiment methods
    """
    ec = EyelinkController(*std_args, **std_kwargs)
    ec.flip_and_play()
    ec.close()
