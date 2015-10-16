import warnings
from nose.tools import assert_equal, assert_in, assert_raises

from expyfun import ExperimentController
from expyfun.io import read_tab
from expyfun._utils import _TempDir, _hide_window

warnings.simplefilter('always')

temp_dir = _TempDir()
std_args = ['test']  # experiment name
std_kwargs = dict(output_dir=temp_dir, full_screen=False, window_size=(1, 1),
                  participant='foo', session='01', stim_db=0.0, noise_db=0.0,
                  verbose=True, version='dev')


@_hide_window
def test_parse():
    """Test .tab parsing
    """
    with ExperimentController(*std_args, stim_fs=44100, **std_kwargs) as ec:
        ec.identify_trial(ec_id='one', ttl_id=[0])
        ec.start_stimulus()
        ec.write_data_line('misc', 'trial one')
        ec.stop()
        ec.trial_ok()
        ec.write_data_line('misc', 'between trials')
        ec.identify_trial(ec_id='two', ttl_id=[1])
        ec.start_stimulus()
        ec.write_data_line('misc', 'trial two')
        ec.stop()
        ec.trial_ok()
        ec.write_data_line('misc', 'end of experiment')

    assert_raises(ValueError, read_tab, ec.data_fname, group_start='foo')
    assert_raises(ValueError, read_tab, ec.data_fname, group_end='foo')
    assert_raises(ValueError, read_tab, ec.data_fname, group_end='trial_id')
    assert_raises(RuntimeError, read_tab, ec.data_fname, group_end='misc')
    data = read_tab(ec.data_fname)
    keys = list(data[0].keys())
    assert_equal(len(keys), 6)
    for key in ['trial_id', 'flip', 'play', 'stop', 'misc', 'trial_ok']:
        assert_in(key, keys)
    assert_equal(len(data[0]['misc']), 1)
    assert_equal(len(data[1]['misc']), 1)
    data = read_tab(ec.data_fname, group_end=None)
    assert_equal(len(data[0]['misc']), 2)  # includes between-trials stuff
    assert_equal(len(data[1]['misc']), 2)
