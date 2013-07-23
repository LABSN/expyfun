from expyfun import ExperimentController


def test_experiment_init():
    """Test initialization of experiment
    """
    ec = ExperimentController()

def test_with_support():
    """Test experiment 'with' statement support
    """
    with ExperimentController() as ec:
        print ec


