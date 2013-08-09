from .experiment_controller import ExperimentController


class EyelinkController(ExperimentController):
    def __init__(self, *args, **kwargs):
        super(EyelinkController, self).__init__(*args, **kwargs)

    def flip_and_play(self):
        super(EyelinkController, self).flip_and_play()
        # XXX stamp EL immediately after flip
