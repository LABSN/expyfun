from expyfun import ExperimentController

with pyexpfun.ExperimentController() as ec:
    # XXX make some frames, get some audio read in
    ec.call_on_flip_and_play(print, ‘Flipping now’)
    ec.load_buffer(audio)
    ec.flip_and_play()

