"""
==============================================
Display text with different formatting methods
==============================================

This example demonstrates differences between the Text and AttrText classes.
"""
# Author: Dan McCloy <drmccloy@uw.edu>
#
# License: BSD (3-clause)

print(__doc__)

from expyfun import ExperimentController, analyze
from expyfun.visual import _convert_color

# Colors
blue = _convert_color('#00CEE9')
pink = _convert_color('#FF97AF')
white = (255, 255, 255, 255)
# Text
message_one_a = ('This text can only have a single color, font, and size for '
                 'the whole sentence, because it is an instance of the Text() '
                 'class.')
message_one_b = ('Additional calls to ec.screen_text() can have different '
                 'formatting, but have to be manually positioned.')
message_one_c = 'Press any key to continue.'
# AttrText()
message_two = ('This text can have {{color {0}}}different {{color {1}}}colors '
               'speci{{color {2}}}fied inline, because it\'s an {{color {0}}}'
               'instance of the {{color {1}}}AttrText() class. {{color {2}}}'
               'Specifying different typefaces or sizes inline is buggy and '
               'not recommended.'
               '\n\n\nPress any key to quit.').format(blue, pink, white)

with ExperimentController('textDemo', participant='foo', session='001',
                          output_dir=None) as ec:
    ec.wait_secs(0.1)
    ec.screen_text(message_one_a, pos=[0, 0.5])
    ec.screen_text(message_one_b, pos=[0, 0.2], font_name='Times New Roman',
                   font_size=32, color='#00CEE9')
    ec.screen_text(message_two, pos=[0, -0.2], attributed=True)
    screenshot = ec.screenshot()  # only because we want to show it in the docs
    ec.flip()
    ec.wait_one_press()

import matplotlib.pyplot as plt
plt.ion()
analyze.plot_screen(screenshot)
