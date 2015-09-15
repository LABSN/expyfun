import warnings
import numpy as np
from nose.tools import assert_raises, assert_equal

from expyfun import ExperimentController, visual
from expyfun._utils import _hide_window, requires_opengl21

warnings.simplefilter('always')

std_kwargs = dict(output_dir=None, full_screen=False, window_size=(1, 1),
                  participant='foo', session='01', stim_db=0.0, noise_db=0.0,
                  verbose=True)


@_hide_window
@requires_opengl21
def test_visuals():
    """Test EC visual methods
    """
    with ExperimentController('test', **std_kwargs) as ec:
        assert_raises(TypeError, visual.Circle, ec, n_edges=3.5)
        assert_raises(ValueError, visual.Circle, ec, n_edges=3)
        circ = visual.Circle(ec)
        circ.draw()
        assert_raises(ValueError, circ.set_radius, [1, 2, 3])
        assert_raises(ValueError, circ.set_pos, [1])
        assert_raises(ValueError, visual.Triangle, ec, [5, 6])
        tri = visual.Triangle(ec, [[-1, 0, 1], [-1, 1, -1]], units='deg',
                              line_width=1.0)
        tri.draw()
        rect = visual.Rectangle(ec, [0, 0, 1, 1], line_width=1.0)
        rect.draw()
        diamond = visual.Diamond(ec, [0, 0, 1, 1], line_width=1.0)
        diamond.draw()
        assert_raises(TypeError, visual.ConcentricCircles, ec, colors=dict())
        assert_raises(TypeError, visual.ConcentricCircles, ec,
                      colors=np.array([]))
        assert_raises(ValueError, visual.ConcentricCircles, ec, radii=[[1]])
        assert_raises(ValueError, visual.ConcentricCircles, ec, radii=[1])
        fix = visual.ConcentricCircles(ec, radii=[1, 2, 3],
                                       colors=['w', 'k', 'y'])
        fix.set_pos([0.5, 0.5])
        fix.set_radius(0.1, 1)
        fix.set_radii([0.1, 0.2, 0.3])
        fix.set_color('w', 1)
        fix.set_colors(['w', 'k', 'k'])
        fix.set_colors(('w', 'k', 'k'))
        assert_raises(IndexError, fix.set_color, 'w', 3)
        assert_raises(ValueError, fix.set_colors, ['w', 'k'])
        assert_raises(ValueError, fix.set_colors, np.array(['w', 'k', 'k']))
        assert_raises(IndexError, fix.set_radius, 0.1, 3)
        assert_raises(ValueError, fix.set_radii, [0.1, 0.2])
        fix.draw()
        fix_2 = visual.FixationDot(ec)
        fix_2.draw()
        assert_raises(ValueError, rect.set_pos, [0, 1, 2])
        img = visual.RawImage(ec, np.ones((3, 3, 4)))
        print(img.bounds)  # test bounds
        assert_equal(img.scale, 1)
        img.draw()
        line = visual.Line(ec, [[0, 1], [1, 0]])
        line.draw()
        assert_raises(ValueError, line.set_line_width, 100)
        line.set_line_width(2)
        line.draw()
        assert_raises(ValueError, line.set_coords, [0])
        line.set_coords([0, 1])
        ec.set_background_color('black')
        text = visual.Text(ec, 'Hello {color (255 0 0 255)}Everybody!',
                           pos=[0, 0], color=[1, 1, 1], wrap=False)
        text.draw()
        text.set_color(None)
        text.draw()
        text = visual.Text(ec, 'Thank you, come again.', pos=[0, 0],
                           color='white', attr=False)
        text.draw()
        text.set_color('red')
        text.draw()
