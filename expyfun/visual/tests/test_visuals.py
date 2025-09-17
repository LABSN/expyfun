import numpy as np
import pytest
from numpy.testing import assert_equal

from expyfun import ExperimentController, fetch_data_file, visual
from expyfun._utils import requires_opengl21, requires_video

std_kwargs = dict(
    output_dir=None,
    full_screen=False,
    window_size=(1, 1),
    participant="foo",
    session="01",
    stim_db=0.0,
    noise_db=0.0,
    verbose=True,
    version="dev",
)


@requires_opengl21
def test_visuals(hide_window):
    """Test EC visual methods."""
    with ExperimentController("test", **std_kwargs) as ec:
        pytest.raises(TypeError, visual.Circle, ec, n_edges=3.5)
        pytest.raises(ValueError, visual.Circle, ec, n_edges=3)
        circ = visual.Circle(ec)
        circ.draw()
        pytest.raises(ValueError, circ.set_radius, [1, 2, 3])
        pytest.raises(ValueError, circ.set_pos, [1])
        pytest.raises(ValueError, visual.Triangle, ec, [5, 6])
        tri = visual.Triangle(
            ec, [[-1, 0, 1], [-1, 1, -1]], units="deg", line_width=1.0
        )
        tri.draw()
        rect = visual.Rectangle(ec, [0, 0, 1, 1], line_width=1.0)
        rect.draw()
        diamond = visual.Diamond(ec, [0, 0, 1, 1], line_width=1.0)
        diamond.draw()
        pytest.raises(TypeError, visual.ConcentricCircles, ec, colors=dict())
        pytest.raises(TypeError, visual.ConcentricCircles, ec, colors=np.array([]))
        pytest.raises(ValueError, visual.ConcentricCircles, ec, radii=[[1]])
        pytest.raises(ValueError, visual.ConcentricCircles, ec, radii=[1])
        fix = visual.ConcentricCircles(ec, radii=[1, 2, 3], colors=["w", "k", "y"])
        fix.set_pos([0.5, 0.5])
        fix.set_radius(0.1, 1)
        fix.set_radii([0.1, 0.2, 0.3])
        fix.set_color("w", 1)
        fix.set_colors(["w", "k", "k"])
        fix.set_colors(("w", "k", "k"))
        pytest.raises(IndexError, fix.set_color, "w", 3)
        pytest.raises(ValueError, fix.set_colors, ["w", "k"])
        pytest.raises(ValueError, fix.set_colors, np.array(["w", "k", "k"]))
        pytest.raises(IndexError, fix.set_radius, 0.1, 3)
        pytest.raises(ValueError, fix.set_radii, [0.1, 0.2])
        fix.draw()
        fix_2 = visual.FixationDot(ec)
        fix_2.draw()
        pytest.raises(ValueError, rect.set_pos, [0, 1, 2])
        for shape in ((3, 3, 3), (3, 3, 4), (3, 3), (3,), (3,) * 4):
            data = np.ones(shape)
            if len(shape) not in (2, 3):
                pytest.raises(RuntimeError, visual.RawImage, ec, data)
            else:
                img = visual.RawImage(ec, data)
            print(img.bounds)  # test bounds
            assert_equal(img.scale, 1)
            # test get_rect
            imgrect = visual.Rectangle(ec, img.get_rect())
            assert_equal(
                imgrect._points["fill"][(0, 2, 0, 1), (0, 0, 1, 1)], img.bounds
            )
            img.draw()
        line = visual.Line(ec, [[0, 1], [1, 0]])
        line.draw()
        pytest.raises(ValueError, line.set_line_width, 100)
        line.set_line_width(2)
        line.draw()
        pytest.raises(ValueError, line.set_coords, [0])
        line.set_coords([0, 1])
        ec.set_background_color("black")
        text = visual.Text(
            ec,
            "Hello {color (255, 0, 0, 255)}Everybody!",
            pos=[0, 0],
            color=[1, 1, 1],
            wrap=False,
        )
        text.draw()
        text.set_color(None)
        text.draw()
        text = visual.Text(
            ec, "Thank you, come again.", pos=[0, 0], color="white", attr=False
        )
        text.draw()
        text.set_color("red")
        text.draw()
        bar = visual.ProgressBar(ec, [0, 0, 1, 0.2])
        bar = visual.ProgressBar(ec, [0, 0, 1, 1], units="pix")
        bar.update_bar(0.5)
        bar.draw()
        pytest.raises(ValueError, visual.ProgressBar, ec, [0, 0, 1, 0.1], units="deg")
        pytest.raises(ValueError, visual.ProgressBar, ec, [0, 0, 1, 0.1], colors=["w"])
        pytest.raises(ValueError, bar.update_bar, 500)


@requires_video()
def test_video(hide_window, monkeypatch):
    """Test EC video methods."""
    std_kwargs.update(dict(window_size=(640, 480)))
    video_path = fetch_data_file("video/example-video.mp4")
    with ExperimentController("test", **std_kwargs) as ec:
        ec.load_video(video_path)
        ec.video.play()
        pytest.raises(ValueError, ec.video.set_pos, [1, 2, 3])
        pytest.raises(ValueError, ec.video.set_scale, "foo")
        pytest.raises(ValueError, ec.video.set_scale, -1)
        ec.wait_secs(0.1)
        ec.video.set_visible(False)
        ec.wait_secs(0.1)
        ec.video.set_visible(True)
        ec.video.set_scale("fill")
        ec.video.set_scale("fit")
        ec.video.set_scale("0.5")
        ec.video.set_pos(pos=(0.1, 0), units="norm")
        ec.video.pause()
        ec.video.draw()
        ec.delete_video()
    # test ec.video.play(audio=True)
    with monkeypatch.context() as m:
        m.delenv("SOUND_CARD_NAME", raising=False)
        m.delenv("SOUND_CARD_API", raising=False)
        with ExperimentController(
            "test",
            audio_controller=dict(TYPE="sound_card", SOUND_CARD_BACKEND="pyglet"),
            **std_kwargs,
        ) as ec:
            ec.load_video(video_path)
            ec.video.play(audio=True)
            ec.wait_secs(0.1)
            ec.video.pause()
            ec.delete_video()
