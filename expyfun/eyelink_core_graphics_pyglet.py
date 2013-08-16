try:
    import pylink
except ImportError:
    pylink = None
import pyglet

import ctypes
import math
import sys
import array
from psychopy import logging as psylog

from pyglet.gl import (gl, glBegin, glColor4f, glBindTexture, glClearColor,
                       glDeleteTextures, glDisable, glEnable, glEnd,
                       glGenTextures, glLoadIdentity, glMatrixMode,
                       gluNewQuadric, glPopMatrix, glPushMatrix,
                       glTexCoord2i, glTexEnvi, glTexImage2D,
                       glTexParameteri, glTranslatef, gluDisk, gluOrtho2D,
                       gluQuadricDrawStyle, glVertex2f,
                       GL_LINE_LOOP, GL_LINEAR, GL_LINES,
                       GL_MODELVIEW, GL_PROJECTION, GL_QUADS, GL_TEXTURE_ENV,
                       GL_TEXTURE_ENV_MODE,
                       GL_REPLACE, GL_TEXTURE_MIN_FILTER,
                       GL_TEXTURE_MAG_FILTER, GL_TEXTURE_RECTANGLE_ARB,
                       GL_RGBA, GL_RGBA8, GL_UNSIGNED_BYTE, GLU_FILL)

key_trans_list = [[pyglet.window.key.F1, pylink.F1_KEY],
                  [pyglet.window.key.F2, pylink.F2_KEY],
                  [pyglet.window.key.F3, pylink.F3_KEY],
                  [pyglet.window.key.F4, pylink.F4_KEY],
                  [pyglet.window.key.F5, pylink.F5_KEY],
                  [pyglet.window.key.F6, pylink.F6_KEY],
                  [pyglet.window.key.F7, pylink.F7_KEY],
                  [pyglet.window.key.F8, pylink.F8_KEY],
                  [pyglet.window.key.F9, pylink.F9_KEY],
                  [pyglet.window.key.F10, pylink.F10_KEY],
                  [pyglet.window.key.PAGEUP, pylink.PAGE_UP],
                  [pyglet.window.key.PAGEDOWN, pylink.PAGE_DOWN],
                  [pyglet.window.key.UP, pylink.CURS_UP],
                  [pyglet.window.key.DOWN, pylink.CURS_DOWN],
                  [pyglet.window.key.LEFT, pylink.CURS_LEFT],
                  [pyglet.window.key.RIGHT, pylink.CURS_RIGHT],
                  [pyglet.window.key.BACKSPACE, '\b'],
                  [pyglet.window.key.RETURN, pylink.ENTER_KEY],
                  [pyglet.window.key.ESCAPE, pylink.ESC_KEY],
                  ]


class CalibratePyglet(pylink.EyeLinkCustomDisplay):
    def __init__(self, window):
        self.size = (window.width, window.height)
        self.window = window
        self.keys = []

        if sys.byteorder == 'little':
            self.byteorder = 1
        else:
            self.byteorder = 0

        pylink.EyeLinkCustomDisplay.__init__(self)

        self.__target_beep__ = None
        self.__target_beep__done__ = None
        self.__target_beep__error__ = None

        self.imagebuffer = array.array('I')
        self.pal = None

        self.width = window.width
        self.height = window.height

        self.pos = []
        self.state = 0

    def setup_event_handlers(self):
        self.label = pyglet.text.Label('Eye Label', font_size=20,
                                       color=(255, 255, 0, 255),
                                       anchor_x='center', anchor_y='center')
        #glPushMatrix() #enable this if needed
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluOrtho2D(0, self.width, 0, self.height)
        glClearColor(0.0, 0.0, 0.0, 1.0)

        glEnable(GL_TEXTURE_RECTANGLE_ARB)
        self.texid = gl.GLuint()
        glGenTextures(1, ctypes.byref(self.texid))

        def on_key_press(symbol, modifiers):
            psylog.debug('Keypress: {0}'.format(symbol))
            psylog.flush()
            self.translate_key_message((symbol, modifiers))

        def on_mouse_press(x, y, button, modifiers):
            psylog.debug('Mouse: ({0}, {1})'.format(x, y))
            psylog.flush()
            self.state = 1

        def on_mouse_motion(x, y, dx, dy):
            self.pos = (x, y)

        def on_mouse_release(x, y, button, modifiers):
            self.state = 0

        def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
            self.pos = (x, y)

        self.window.push_handlers(on_key_press, on_mouse_press,
                                  on_mouse_motion, on_mouse_release,
                                  on_mouse_drag)

    def release_event_handlers(self):
        self.window.pop_handlers()
        #glClearColor(1, 1, 1, 1)
        glDeleteTextures(1, ctypes.byref(self.texid))
        #glPopMatrix() #enable this if needed

    def setup_cal_display(self):
        self.window.clear()
        self.window.flip()

    def exit_cal_display(self):
        self.clear_cal_display()

    def record_abort_hide(self):
        pass

    def clear_cal_display(self):
        self.window.clear()
        self.window.flip()

    def erase_cal_target(self):
        self.window.clear()
        self.window.flip()

    def draw_cal_target(self, x, y):
        self.window.clear()
        #self.render(x,y)
        glColor4f(1.0, 0.0, 0.0, 1.0)
        glPushMatrix()
        glTranslatef(x, y, 0)
        q = gluNewQuadric()
        gluQuadricDrawStyle(q, GLU_FILL)
        gluDisk(q, 3.0, 12.0, 60, 1)
        glPopMatrix()
        self.window.flip()

    def render(self, x, y):
        glColor4f(1.0, 0.0, 0.0, 1.0)
        glPushMatrix()
        glTranslatef(x, y, 0)
        q = gluNewQuadric()
        gluQuadricDrawStyle(q, GLU_FILL)
        gluDisk(q, 3.0, 12.0, 60, 1)
        glPopMatrix()

    def play_beep(self, eepid):
        """Play a sound during calibration/drift correct."""

    def translate_key_message(self, event):
        key = 0
        mod = 0
        if len(event) > 0:
            key = None
            for trans in key_trans_list:
                if event[0] == trans[0]:
                    key = trans[1]
                    break
            key = event[0] if key is None else key

        if len(event) > 0:
            self.keys.append(pylink.KeyInput(key, mod))

        return key

    def get_input_key(self):
        self.window.dispatch_events()
        if len(self.keys) > 0:
            k = self.keys
            self.keys = []
            return k
        else:
            return None

    def get_mouse_state(self):
        if len(self.pos) > 0:
            l = (int)(self.width * 0.5 - self.width * 0.5 * 0.75)
            r = (int)(self.width * 0.5 + self.width * 0.5 * 0.75)
            b = (int)(self.height * 0.5 - self.height * 0.5 * 0.75)
            t = (int)(self.height * 0.5 + self.height * 0.5 * 0.75)
            mx, my = 0, 0
            if self.pos[0] < l:
                mx = l
            elif self.pos[0] > r:
                mx = r
            else:
                mx = self.pos[0]

            if self.pos[1] < b:
                my = b
            elif self.pos[1] > t:
                my = t
            else:
                my = self.pos[1]

            mx = (int)((mx - l) * self.img_size[0] // (r - l))
            my = self.img_size[1] - (int)((my - b) * self.img_size[1]
                                          // (t - b))
            return ((mx, my), self.state)
        else:
            return((0, 0), 0)

    def exit_image_display(self):
        self.window.clear()
        self.window.flip()

    def alert_printf(self, msg):
        print "alert_printf"

    def setup_image_display(self, width, height):
        self.img_size = (width, height)
        self.window.clear()

    def image_title(self, text):
        self.label.text = text

    def draw_image_line(self, width, line, totlines, buff):
        i = 0
        while i < width:
            if buff[i] >= len(self.pal):
                buff[i] = len(self.pal) - 1
            self.imagebuffer.append(self.pal[buff[i] & 0x000000FF])
            i += 1
        if line == totlines:
            #asp = ((float)(self.size[1]))/((float)(self.size[0]))
            asp = 1
            r = (float)(self.width * 0.5 - self.width * 0.5 * 0.75)
            l = (float)(self.width * 0.5 + self.width * 0.5 * 0.75)
            t = (float)(self.height * 0.5 + self.height * 0.5 * asp * 0.75)
            b = (float)(self.height * 0.5 - self.height * 0.5 * asp * 0.75)

            self.window.clear()
            self.label.x = (int)(self.width * 0.5)
            self.label.y = b - 30
            self.label.draw()
            self.draw_cross_hair()
            glEnable(GL_TEXTURE_RECTANGLE_ARB)
            glBindTexture(GL_TEXTURE_RECTANGLE_ARB, self.texid.value)
            glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER,
                            GL_LINEAR)
            glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER,
                            GL_LINEAR)
            glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE)
            glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_RGBA8, width,
                         totlines, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                         self.imagebuffer.tostring())

            glBegin(GL_QUADS)
            glTexCoord2i(0, 0)
            glVertex2f(r, t)
            glTexCoord2i(0, self.img_size[1])
            glVertex2f(r, b)
            glTexCoord2i(self.img_size[0], self.img_size[1])
            glVertex2f(l, b)
            glTexCoord2i(self.img_size[1], 0)
            glVertex2f(l, t)
            glEnd()
            glDisable(GL_TEXTURE_RECTANGLE_ARB)
            self.draw_cross_hair()
            self.window.flip()
            self.imagebuffer = array.array('I')

    def _colorindex_to_color(colorindex):
        if colorindex == pylink.CR_HAIR_COLOR:
            color = (1.0, 1.0, 1.0, 1.0)
        elif colorindex == pylink.PUPIL_HAIR_COLOR:
            color = (1.0, 1.0, 1.0, 1.0)
        elif colorindex == pylink.PUPIL_BOX_COLOR:
            color = (0.0, 1.0, 0.0, 1.0)
        elif colorindex == pylink.SEARCH_LIMIT_BOX_COLOR:
            color = (1.0, 0.0, 0.0, 1.0)
        elif colorindex == pylink.MOUSE_CURSOR_COLOR:
            color = (1.0, 0.0, 0.0, 1.0)
        else:
            color = (0.0, 0.0, 0.0, 0.0)
        return color

    def draw_line(self, x1, y1, x2, y2, colorindex):
        color = self._colorindex_to_color(colorindex)
        #asp = ((float)(self.size[1]))/((float)(self.size[0]))
        asp = 1
        x11, x22, y11, y22 = self._get_rltb_xy(asp, x1, x2, y1, y2)
        glBegin(GL_LINES)
        glColor4f(color[0], color[1], color[2], color[3])
        glVertex2f(x11, y11)
        glVertex2f(x22, y22)
        glEnd()

    def _get_rltb(self, asp, x1, x2, y1, y2):
        r = (float)(self.width * 0.5 - self.width * 0.5 * 0.75)
        l = (float)(self.width * 0.5 + self.width * 0.5 * 0.75)
        t = (float)(self.height * 0.5 + self.height * 0.5 * asp * 0.75)
        b = (float)(self.height * 0.5 - self.height * 0.5 * asp * 0.75)
        x11 = float(float(x1) * (l - r) / float(self.img_size[0]) + r)
        x22 = float(float(x2) * (l - r) / float(self.img_size[0]) + r)
        y11 = float(float(y1) * (b - t) / float(self.img_size[1]) + t)
        y22 = float(float(y2) * (b - t) / float(self.img_size[1]) + t)
        return (x11, x22, y11, y22)

    def draw_lozenge(self, x, y, width, height, colorindex):
        color = self._colorindex_to_color(colorindex)
        width = int((float(width) / float(self.img_size[0]))
                    * self.img_size[0])
        height = int((float(height) / float(self.img_size[1]))
                     * self.img_size[1])
        #asp = ((float)(self.size[1]))/((float)(self.size[0]))
        asp = 1
        r, l, t, b = self._get_rltb(asp, x, x + width, y, y + height)
        glColor4f(color[0], color[1], color[2], color[3])

        xw = math.fabs(float(l - r))
        yw = math.fabs(float(b - t))
        sh = min(xw, yw)
        rad = float(sh * 0.5)
        x = float(min(l, r) + rad)
        y = float(min(t, b) + rad)

        if xw == sh:
            st = 180
        else:
            st = 90
        glBegin(GL_LINE_LOOP)
        degInRad = (float)(float(st) * (3.14159 / 180.0))
        for i in range(st, st + 180):
            degInRad = (float)(float(i) * (3.14159 / 180.0))
            glVertex2f((float)(float(x) + math.cos(degInRad) * rad),
                       float(y) + (float)(math.sin(degInRad) * rad))
        if xw == sh:  # short horizontally
            y = (float)(max(t, b) - rad)
        else:  # short vertically
            x = (float)(max(l, r) - rad)

        for i in range(st + 180, st + 360):
            degInRad = (float)(float(i) * (3.14159 / 180.0))
            glVertex2f((float)(float(x) + math.cos(degInRad) * rad),
                       float(y) + (float)(math.sin(degInRad) * rad))
        glEnd()

    def set_image_palette(self, r, g, b):
        self.imagebuffer = array.array('I')
        self.clear_cal_display()
        sz = len(r)
        i = 0
        self.pal = []
        while i < sz:
            rf = int(r[i])
            gf = int(g[i])
            bf = int(b[i])
            if self.byteorder:
                self.pal.append(0xff << 24 | (bf << 16) | (gf << 8) | (rf))
            else:
                self.pal.append((rf << 24) | (gf << 16) | (bf << 8) | 0xff)
            i += 1
