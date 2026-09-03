"""Fenster, GL-context und der frame-takt.

Stand als erste 70 zeilen von `main()` in `test.py`. Alles hier ist einmalige
einrichtung, die mit der simulation nichts zu tun hat -- deshalb liegt sie
jetzt vor der tuer statt im startskript.
"""
import os

import moderngl
import pygame
from pygame.locals import DOUBLEBUF, OPENGL, RESIZABLE


class Window:
    """Das SDL/OpenGL-fenster und die uhr, die den frame-takt vorgibt."""

    def __init__(self, config):
        # VSync ueber Umgebungsvariable aktivieren
        self.vsync = bool(config.get('window.vsync', True))
        os.environ['SDL_VIDEO_VSYNC'] = '1' if self.vsync else '0'

        # Windows-DPI-Awareness VOR pygame.init() setzen. python.exe ist ohne
        # Manifest standardmaessig DPI-unaware; ohne diesen Hint skaliert der
        # DWM das fertige (bereits scharf gerenderte) Fenster per
        # Bitmap-Stretch auf die physische Aufloesung hoch, sobald die
        # Windows-Skalierung > 100% ist -> das ganze Fenster inkl. HUD-Text
        # wirkt unscharf. SDL_WINDOWS_DPI_AWARENESS ist der von SDL2
        # unterstuetzte Hint dafuer und muss vor SDL_Init/pygame.init gesetzt
        # sein.
        if os.name == 'nt':
            os.environ.setdefault('SDL_WINDOWS_DPI_AWARENESS', 'permonitorv2')

        # Starte Pygame mit OpenGL.
        #
        # NUR display und font -- NICHT pygame.init(). pygame.init() faehrt
        # JEDES untermodul hoch, auch mixer und joystick, und beide zaehlen
        # dabei die geraete des rechners auf. Auf diesem system kostet das
        # gemessen 25.2 s (mixer) + 20.1 s (joystick) = 45.3 s, in denen das
        # fenster noch gar nicht existiert -- der start wirkt schlicht wie ein
        # absturz. Die dauer haengt an audio-/HID-treibern, nicht am spiel: sie
        # kann sich jederzeit wieder aendern. Deshalb wird hier gar nicht erst
        # geraten, sondern nur initialisiert, was das spiel wirklich benutzt.
        # Verwendet werden ausschliesslich display, event, font, image, key,
        # mouse und time; von denen brauchen nur display und font ein init.
        pygame.display.init()
        pygame.font.init()

        self.width = int(config.get('window.width', 800))
        self.height = int(config.get('window.height', 800))

        # OpenGL-Flag fuer pygame Display; moderngl haengt sich an den von
        # pygame/SDL erstellten GL-context (ein wrapper, geteilt mit dem
        # Renderer). RESIZABLE: die aufloesung ist dynamisch -- das fenster
        # darf frei skaliert oder maximiert werden, viewport/FXAA-targets/
        # UI-skala folgen ueber den WINDOWSIZECHANGED-handler in der
        # hauptschleife.
        flags = DOUBLEBUF | OPENGL
        if bool(config.get('window.resizable', True)):
            flags |= RESIZABLE
        self.screen = pygame.display.set_mode(
            (self.width, self.height), flags, vsync=1 if self.vsync else 0
        )
        self.ctx = moderngl.create_context()
        info = self.ctx.info
        print(info['GL_VENDOR'], info['GL_RENDERER'], info['GL_VERSION'])
        pygame.display.set_caption(
            str(config.get('window.caption', "Orbital Mechanics - OpenGL Renderer")))

        self.clock = pygame.time.Clock()
        self.fps = int(config.get('window.fps', 180))

    def tick(self):
        """Einen frame abwarten und das ECHTE delta in sekunden liefern."""
        return self.clock.tick(self.fps) / 1000.0

    def close(self):
        pygame.quit()
