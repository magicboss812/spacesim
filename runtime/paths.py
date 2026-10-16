"""Wo die einstellbaren dateien liegen -- im repo und in der gebauten app.

Im repo ist das die repo-wurzel (`config/`, `screenshots/` daneben), wie
bisher. In einer PyInstaller-app (`sys.frozen`) zeigt `__file__` dagegen in
den mitgelieferten, schreibgeschuetzten teil der app; `config/` muss dort
liegen, wo der spieler sie findet und aendern kann:

  * Windows/Linux: neben der exe (`SpaceSim/config/config.json`);
  * macOS: `~/Library/Application Support/spacesim/`. NICHT neben der `.app`:
    eine heruntergeladene app startet Gatekeeper aus einem zufaelligen,
    schreibgeschuetzten ordner (app translocation), und eine datei IN der
    `.app` zu aendern bricht deren signatur.

Fehlt dort eine datei, kopiert `prepare_frozen()` die mitgelieferte vorlage
hinein; eine vorhandene wird nie ueberschrieben.

Dieses modul importiert nichts schweres; `main.py` ruft `prepare_frozen()`
vor allem anderen auf, damit auch ein absturz beim import im log landet.
"""
import shutil
import sys
import traceback
from pathlib import Path

FROZEN = bool(getattr(sys, 'frozen', False))
_REPO_ROOT = Path(__file__).resolve().parent.parent


def bundle_dir() -> Path:
    """Der mitgelieferte, schreibgeschuetzte teil (vorlagen, shader, fonts)."""
    if FROZEN:
        return Path(getattr(sys, '_MEIPASS', Path(sys.executable).parent))
    return _REPO_ROOT


def user_dir() -> Path:
    """Wurzel fuer `config/` und `screenshots/`: das, was der spieler aendert."""
    if not FROZEN:
        return _REPO_ROOT
    if sys.platform == 'darwin':
        return Path.home() / 'Library' / 'Application Support' / 'spacesim'
    return Path(sys.executable).resolve().parent


def config_dir() -> Path:
    return user_dir() / 'config'


def _excepthook(exc_type, exc, tb):
    """Eine fenster-app (`--windowed`) hat keine konsole: ohne diese datei
    verschwaende ein absturz spurlos."""
    text = ''.join(traceback.format_exception(exc_type, exc, tb))
    try:
        user_dir().mkdir(parents=True, exist_ok=True)
        (user_dir() / 'crash.log').write_text(text, encoding='utf-8')
    except OSError:
        pass
    if sys.__stderr__ is not None:
        sys.__stderr__.write(text)


def prepare_frozen():
    """In der gebauten app: config-vorlagen anlegen, abstuerze protokollieren.
    Im repo ein no-op.

    Der numba-cache braucht nichts: in einer gebauten app nimmt numba von
    selbst seinen benutzerweiten cache (`UserWideCacheLocator`, Windows
    `%LOCALAPPDATA%\\numba`, macOS `~/Library/Caches/numba`). Gemessen im
    Linux-build: erster start 54 s, zweiter 17 s."""
    if not FROZEN:
        return
    sys.excepthook = _excepthook
    target = config_dir()
    target.mkdir(parents=True, exist_ok=True)
    for name in ('config.json', 'solar_system.json'):
        dst = target / name
        src = bundle_dir() / 'config' / name
        if not dst.exists() and src.exists():
            shutil.copy2(src, dst)
