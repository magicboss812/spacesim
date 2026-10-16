# PyInstaller-bauplan fuer spacesim -- `pyinstaller spacesim.spec` aus der
# repo-wurzel, auf dem ZIELSYSTEM (PyInstaller baut nicht ueber kreuz).
#
# Ergebnis in dist/:
#   Windows/Linux: SpaceSim/SpaceSim(.exe) + SpaceSim/config/ (aenderbar)
#   macOS:         SpaceSim.app; die config legt die app beim ersten start in
#                  ~/Library/Application Support/spacesim/ an (runtime/paths.py)
#
# `config/` wird zusaetzlich als vorlage IN die app gepackt: fehlt eine datei
# am aenderbaren ort, kopiert `prepare_frozen()` sie von dort.
import shutil
import sys
from pathlib import Path

ROOT = Path(SPECPATH)

a = Analysis(
    [str(ROOT / 'main.py')],
    pathex=[str(ROOT)],
    datas=[
        (str(ROOT / 'render' / 'gl'), 'render/gl'),
        (str(ROOT / 'ui' / 'assets'), 'ui/assets'),
        (str(ROOT / 'config' / 'config.json'), 'config'),
        (str(ROOT / 'config' / 'solar_system.json'), 'config'),
    ],
    # Nur von docs/-skripten gebraucht, nie vom spiel.
    excludes=['astropy', 'poliastro', 'matplotlib', 'tkinter', 'IPython',
              'pytest', 'scipy'],
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SpaceSim',
    console=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name='SpaceSim')

if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name='SpaceSim.app',
        bundle_identifier='org.spacesim.app',
        info_plist={
            'NSHighResolutionCapable': True,
            'LSMinimumSystemVersion': '11.0',
        },
    )
else:
    # Die aenderbare config neben die exe -- dort sucht runtime/paths.py.
    dst = Path(DISTPATH) / 'SpaceSim' / 'config'
    dst.mkdir(parents=True, exist_ok=True)
    for name in ('config.json', 'solar_system.json'):
        shutil.copy2(ROOT / 'config' / name, dst / name)
