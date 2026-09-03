"""spacesim -- 2D-N-Koerper-Bahnmechanik mit spielbarem raumschiff.

DAS IST DER EINSTIEGSPUNKT.  `python main.py`

Er hiess bis zur umstrukturierung `test.py` -- eine datei, die kein test war,
neben einem ordner `tests/`, der welche enthaelt. `main.py` war derweil ein
zehnzeiliger stummel. Die beiden sind jetzt getauscht.

Was hier steht, ist absichtlich nur die abfolge. Der aufbau liegt in
`runtime/bootstrap.py`, die schleife in `runtime/loop.py`.

Umgebungsvariablen: SPACESIM_CONFIG (alternative konfiguration),
SPACESIM_MAX_FRAMES (nach n frames beenden -- fuer messlaeufe),
SPACESIM_PREDICTOR_ASYNC.
"""
from runtime.bootstrap import build_app, load_config
from runtime.loop import run


def main():
    config = load_config()
    app = build_app(config)
    run(app)


if __name__ == "__main__":
    main()
