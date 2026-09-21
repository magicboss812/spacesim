"""spacesim -- 2D-N-Koerper-Bahnmechanik mit spielbarem raumschiff.

DAS IST DER EINSTIEGSPUNKT.  `python main.py`

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
