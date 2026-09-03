"""Der laufzeit-unterbau: fenster, aufbau, hauptschleife, eingabe.

    window.py         pygame display/font, GL-context, DPI, vsync, titel
    gl_device.py      GLDeviceMixin -- FXAA-ziele, resize, present
    bootstrap.py      baut welt/kamera/predictor/renderer/UI zusammen
    loop.py           die hauptschleife und die TIMING-ausgabe
    input.py          tastenbelegung und die klick-geste
    system_loader.py  koerper aus `config/solar_system.json`

Einstieg ist `main.py` im wurzelverzeichnis.
"""
