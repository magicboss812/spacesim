Funktionen, Bug-Fixes, UI und UX Erweiterungen sind hier aufgelistet. Die werden mithilfe von `claude` in der Implementierung hinzugefügt.

☑Fertig
❌Unfinished

| **Typ**   | **Änderung**                      | **Beschreibung**                                                                                                                                                 | Fertig |
| --------- | --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ |
| Funktion  | Hilfspunkte (Apoapsis, Periapsis) | **Apoapsis**: planetenfernster Punkt, **Periapsis**: planetennächsten Punkt, für Orbit/Deorbiting                                                                | ☑      |
| Funktion  | Rotationsrichtung/-vektoren       | Normale, Anti-normale, rechtläufig oder rückläufig, wichtige Navigation für Orbiting                                                                             | ☑      |
| Bug-Fix   | Integrator-Switch                 | RKN hat leichten Drift, nötig wäre ein Vergleich zweier Näherungsverfahren für long-term Stability                                                               | ❌      |
| UI/UX     | OpenGL Wrapper Wechsel            | von pyopengl auf moderngl wechseln, performanter für UI                                                                                                          | ☑      |
| UI        | Spiel-HUD                         | Settings, Geschwindigkeit, Reference-Frame-Selector, Live-Debug View (Vektoren, Position, usw.)                                                                  | ❌      |
| Funktion  | Settings Konfigurations-Datei     | .json config Datei wird importiert, beinhaltet alle zentralen Parameter aus dem Code (wenn das Project kompiliert wird, kann trotzdem Parameter geändert werden) | ❌      |
| UX-UI     | Anleitung-Menü                    | Grafiken, Bilder, Spiel-Screenshots, um neue Spieler zu lehren; muss einfache Erklärungen beinhalten                                                             | ❌      |
| Funktion  | Nah-Ansicht, 2. View              | Zweite Ansicht/Zweites Rendering für Landungen u.a.                                                                                                              | ❌      |
| Funktion  | Landung                           | velocity clamping wenn das Schiff Distanz 0 zu einem Körper hat, vergewissert kein glitching des Schiffs in den Boden sowie einfache collision Berechnung        | ❌      |
| Render    | Custom Render Engine              | für z.B. flüssiges Zoomen (gerade stufenweise), bessere Shaders/Grafik, Menüs (HTML & CSS wären sehr hilfreich)                                                  | ❌      |
| Bug-Fixes | Velocity-Vektor                   | Velocity Vektor wird vom reference Frame nicht transformiert                                                                                                     | ❌      |
