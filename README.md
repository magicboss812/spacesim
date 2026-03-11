# Stand 11.03.2026
Dies ist der heutige Stand des Spiels. Das meiste wurde bereits im Logbuch dokumentiert und erklärt, sowie mit der Lehrkraft evaluiert

Wie lässt es sich starten?
- letzte Version von Python
- mit pip ausführen: pip install pygame pyopengl pyopengl-accelerate

### bodies.py
Erstellt eine Liste mit den Objekten "body" mit vorbestimmten Attributen, sowie Keplerbahnen.
### camera.py
Stellt die Camera Bewegung und Berechnung bereit. Aktualisiert ebenfalls Steuerungsinputs und wandelt Weltkoordinaten in Bildschirmkoordinaten um. Wohlmöglich kann es auch als Spieler bezeichnet werden, falls weitere Steuerungen dort abgelagert werden
### loader.py
Ladet alle Attributen aus einer json Datei in ein neu erstelltes Objekt "body".
### main.py
Endgültige Startdatei zum Starten und Initialisiern von Konfigurationen, Systemen und dem Fenster. Zurzeit leer
### predictor.py
Vorausberechnung der Umlaufbahn des Raumschiffs. Zurzeit nur des Raumschiffs, Keplerbahnen werden noch nicht abgebildet. Berechnet mit der gleichen Physik die Position des Raumschiffs in der Zukunft basierend auf Beschleunigung und Geschwindigkeit
### predictor_mp.py
Eine Variante des Predictors, jedoch mit einer Multiprocessing Erweiterung, die mehr CPU-Kerne nutzen soll, um so die Leistung erheblich zu steigern. Wird im aktuellen Zustand von keiner Datei benutzt
### rendering.py
OpenGL Rendering mit Pygame integration zur Fenstererstellung. Kann glow und Atmosphären rendern, verfügt über Anti-Aliasing (FXAA), um zu scharfe pixelartige Kanten weicher zu machen
### schiff.py
Schiffsteurung, zurzeit nicht benutzt
### solar_system.json
Daten von Planeten aus einem Sternensystem. Voll änderbar und theoretisch modding Support.
### test.py
Debug-Datei, um das Spiel in einem Debug-Interface zu testen (entsprechend dokumentiert, kann flexibel geändert werden)
### vec.py
Vektoroperatoren, können später zu C geändert werden, um die Leistung deutlich zu steigern (Logbuch-Eintrag)
### world.py
Erstellung der Weltbühne und Berechnung von Physik. Newtons Gleichungen
