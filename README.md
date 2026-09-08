# Projekt Stand 08.09.2026
![spiel](https://i.imgur.com/fqPf2Da.png)
## Spiel starten
**Python-Version**: Python 3.13

**Python Vorbereitung**: `pip install pygame moderngl numba numpy imgui-bundle`, benötigte Packages für den Spielcode
![pip-install](https://imgur.com/JliDcew.gif)

**Spiel Starten**: `python main.py`  (aus dem ordner `spacesim/` heraus)
![spiel-starten](https://i.imgur.com/kVPjbHY.gif)

## Neuerungen bis zum letzten Commit:
- Ordnerstruktur überarbeitet & in jeweilige Spielbereiche eingeteilt (siehe Kapitel "Repo-Struktur)
- unabhängige Konfigurationsdatei von nötigen Parametern aus dem Spielcode (default Werte stehen trotzdem im Code)
- imgui Developer/Debugging Benutzeroberfläche mit aktiver Änderung einige Parameter und Einstellungen im Spiel (Aufrufbar über Taste `F1`)
- benutzerdefiniertes User Interface ("benutzerdefinierte Benutzeroberfläche") mit 2 Schriftarten, inspiriertes Design des "Nav-Ball"'s aus `Kerbal Space Program 2`, Referenz-Körper Menü, Orbitvorhersage-Slider, System-Karte, Orbitaldaten von Schiff und Planeten, Richtungs-Einrasterung, individuelle Einstellung des Schubs & Maneuver Tool
- eigenes Planetendesign aus Vektorgrafiken, prozedurale Generierung von Designs für alle Körper, korrekte veranschaulichte Skalierung der Planeten
- bewegbare Kamera mit unabhängiger Steuerung in jedem Referenz-System
- sichtbare Planeten Orbits + individuelle Planetenvorhersage mit Radius-Kappe, welches mit Vorhersage des Schiffes verknüpft ist (Endkappe von Schiffvorhersage beeinflusst Endkappe von Planetenvorhersage; gleiche Zeit = unterschiedliche Standorte)
- überarbeitetes Schiffdesign, ebenfalls Vektorgrafik & Leuchteffekt aus dem Triebwerk
- relative Abminderung der Größe von Orbitvorhersage-Linie und Schifficon
- Größen- und Motion-Effekt beim navigieren durch Sternenhintergrund UND Dreieck-Grid im Pixel-Design
- prozedurale Körper-Icons mit festem Muster aber unterschiedlichen Anordnungen der Planeten + relativer Größenfaktor pro Planet (Sonne größten Symbol, Erde optisch kleineren Icon)
- Maneuver-Planer Werkzeug bis zu 5 Knoten am Stück, Orbitvorhersage, anpassbare Geschwindigkeitsänderung in prograde (rechtläufig) und senkrechter Richtung mit Knotenausführung und automatischem Schub

## Repo-Struktur
Die Repository ist nun in einzelne Ordner eingeteilt worden und jeder trägt einen Teilbereich in dem Spiel, um die zu großen Dateien `rendering.py` und `predictor.py` aufzuteilen.

**/bodies/**
- Körpercode für Rendering, Initialisierung, Vorhersage und Design

**/config/**
- Konfigurationsdateien, jeweils `config.json` für feste Parameter und `solar_system.json` für die Körper + ConfigLoader

**/graphify-out/** (von KI genutzt)
- Graphen-Ansicht jeder Methode, Call, Klasse und Datei aus dem Code (als Tokenspar-Tool für `claude code`, welches für die Entwicklung des Spiels genutzt wurde)

**/Obsidian Vault/**
- Schreibteil & Dokumentation des Spiels und der Seminararbeit im rohen Format (im Moment: Markdown)

**/physics/**
- Funktionen für Referenz-Körper-System, Vektoroperatoren, Welt-/Spielerphysik, Schubanwendung, Verfahren zur Lösung von Anfangswertproblemen für Schiffphysik und Schiffvorhersage und Berechnung von Keplerbahnen (Planeten only)

**/render/**
- Renderpipeline des Spielfensters, aller UI-Elemente, Körper, Hintergrund, Vorhersagelinien, Text, Schiff, Bahnen und OpenGL Shaders (`render/gl/`)

**/runtime/**
- Startloop des Spiels, Inputhandler, Fenster, usw.

**/ship/**
- expliziter Code für Schiffdesign, Maneuver-Tool Funktion, Kamera und Schiffvorhersage

**/tests/**
- Python Testskripts für jeweilige Elemente/Bereiche in dem Code, besonders hilfreich für Debugging

**/tools/**
- einzelne Datei für Screenshoterstellung bestimmter Elemente oder des gesamten Spielfensters

**/ui/**
- UI Code für Schriftarten, HUD (Head-Up-Display), widgets, devui, Text, Design und initialisierung
