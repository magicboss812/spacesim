Zuerst kam in den Sinn ein Simulationsspiel für die Seminararbeit zu fertigen. Grundlegende Zuerst kam in den Sinn ein Simulationsspiel für die Seminararbeit zu fertigen. Grundlegende Motive waren **N-Body Physik**, **2D**, **Python**, **Reference-System**. Das LogBuch beschreibt die Anfangsentwicklung mithilfe von einem 100% ki-generiertem Prototyp, und ebenso daraus resultierende Hindernisse.

Kurzgefasst war die KI (`openai-gpt-5-1`) dabei nicht in der Lage, ein komplettes Spiel in nur ein paar Prompts zu erstellen. Es erwies sich deutlich sinnvoller, das Spiel **mithilfe** von KI, gesteuert vom User (mich) zu programmieren. Die primären Funktionen werden berücksichtigen und verstanden, während z.B. Rendering und wichtige bedingte Code-Implementierungen von der KI mehr ausgearbeitet wird als vom User. Dies würde den Eigenanteil am Spiel nicht negativ beeinflussen.

Das Spieldateien werden mit Python erstellt und haben jeweils eine eigene Funktion. So wäre die Hierarchie ganz eindeutig und Änderungen können besser vorgenommen werden. Die Dateien werden auch unabhängiger, das erleichtert das Debugging und die fehlerfreie Implementierung:

Beispiel-Hierarchie, entspricht ziemlich sehr dem jetzigen Entwicklungsstand (21.06.2026)
```markdown-tree
spacesim
	test.py --> enthält viele wichtige Debugging Elemente
	main.py --> fertige Ausführdatei, sauber und unabhängiger
	bodies.py --> initialisiert die Körper
	camera.py --> verwaltet die Kamerasteuerung
	rendering.py --> Rendering-Pipeline (Shader, Körper, UI)
	predictor.py --> Predictor-Navigation
	loader.py --> ladet Config-Dateien für Parameter und Körper
	reference_frames.py --> transformiert Reference-System
	schiff.py --> Schiffkontrolle: Drehung & Schub
	Config
		config.json --> alle Parameter in dem Code
		solar_system.json --> Körper-Parameter
	vec.py --> Vektor-Operatoren für Python
	world.py --> Weltphysik für Schiff & Körper
