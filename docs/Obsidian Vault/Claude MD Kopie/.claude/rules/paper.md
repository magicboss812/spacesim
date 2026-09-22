---
paths:
  - "spacesim/docs/**"
---

# Die Seminararbeit — `spacesim/docs/`

Ein **Obsidian-Vault**, Stand: im Schreiben. Vault-Wurzel ist `spacesim/docs/`
(dort liegt `.obsidian/`), deshalb sind Wikilinks wurzelrelativ:
`[[Obsidian Vault/Evaluation/Integrator.md]]`, nicht `[[Integrator]]` von
`_Arbeit/` aus. `docs/` ist im `spacesim`-Repo **ungetrackt** (`?? docs/`) —
die Git-Regel aus `CLAUDE.md` gilt auch hier.

**Die Abgabe ist deutsch.** Jeder Satz, der in `_Arbeit/` landet, ist deutsch —
egal in welcher Sprache der Nutzer mit mir redet oder die Quelle geschrieben ist.
Erklärungen im Chat folgen der Sprache der Frage.

## Ordner und ihre Rolle

Kurz: `_Arbeit/` = Abgabe, `Quellen/` = Belege, `Evaluation/*.md` = Rohmaterial.
**Nicht vermischen** — Rohmaterial wandert nie ungeprüft in den Fließtext.

| Pfad (unter `Obsidian Vault/`) | Rolle | Umgang |
|---|---|---|
| `Evaluation/_Arbeit/NN Titel.md` | **Fließtext der Abgabe**, ein File pro Kapitel (`01 Einleitung`, `02 Physikalische Grundlagen`, …) | nur auf Anfrage ändern; neue Kapitel mit der nächsten Nummer anlegen |
| `Evaluation/_Arbeit/00 Gliederung (KI Ansatz, irrelevant).md` | verworfener KI-Entwurf der Gliederung | **nicht bindend.** Leitfrage, Seitenplan und Kapitelnummern dort gelten nicht; maßgeblich ist die **Problemfrage in `01 Einleitung.md`** und die Kapitelverweise in den Kapiteln selbst |
| `Evaluation/_Arbeit/Quellen/Quellen.md` | **das eine Literaturverzeichnis** | jede neue Quelle hier eintragen, bevor sie im Text steht |
| `Evaluation/_Arbeit/Quellen/Literatur/` | PDFs: Apollo-11-Mission-Report `[2]`, Curtis `[Curtis]` | lesen siehe unten |
| `Evaluation/_Arbeit/Quellen/orbital-mechanics-notes-main/…/` | komplettes Repo des Wikis **orbital-mechanics.space** `[8]` (gitignored) | **nur zum Verstehen, und nur auf Nachfrage** — siehe unten |
| `Evaluation/*.md` (`Physik`, `Integrator`, `Anfangsidee`, `Project-Entwicklung`) | Notizen, Rechnungen, Begründungen | Rohmaterial mit **bekannten Fachfehlern** — vor Übernahme gegen Quelle bzw. Code prüfen |
| `Evaluation/KI-Nutzung.md` | **Protokoll jeder KI-Hilfe** (Modell / Nutzung / Ergebnis) | siehe „KI-Nutzung protokollieren" |
| `Evaluation/Grafiken & Bilder/` | Abbildungen der Arbeit | Einbettung `![[…]]`, Bildquelle in `Quellen.md` → `## Bildquellen` |
| `Readme's Github/` | datierte Entwicklungsberichte (März–Juni 2026) | Material für das Entwicklungskapitel; beschreibt den Code **zu diesem Datum** (`test.py`, pyopengl) — nie als aktuellen Stand zitieren |
| `Plan für SpaceSim.md`, `Planumsetzung/Plan für SpaceSim.md` | Feature-Pläne; **zwei abweichende Kopien** | nur lesen |
| `logbuch bis 11.03.2026.docx` | LogBuch der Anfangsentwicklung (KI-Prototyp) | Material für das Entwicklungskapitel; `~$…docx` ist Words Sperrdatei |
| `Claude MD Kopie/` | veralteter Schnappschuss von `CLAUDE.md` + Rules | **weder lesen noch pflegen** — die echten liegen in `Werk/` |

Ignorieren: `.obsidian/`, `.trash/`, `.makemd/`, `.space/`, `.claudian/`,
`docs/.claude/` (leer), `Tags/`, `__pycache__/`.

## Quellen lesen

- **Curtis**, *Orbital Mechanics for Engineering Students*, 4. Aufl. (947 PDF-Seiten,
  Textebene vorhanden). **Druckseite ≠ PDF-Seite, und der Versatz ist nicht
  konstant**: gemessen PDF 22 = S. 14, PDF 101 = S. 93, PDF 301 = S. 295 (Versatz
  4–8). Die zitierte Seitenzahl kommt immer aus der Kopfzeile der gelesenen
  Seite, nie aus einer Umrechnung.
- **Apollo 11 Mission Report** (MSC-00171, 360 PDF-Seiten) ist ein **OCR-Scan** —
  der Text ist verrauscht, Zahlen daraus am Seitenbild (`Read` mit `pages`)
  gegenprüfen. Seitenzählung ist *Abschnitt-Seite* (`7-1` = Abschnitt 7 TRAJECTORY,
  S. 1), Vorspann römisch.
- Lesen: `Read` mit `pages` (max. 20 pro Aufruf), oder für Suche
  `pdftotext -layout -f N -l M <pdf> <scratchpad>/x.txt` (liegt in
  `/mingw64/bin`) und dann greppen. Nie den ganzen Curtis in den Kontext laden.
- **`orbital-mechanics-notes-main/`** ist ein ganzes Repo (Wiki-Quelltext,
  Notebooks, Skripte, Blender-Dateien, Bilder) der Seite orbital-mechanics.space
  `[8]`. **Nur benutzen, wenn der Nutzer etwas zum Verstehen fragt** — nicht
  von mir aus durchsuchen, nicht beim Schreiben oder Belegen heranziehen, nicht
  als Quelle vorschlagen. Wenn ich es benutze, **ausdrücklich sagen, dass die
  Erklärung aus diesem Ordner kommt, mit Dateipfad**. Inhalt sind nur die
  Kapitel-`.md`; `_toc.yml` ist das Inhaltsverzeichnis, URL-Pfad = Dateipfad
  (`…/intro/mass-force-and-newtons-law-of-gravitation.html` ↔
  `intro/mass-force-and-newtons-law-of-gravitation.md`). Die Seite baut selbst
  auf Curtis auf — für Belege in der Arbeit Curtis zitieren.
- **Beim Erklären** immer trennen: *was die Quelle sagt* (mit Seite) und *was ich
  ergänze*. Wo es passt, auf die Simulation abbilden (Zweikörperproblem ↔
  `bodies.body.kepler_relative_xy()`, Integrator ↔ `.claude/rules/physics-world.md`,
  Bezugssysteme ↔ `.claude/rules/reference-frames.md`, Manöver/Δv ↔
  `.claude/rules/maneuver.md`).

## Zitieren

- Nummerierte Kurzbelege aus `Quellen.md`: `[n]`, mit Seite `[n, S. x]`. Der
  Bestand ist uneinheitlich (`[1 S.2]`, `[Curtis S.23]`, `[Curtis, Kap. 2]`) —
  neuen Text in `[n, S. x]` schreiben, Altbestand nur auf Anfrage angleichen.
- Neue Quelle → nächste freie Nummer, richtige Rubrik (*Beitragsquellen* = Web,
  *Literatur / Papers*, *Bildquellen*), bei Web-Quellen `zuletzt TT.MM.JJJJ`.
  Achtung: `[3]` ist derzeit **nicht vergeben** (Lücke), Curtis hat noch keine Nummer.
- **Nie eine Seite, ein Zitat oder eine Quelle angeben, die ich nicht selbst
  gelesen habe.** Ein Beleg muss die Aussage auf genau dieser Seite tragen.
- Englische Quelle wörtlich zitiert → deutsche Übersetzung mit Vermerk
  „(eigene Übersetzung)". Sonst paraphrasieren.
- Rangfolge für Fachaussagen: Curtis > NASA-Berichte (`[2]`) > NASA-Webseiten >
  orbital-mechanics.space > Sonstiges (z. B. GRIN `[9]` ist eine
  Studierendenarbeit — schwach, nur ersetzen, wenn der Nutzer will).
- **Aussagen über die eigene Simulation** brauchen keine Literatur, aber einen
  Messwert aus dem Code: Zahlen aus `config/config.json` oder einem Test
  (`.claude/rules/tests.md`), nie aus `Evaluation/*.md` abschreiben. Beispiel:
  der Code rechnet mit `G = 6.6730831e-11` (`config.json`), Kap. 2 nennt
  6,6742·10⁻¹¹, CODATA 2018 ist 6,67430·10⁻¹¹ — weicht das Spiel vom
  Literaturwert ab, sagt der Text das.

## Deutsch schreiben

- Wissenschaftlicher Stil, unpersönlich („die vorliegende Arbeit", Passiv) —
  kein „ich"/„wir" im Fließtext. Präsens für Sachverhalte, Präteritum für die
  eigene Entwicklung und für historische Missionen.
- Zahlen: Dezimalkomma, im LaTeX `3{,}986`; Zehnerpotenz `\cdot 10^{14}`, nie
  `*` oder `\times`; Einheit aufrecht mit Abstand `\ \mathrm{m/s^2}`. SI-Einheiten.
- Ein Begriff pro Konzept, durchgehend:

  | Englisch | in der Arbeit |
  |---|---|
  | gravity assist / swing-by | Gravitationsmanöver (Swing-by nur als Einführung des Synonyms) |
  | delta-v | Geschwindigkeitsänderung $\Delta v$ |
  | periapsis / apoapsis | Periapsis / Apoapsis |
  | semi-major axis | große Halbachse $a$ |
  | sphere of influence | Einflusssphäre (SOI) |
  | reference frame | Bezugssystem |
  | prograde / retrograde | prograd / retrograd |
  | patched conics | Patched-Conics-Näherung |

- Die deutschen Code-Namen (`Schiff`, `Erde`) sind keine Fachbegriffe — im
  Fließtext steht „Raumfahrzeug", „Erde", kein Code.

## Obsidian-Syntax erhalten

- Formeln: `$$ … \tag{n}$$`, darunter der Block-Anker `^eq-name`, Verweis
  `[[#^eq-name|(n)]]`. Nummern laufen durch die ganze Arbeit; beim Einfügen
  einer Formel alle folgenden `\tag` **und** ihre Verweise nachziehen.
- Abbildung: `![[…]]`, darunter `*Abb. n, Beschreibung – Quelle*`. Selbst
  erzeugte oder KI-erzeugte Grafiken werden als solche ausgewiesen.
- Frontmatter und Farb-/Sticker-Properties (`color`, `sticker`, `tags`) nicht anfassen.

## Wie ich helfe

- **Der Text gehört dem Nutzer.** Standard ist: erklären, Fehler zeigen,
  Formulierungen *vorschlagen*. Direkt in `_Arbeit/` schreiben nur, wenn darum
  gebeten; dann seinen Ton und Aufbau erhalten, nicht neu formulieren, was nicht
  gefragt war. Fachfehler immer melden, auch ungefragt.
- Jedes Kapitel dient der Problemfrage aus `01 Einleitung.md` (Δv-Ersparnis von
  Transfers und Gravitationsmanövern gegenüber direktem Flug, quantitativ in der
  eigenen Simulation nachgewiesen). Abschweifungen benennen.

## KI-Nutzung protokollieren

`Evaluation/KI-Nutzung.md` ist Teil der Redlichkeit der Arbeit. Nach **jeder**
Hilfe, die Text, Formeln, Abbildungen oder Quellenauswahl in `_Arbeit/`
beeinflusst, eine Zeile anhängen — Modell (exakte ID, z. B. `claude-opus-5`),
was gemacht wurde (mit Kapitelnummer), Ergebnis/Bewertung — und dem Nutzer
sagen, dass sie drinsteht. Reines Erklären einer Quelle ohne Textänderung
braucht keine Zeile.
