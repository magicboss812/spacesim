#version 330

// Hintergrund-ebene: grundverlauf, akzent-tiefenglut, dreiecksgitter,
// akzent-knoten, rasterschleier, vignette. Alles prozedural in EINEM
// vollbild-pass -- siehe background.py und .claude/rules/background.md.
//
// WARUM FRAGMENT-SHADER UND NICHT GEOMETRIE: als linien gezeichnet braucht
// das gitter eine obergrenze fuer die linienzahl, und die laesst beim
// hineinzoomen eine ganze dekade in EINEM bild verschwinden. Hier kostet es
// pro pixel statt pro linie, also gibt es keine obergrenze und kein poppen.
//
// VIRTUELLES PIXELRASTER. Der ganze pass rechnet auf gerundeten koordinaten
// (u_pixel), also in groben "hardware-pixeln" statt in bildschirmpixeln. Das
// bindet den hintergrund an das spieler-HUD, dessen anzeigeschrift laut
// ui/theme.py "gerastert OHNE kantenglaettung und auf ein vielfaches von
// fuenf pixel gerundet" ist. Alle kanten sind deshalb `step`, nicht
// `smoothstep`: eine weiche kante waere hier der fremdkoerper, nicht der
// treppeneffekt.
//
// FORMSPRACHE: FASE STATT RUNDUNG (ui/theme.py). Die knoten sind darum
// pixel-RAUTEN -- manhattan-abstand im virtuellen raster --, keine runden
// punkte: eine raute ist die 45-grad-fase in ihrer kleinsten form.
//
// PALETTE: die vier HUD-farben plus dessen feste grundierung. Grund
// #04070c (sehr dunkles marineblau), linien in der HUD-kantenfarbe
// #7fb4cc, knoten im akzent (Vorgabe: HUD-cyan #17b2c4). Ein warmer
// hintergrund entwertet die cyan-daten davor.
//
// KOORDINATEN: gl_FragCoord.y zeigt nach OBEN, welt-y ebenfalls -- dieser
// shader braucht also keinen y-flip (anders als line.vert). Alles, was von
// der ankerposition abhaengt, kommt als PHASE herein: bei 1e11 m bliebe von
// einer float32-weltkoordinate nichts uebrig.

const float SQRT3 = 1.7320508075688772;

const float LINE_BASE = 0.10;
const float NODE_BASE = 0.55;

// Halbe linienbreite in VIRTUELLEN pixeln. 0.5 = genau ein virtuelles pixel
// breit; die kante ist hart, die breite also exakt quantisiert.
const float LINE_HALF_WIDTH = 0.5;

// Knotenradius in virtuellen pixeln (manhattan) -- 2.0 ergibt eine raute aus
// 13 virtuellen pixeln.
const float NODE_RADIUS = 2.0;

// Die drei linienscharen unterschiedlich gewichten -- so hat das feld eine
// maserung statt perfekter isotropie.
const vec3 FAMILY_WEIGHT = vec3(1.0, 0.7, 0.7);

// HUD-kantenfarbe #7fb4cc (ui/theme.py Palette.edge).
const vec3 GRID_RGB = vec3(0.498039, 0.705882, 0.800000);

// Grundierung: um #04070c herum, oben eine spur heller.
const vec3 GROUND_TOP = vec3(0.027451, 0.047059, 0.078431);   // #070c14
const vec3 GROUND_MID = vec3(0.015686, 0.027451, 0.047059);   // #04070c
const vec3 GROUND_BOT = vec3(0.007843, 0.015686, 0.039216);   // #02040a

uniform vec2 u_viewport;
uniform vec3 u_accent;
uniform float u_grid_opacity;
uniform float u_pixel;
uniform float u_pixel_round;

uniform int u_level_count;
uniform float u_level_sp[4];       // bildschirm-zellweite in px
uniform float u_level_alpha[4];    // linien-deckkraft
uniform float u_level_node[4];     // knoten-deckkraft
uniform vec2 u_level_phase[4];     // gitterphase (a, b) des ankers, mod 2

out vec4 fragColor;

// Abstand zur naechsten linie einer schar, in pixeln.
float line_distance(float t, float sp) {
    return abs(t - round(t)) * sp;
}

// Form EINER virtuellen zelle, als harte maske (kein smoothstep -- der
// treppeneffekt ist hier das ziel).
//
//   u_pixel_round = 0  ->  die volle quadratische zelle. Das ist der reine
//                          pixelraster-look: benachbarte zellen stossen
//                          nahtlos aneinander.
//   u_pixel_round = 1  ->  runder punkt mit spalt, also eine leuchtpunkt-
//                          matrix. Die einzelne marke wird als PUNKT lesbar,
//                          statt als teil einer flaeche -- das ist es, was
//                          eine duenne diagonale von "weichem raster" zu
//                          "gezeichneter linie" macht.
//
// Der spalt waechst mit der rundung: eine runde zelle ohne spalt saehe nur
// aus wie ein leicht kleineres quadrat.
//
// FUELLGRAD-AUSGLEICH: bei round = 1 bleiben nur noch rund 53 % der zelle
// stehen (gemessen in tests/background_test.py §8c). Ohne ausgleich wuerde
// `grid_opacity` bei jeder rundung etwas anderes bedeuten und man muesste
// beide regler gemeinsam nachziehen. Die tinte wird deshalb durch den
// erwarteten fuellgrad geteilt -- danach sind die beiden knoepfe orthogonal.
const float CELL_FILL_ROUND = 0.53;

float cell_mask(vec2 frag_raw, float px) {
    if (u_pixel_round <= 0.0) {
        return 1.0;
    }
    vec2 q = abs(fract(frag_raw / px) - 0.5) * 2.0;   // 0 mitte .. 1 rand
    float d = mix(max(q.x, q.y), length(q), u_pixel_round);
    return step(d, 1.0 - 0.18 * u_pixel_round);
}

void main() {
    vec2 vp = u_viewport;
    float px = max(1.0, u_pixel);

    // Auf die mitte der virtuellen zelle rasten. ALLES darunter rechnet mit
    // dieser koordinate, damit jede kante auf dem raster liegt.
    vec2 frag = (floor(gl_FragCoord.xy / px) + 0.5) * px;
    vec2 centre = vp * 0.5;
    vec2 rel = frag - centre;

    // Tinte (gitter, knoten) wird getrennt gesammelt: die zellmaske darf nur
    // sie treffen, nicht den grundverlauf -- ein loechriger himmel waere
    // etwas anderes als eine leuchtpunkt-matrix.
    vec3 ink = vec3(0.0);

    // ------------------------------------------------------ grundverlauf
    float ty = 1.0 - frag.y / max(vp.y, 1.0);
    vec3 col = (ty < 0.55)
        ? mix(GROUND_TOP, GROUND_MID, ty / 0.55)
        : mix(GROUND_MID, GROUND_BOT, (ty - 0.55) / 0.45);

    // ------------------------------------- akzent-tiefenglut (aussermittig)
    vec2 glow_c = vec2(vp.x * 0.72, vp.y * 0.72);
    float glow_r = max(vp.x, vp.y) * 0.7;
    float gd = clamp(length(frag - glow_c) / max(glow_r, 1.0), 0.0, 1.0);
    float glow = mix(0.10, 0.028, smoothstep(0.0, 0.45, gd))
               * (1.0 - smoothstep(0.45, 1.0, gd));
    col += u_accent * glow;

    // ------------------------------------------------- radialer zerfall
    // Ohne ihn liest sich das gitter als begrenztes zeichenblatt. Der
    // zerfall selbst bleibt weich -- er ist eine deckkraft ueber viele
    // hundert pixel, keine kante.
    float diag = max(length(centre), 1.0);
    float dissolve = 1.0 - smoothstep(0.35, 1.05, length(rel) / diag);

    // ------------------------------------------------------------ gitter
    if (dissolve > 0.0) {
        for (int i = 0; i < u_level_count; ++i) {
            float sp = u_level_sp[i];
            if (sp <= 0.0) {
                continue;
            }
            // (a, b) sind die gitterkoordinaten: an einem knoten ganzzahlig
            // mit GERADER summe. Siehe background.py::_phases.
            float a = SQRT3 * rel.x / sp + u_level_phase[i].x;
            float b = rel.y / sp + u_level_phase[i].y;

            // Die drei scharen sind exakt b, m und n -- daran sieht man auch,
            // dass sie konkurrent sind.
            float m = (a + b) * 0.5;
            float n = (a - b) * 0.5;

            // Abstaende in VIRTUELLEN pixeln, damit die linienbreite in
            // rasterzellen zaehlt und nicht in bildschirmpixeln.
            float db = line_distance(b, sp) / px;
            float dm = line_distance(m, sp) / px;
            float dn = line_distance(n, sp) / px;

            // Harte kante: step, nicht smoothstep.
            float cov = step(db, LINE_HALF_WIDTH) * FAMILY_WEIGHT.x
                      + step(dn, LINE_HALF_WIDTH) * FAMILY_WEIGHT.y
                      + step(dm, LINE_HALF_WIDTH) * FAMILY_WEIGHT.z;

            ink += GRID_RGB * (u_level_alpha[i] * LINE_BASE * dissolve
                               * u_grid_opacity * min(cov, 1.6));

            // ------------------------------------------- akzent-knoten
            float na = u_level_node[i];
            if (na <= 0.0) {
                continue;
            }

            // FRUEHAUSSTIEG, und der traegt den ganzen shader: ein knoten
            // liegt auf ALLEN DREI scharen. Ist der pixel von auch nur einer
            // weiter entfernt als der knotenradius, kann kein knoten in
            // reichweite liegen. Die drei abstaende stehen ohnehin schon da,
            // der test kostet also nichts -- und er wirft rund 99 % der
            // pixel raus, bevor die 3x3-suche laeuft. Ohne ihn lief die
            // schleife fuer JEDEN pixel: gemessen 9.3 ms je bild statt
            // 0.35 ms.
            if (max(db, max(dm, dn)) > NODE_RADIUS) {
                continue;
            }

            // Naechsten gueltigen knoten suchen. Das lattice ist in (m, n)
            // ein 120-grad-basisgitter, dessen naechster punkt sich nicht
            // durch schlichtes runden findet -- also die 3x3-nachbarschaft
            // abklappern.
            float mr = floor(m + 0.5);
            float nr = floor(n + 0.5);
            float best = 1.0e18;
            for (int dmi = -1; dmi <= 1; ++dmi) {
                for (int dni = -1; dni <= 1; ++dni) {
                    float M = mr + float(dmi);
                    float N = nr + float(dni);
                    // zurueck nach (q, p): q = M + N, p = M - N
                    vec2 d = vec2((a - (M + N)) * sp / SQRT3,
                                  (b - (M - N)) * sp) / px;
                    // MANHATTAN, nicht euklidisch: das gibt die pixel-raute.
                    best = min(best, abs(d.x) + abs(d.y));
                }
            }
            ink += u_accent * (na * NODE_BASE * dissolve * u_grid_opacity
                               * step(best, NODE_RADIUS));
        }
    }

    // Erst jetzt die zellform anwenden -- auf die tinte, nicht auf den grund,
    // und mit ausgeglichenem fuellgrad, damit u_grid_opacity unabhaengig von
    // u_pixel_round dasselbe bedeutet.
    col += ink * cell_mask(gl_FragCoord.xy, px)
               / mix(1.0, CELL_FILL_ROUND, u_pixel_round);

    // ------------------------------------------------------ rasterschleier
    // Jede zweite VIRTUELLE zeile minimal aufhellen -- an das raster
    // gebunden, nicht an echte pixel, sonst kaempfen zwei raster
    // gegeneinander.
    //
    // Die staerke ist bewusst niedrig: der grund ist fast schwarz, eine
    // aufhellung von 0.010 ist dort schon eine RELATIVE modulation von 18 %
    // und streift den schirm sichtbar. Gemessen an einer ruhigen spalte:
    // 57 gegen 47 (summe ueber RGB) bei 0.010. Halbiert bleibt die textur
    // spuerbar, ohne zu streifen.
    float row = floor((vp.y - frag.y) / px);
    if (mod(row, 2.0) < 1.0) {
        col += vec3(0.005, 0.0065, 0.008);
    }

    // ------------------------------------------------------------ vignette
    vec2 vig_c = vec2(vp.x * 0.5, vp.y * 0.55);
    float vd = length(frag - vig_c) / (max(vp.x, vp.y) * 0.78);
    col *= 1.0 - 0.55 * smoothstep(0.38, 1.0, vd);

    fragColor = vec4(col, 1.0);
}
