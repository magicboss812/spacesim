#version 330

// Sternenfeld als INSTANZIERTE quads -- bewusst keine punkt-sprites.
//
// Die tabelle (background.build_star_table) liegt EINMAL im VBO und wird nie
// je bild neu geschrieben: parallaxe, funkel- und zoomphase stehen je stern
// darin, drift und zoom kommen als uniforms. `a_corner` ist das gemeinsame
// einheitsquadrat, vier ecken als TRIANGLE_STRIP.
//
// WARUM KEINE PUNKT-SPRITES. Der erste entwurf zeichnete GL_POINTS und holte
// die zellform aus `gl_PointCoord`. Auf dem NVIDIA-treiber dieses rechners
// (4.6.0, 595.71) liefert `gl_PointCoord` in JEDEM fragment exakt (0, 0) --
// die punkte decken ihre 9x9 fragmente korrekt ab, aber die koordinate ist
// tot. Damit war `q = abs(0 - 0.5)*2 = 1`, der zellmaske fiel jedes fragment
// zum opfer, und das ganze sternenfeld verschwand, sobald `pixel_round > 0`
// stand -- also in der Vorgabe. Ein quad mit selbst gefuehrtem `v_uv` haengt
// von keiner treiber-eigenheit ab; `gl_PointSize`/`PROGRAM_POINT_SIZE` fallen
// gleich mit weg. tests/background_gl_test.py haelt das fest.
//
// DAS ATMENDE FELD. Beim hineinzoomen soll das feld mitzoomen -- sterne
// laufen auseinander --, ohne dass es beim herauszoomen verklumpt. Dazu
// bekommt jeder stern eine eigene, gleichverteilte zoomphase; daraus:
//
//     f = fract(star_zoom + zoomphase)       in [0, 1)
//     e = exp2(f)                            in [1, 2)   -- kacheldehnung
//
// Die kachel des sterns ist `viewport * e`, seine sichtbarkeitswahrschein-
// lichkeit also 1/e^2. Beim zoomen waechst e bis 2, dann faellt der stern auf
// 1 zurueck -- an beiden enden ist er durch das fenster w(f) unsichtbar, es
// gibt also keinen sprung. Weil die zoomphasen gleichverteilt sind, ist die
// ERWARTETE sichtbare sternzahl von `star_zoom` unabhaengig: das feld dehnt
// sich, bleibt aber gleich dicht. Genau das prueft §4b des tests.
//
// PIXELRASTER. Position und groesse werden auf ein virtuelles pixel gerundet
// (u_pixel), damit die sterne dieselbe harte kantenqualitaet haben wie die
// anzeigeschrift des HUDs -- keine kantenglaettung, keine halben pixel.
//
// Konvention: TOP-DOWN wie line.vert (y nach unten), passend zu
// BackgroundLayer.star_pan_px.

in vec2 a_corner;   // einheitsquadrat 0..1, geteilt von allen instanzen
in vec2 a_pos;      // position im einheitsquadrat, 0..1
in vec4 a_param;    // radius, grundhelligkeit, parallaxe, funkelphase
in float a_phase;   // zoomphase, 0..1

uniform vec2 u_viewport;
uniform vec2 u_pan;         // aufsummierte drift in top-down-pixeln
uniform float u_time;
uniform float u_opacity;
uniform float u_star_zoom;  // oktavenzaehler
uniform float u_zoom_amount; // 0 = starres feld, 1 = volles atmen
uniform float u_pixel;      // kantenlaenge des virtuellen pixels

out float v_alpha;
out vec2 v_uv;          // 0..1 ueber das sprite -- ersetzt gl_PointCoord
out float v_cells;      // kantenlaenge des sprites in virtuellen pixeln

void main() {
    float f = fract(u_star_zoom + a_phase);

    // Bei u_zoom_amount = 0 muss EXAKT das alte, starre feld herauskommen:
    // e = 1 und w = 1. Deshalb wird beides gegen den neutralwert gemischt und
    // nicht etwa f auf null gezwungen -- das wuerde die sterne nach ihrer
    // festen phase dauerhaft verschieden hell machen.
    float e = mix(1.0, exp2(f), u_zoom_amount);
    float w = mix(1.0,
                  smoothstep(0.0, 0.18, f) * (1.0 - smoothstep(0.82, 1.0, f)),
                  u_zoom_amount);

    vec2 tile = u_viewport * e;
    // mod() liefert bei positivem divisor immer [0, tile) -- auch fuer
    // negatives argument. Damit kachelt das feld in beide richtungen.
    vec2 p = mod(a_pos * tile - u_pan * a_param.z, tile);

    // Auf das virtuelle pixelraster rasten (mittelpunkt der virtuellen
    // zelle), damit der sprite buendig auf dem raster sitzt.
    float px = max(1.0, u_pixel);
    p = (floor(p / px) + 0.5) * px;

    // Nahe sterne (hohe parallaxe) minimal groesser, danach auf ganze
    // virtuelle pixel gerundet -- ein halbes pixel gibt es hier nicht.
    float r = a_param.x * (0.85 + a_param.z * 0.5);
    float cells = max(1.0, floor(r + 0.5));
    v_cells = cells;
    v_uv = a_corner;

    // Das quad sitzt mittig auf p und ist cells*px gross -- genau die
    // flaeche, die der punkt-sprite gehabt haette.
    vec2 corner = p + (a_corner - 0.5) * (cells * px);

    v_alpha = a_param.y * u_opacity * w
            * (0.72 + 0.28 * sin(u_time * 0.9 + a_param.w));

    gl_Position = vec4(
        (corner.x / u_viewport.x) * 2.0 - 1.0,
        1.0 - (corner.y / u_viewport.y) * 2.0,
        0.0, 1.0
    );
}
