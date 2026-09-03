#version 330

// Das zellmuster der positions-marke, je fragment aufgeloest.
//
// > **Ein fragment gehoert nicht einem primitiv, sondern einem FLAECHENSTUECK.**
// > Die marke ist EIN quad; welche zellen ein pixel ueberdeckt, rechnet dieser
// > shader aus. Es gibt deshalb keine aneinanderstossenden primitive und damit
// > keine naht, an der zwei teildeckungen uebereinander liegen. Genau daran
// > krankte der canvas-entwurf: dort war jede zelle ein eigenes `fillRect`,
// > beide seiten einer gemeinsamen kante schrieben ihre eigene kantenglaettung
// > in dieselbe pixelspalte, und die naht wanderte beim bewegen sichtbar mit.
//
// > **Gemittelt wird ueber das PIXEL-QUADRAT, nicht nur ueber den umriss.**
// > Eine erste fassung glaettete lediglich freie aussenkanten und liess die
// > grenzen zwischen zwei stufen hart. Gemessen schwankte die helligkeit der
// > marke damit ueber einen pixel drift um **±5.4 %** -- die inneren grenzen
// > kippen beim wandern pixel fuer pixel zwischen zwei stufenfarben, und bei
// > deckkraeften von 0.55 bis 1.0 faellt das ins gewicht. Der exakte
// > box-filter unten nimmt stattdessen die flaechenanteile aller ueberdeckten
// > zellen. Bei 3.2 px je zelle laeuft ein uebergang ueber rund einen pixel --
// > das muster bleibt hart, die bewegung wird stetig.

in vec2 v_local;             // icon-koordinaten, y nach oben

uniform uint u_cells[16];    // 2 bit je zelle, 16 je wort -> 256 zellen
uniform int u_grid;          // kantenlaenge N des zellgitters (ungerade)
uniform float u_unit;        // zellbreite in icon-koordinaten
uniform float u_radius_px;   // radius der marke in pixeln
uniform float u_edge_px;     // kantenbreite in pixeln (0 = hart)
uniform float u_cell_gap;    // anteil der zelle, der als spalt frei bleibt
uniform float u_cell_rim;    // BREITE des umrisses in PIXELN
uniform float u_cell_rim_dark;  // wie dunkel dieser umriss wird

uniform vec3 u_tier_dim;
uniform vec3 u_tier_base;
uniform vec3 u_tier_bright;
uniform vec3 u_tier_alpha;   // deckkraft (dunkel, grund, hell)

uniform float u_fade;        // ueberblendung gegen den echten koerper
uniform float u_halo_alpha;
uniform uint u_seed;         // der seed des koerpers, fuer die zell-helligkeit
uniform float u_shade;       // +- helligkeit je zelle (0 = alle gleich)

out vec4 fragColor;

float cell_shade(ivec2 c, uint seed) {
    // Jede Zelle bekommt ihre EIGENE Helligkeit. Drei Stufen sind zu wenig
    // Tiefe: gleich eingestufte Nachbarn werden sonst zu einer Flaeche, und
    // die Marke sieht gedruckt aus statt texturiert. Der Wert haengt nur an
    // (Zelle, Seed), ist also ueber Frames konstant -- er darf nicht
    // flimmern, wenn sich die Marke bewegt.
    uint h = uint(c.x + 512) * 374761393u
           + uint(c.y + 512) * 668265263u + seed;
    h = (h ^ (h >> 13u)) * 1274126177u;
    h = h ^ (h >> 16u);
    return float(h & 0xFFFFu) / 65535.0;
}

uint tier_at(ivec2 c, int r) {
    if (abs(c.x) > r || abs(c.y) > r) {
        return 0u;
    }
    int index = (c.y + r) * u_grid + (c.x + r);
    // Dynamische indizierung eines uniform-ARRAYS ist in GLSL 330 erlaubt
    // (anders als bei sampler-arrays); ein uvec4 wurde es nur, solange 64
    // zellen reichten.
    return (u_cells[index >> 4] >> uint((index & 15) * 2)) & 3u;
}

void main() {
    int r = (u_grid - 1) / 2;
    float s = u_unit;
    float half_s = s * 0.5;
    ivec2 nearest = ivec2(floor(v_local / s + 0.5));

    // Die kantenlaenge des pixelquadrats, in icon-koordinaten. Bei
    // u_edge_px = 0 schrumpft es auf einen punkt -- dann wird wieder hart
    // abgetastet, und das ist die gegenprobe in body_icon_test.py §5.
    //
    // > **Der filter darf nie breiter werden als eine HALBE zelle.** Sonst
    // > mittelt ein feineres raster sich selbst weg: bei 15x15 ist eine zelle
    // > rund 1 px breit, und ein 1-px-filter macht daraus einen gleichmaessigen
    // > fleck. Mehr raster wuerde dann WENIGER muster bedeuten -- genau
    // > verkehrt herum. Mit dem deckel schaerft sich der filter mit dem
    // > raster mit.
    float px = max(1e-4, min(u_edge_px / max(1.0, u_radius_px), s * 0.5));
    float hw = px * 0.5;
    vec2 lo = v_local - hw;
    vec2 hi = v_local + hw;
    float pixel_area = px * px;

    // Vormultiplizierte summe ueber die hoechstens vier (bei kleinem px)
    // ueberdeckten zellen. Die 3x3-nachbarschaft deckt jeden fall ab, solange
    // ein pixel schmaler ist als eine zelle -- bei 3.2 px je zelle mit
    // reichlich luft.
    vec3 acc = vec3(0.0);
    float cov = 0.0;
    for (int dj = -1; dj <= 1; dj++) {
        for (int di = -1; di <= 1; di++) {
            ivec2 c = nearest + ivec2(di, dj);
            uint tier = tier_at(c, r);
            if (tier == 0u) {
                continue;
            }
            // Der SPALT: die zelle wird eingezogen (Vorgabe 0 -- der
            // Umriss unten macht dieselbe Arbeit besser).
            float fill = half_s * (1.0 - clamp(u_cell_gap, 0.0, 0.9));
            vec2 centre = vec2(c) * s;
            vec2 olo = max(lo, centre - fill);
            vec2 ohi = min(hi, centre + fill);
            vec2 d = max(ohi - olo, vec2(0.0));
            float part = (d.x * d.y) / pixel_area;
            if (part <= 0.0) {
                continue;
            }
            vec3 col = (tier == 1u) ? u_tier_dim
                     : ((tier == 2u) ? u_tier_base : u_tier_bright);
            float a = (tier == 1u) ? u_tier_alpha.x
                    : ((tier == 2u) ? u_tier_alpha.y : u_tier_alpha.z);
            // Eigene Helligkeit der Zelle, auf Farbe UND Deckkraft: nur die
            // Farbe zu variieren gibt bunte Flecken, nur die Deckkraft einen
            // Schleier. Beides zusammen liest sich als Licht auf einer
            // Oberflaeche -- die "Schatten" zwischen den Zellen.
            float shade = 1.0 + (cell_shade(c, u_seed) - 0.5) * 2.0 * u_shade;
            col *= shade;
            a = clamp(a * mix(1.0, shade, 0.55), 0.0, 1.0);

            // > **Der UMRISS wird gezeichnet, nicht erhofft.** Vorher entstand
            // > er als Nebenwirkung des over-Operators -- und damit aus der
            // > REIHENFOLGE der Schleife, die willkuerlich ist: waagerechte
            // > und senkrechte Grenzen dunkelten unterschiedlich stark nach,
            // > und der Halo fuellte den Rest wieder auf (gemessen lagen bei
            // > Radius 16 nur 2 von 28 senkrechten und 0 von 26 waagerechten
            // > Grenzen ueberhaupt unter 90 %). Jede Zelle dunkelt jetzt zu
            // > ihrem EIGENEN Rand hin nach, in beiden Achsen gleich, weil
            // > `max(|qx|,|qy|)` ein Quadrat ist. Das ist zugleich die
            // > "gemalte" Kantenglaettung: der dunkle Saum liest sich als
            // > Schatten der Zelle.
            // Die Breite ist ein BILDSCHIRM-mass, kein Anteil der Zelle --
            // dieselbe Regel wie fuer die Linienbreiten in body_line.vert.
            // Als Anteil war der Saum bei Radius 16 nur 0.55 px breit, und ob
            // er ueberhaupt abgetastet wurde, hing an der Bruchteil-Position
            // der Marke. Weil x und y verschiedene Phasen haben, zeigte die
            // eine Achse ihre Umrisse und die andere nicht: gemessen 7 von 28
            // senkrechten Grenzen unter 90 %, aber 0 von 26 waagerechten.
            // Erst ab rund einem Pixel Breite kommen beide Achsen an.
            float rim_u = min(u_cell_rim / max(1.0, u_radius_px), half_s * 0.7);
            vec2 q = abs(v_local - centre);
            float edge = max(q.x, q.y);
            float rim = clamp((edge - (half_s - rim_u)) / max(rim_u, 1e-6),
                              0.0, 1.0);
            col *= mix(1.0, u_cell_rim_dark, rim);
            // NACHEINANDER ueberblenden, nicht mitteln -- das ist die
            // Architektur des Canvas-Entwurfs, und daher kommt seine Tiefe.
            // Ein linearer Flaechenschnitt ist der mathematisch saubere
            // Box-Filter und sieht genau deshalb flach aus: er zieht alles
            // zum Mittelwert. Der over-Operator laesst dagegen jede
            // Teildeckung auf der vorigen liegen, so dass Zellgrenzen
            // nachdunkeln (die "Umrisse") und zwischen zwei Stufen viele
            // Zwischentoene entstehen statt eines Uebergangs.
            float ca = a * part;
            acc = acc * (1.0 - ca) + col * ca;
            cov = cov * (1.0 - ca) + ca;
        }
    }

    // Der halo faellt STETIG ab und hat keine kante -- er traegt die marke
    // gegen das sternenfeld, ohne selbst als form zu lesen.
    float rr = length(v_local);
    float halo = u_halo_alpha * exp(-rr * rr * 1.9);

    // Zellen ueber halo zusammensetzen, beides vormultipliziert.
    float behind = halo * (1.0 - cov);
    float alpha = cov + behind;
    if (alpha <= 0.002) {
        discard;
    }
    fragColor = vec4((acc + u_tier_base * behind) / alpha, alpha * u_fade);
}
