#version 330

// Putz-pass im zellraster (eine zelle je fragment, vertex: body_pixel.vert
// ueber den ganzen puffer). Eine zelle, die keinem ihrer vier nachbarn
// gleicht, waehrend zwei davon untereinander gleich sind, ist ein einzelner
// fleck (eine figur, kleiner als eine zelle). Sie nimmt deren farbe an --
// was ein pixel-artist von hand wegputzt. Nur unter gleichem kennzeichen:
// ein uebergang besteht absichtlich aus zwei zustaenden.
//
// Eigener pass, weil der zusammenbau je BILDSCHIRMPIXEL laeuft: dort kostete
// dieselbe pruefung k*k mal so viel.

uniform sampler2D u_cells;
uniform sampler2D u_ids;
uniform ivec2 u_grid;

layout(location = 0) out vec4 o_cell;
layout(location = 1) out vec4 o_id;

bool fetch(ivec2 t, out vec4 cell, out vec4 id) {
    if (t.x < 0 || t.y < 0 || t.x >= u_grid.x || t.y >= u_grid.y) {
        cell = vec4(0.0);
        id = vec4(0.0);
        return false;
    }
    cell = texelFetch(u_cells, t, 0);
    id = texelFetch(u_ids, t, 0);
    return id.a > 0.1;
}

void main() {
    ivec2 t = ivec2(floor(gl_FragCoord.xy));
    vec4 cell, id;
    if (!fetch(t, cell, id)) {
        o_cell = vec4(0.0);
        o_id = vec4(0.0);
        return;
    }
    ivec2 dirs[4] = ivec2[4](ivec2(-1, 0), ivec2(1, 0), ivec2(0, -1), ivec2(0, 1));
    vec4 nc[4], ni[4];
    bool on[4];
    int same = 0;
    for (int i = 0; i < 4; ++i) {
        on[i] = fetch(t + dirs[i], nc[i], ni[i]) && abs(ni[i].a - id.a) < 0.05;
        if (on[i] && distance(ni[i].rgb, id.rgb) < 0.008) {
            same++;
        }
    }
    if (same == 0) {
        for (int i = 0; i < 4; ++i) {
            bool pair = false;
            for (int j = 0; j < 4; ++j) {
                if (j != i && on[i] && on[j] && distance(ni[i].rgb, ni[j].rgb) < 0.008) {
                    pair = true;
                }
            }
            if (pair) {
                // Farbe der nachbarn, aber die eigene helligkeit und
                // halbtonstufe, damit das licht nicht springt.
                float l0 = max(dot(cell.rgb, vec3(0.299, 0.587, 0.114)), 1e-3);
                float l1 = max(dot(nc[i].rgb, vec3(0.299, 0.587, 0.114)), 1e-3);
                cell = vec4(min(nc[i].rgb * (l0 / l1), vec3(1.0)), cell.a);
                id = ni[i];
                break;
            }
        }
    }
    o_cell = cell;
    o_id = id;
}
