#version 330

// Zieht die koerper-ebene (eine textur in 1/u_cell der aufloesung) auf den
// schirm: jede zelle wird ein u_cell x u_cell grosser block, ohne filter.
// texelFetch statt sampler-filter, damit keine rundung einer
// gleitkomma-texturkoordinate eine zelle um einen pixel verschiebt.
//
// Das raster haengt OBEN LINKS (die welt zaehlt top-down); die textur
// zaehlt ihre zeilen von unten. Die farbe ist vormultipliziert.

uniform sampler2D u_layer;
uniform float u_cell;
uniform float u_height_px;   // hoehe des ziels in pixeln
uniform ivec2 u_cells;       // groesse der ebene in zellen

out vec4 fragColor;

void main() {
    float y_td = u_height_px - gl_FragCoord.y;
    ivec2 c = ivec2(int(floor(gl_FragCoord.x / u_cell)),
                    u_cells.y - 1 - int(floor(y_td / u_cell)));
    if (c.x < 0 || c.y < 0 || c.x >= u_cells.x || c.y >= u_cells.y) {
        discard;
    }
    vec4 color = texelFetch(u_layer, c, 0);
    if (color.a <= 0.0) {
        discard;
    }
    fragColor = color;
}
