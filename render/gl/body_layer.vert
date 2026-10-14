#version 330

// Dreieck-streifen fuer das hochziehen der koerper-ebene. Er deckt nur das
// rechteck, in das in diesem frame wirklich gezeichnet wurde (u_rect, NDC:
// xmin, ymin, xmax, ymax) -- ein vollbild-pass kostete auch dann, wenn nur
// ein kleiner mond im bild steht.
in vec2 a_pos;
uniform vec4 u_rect;

void main() {
    vec2 t = a_pos * 0.5 + 0.5;
    gl_Position = vec4(mix(u_rect.xy, u_rect.zw, t), 0.0, 1.0);
}
