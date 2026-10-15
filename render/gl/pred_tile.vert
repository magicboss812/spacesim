#version 330

// Kachel der vorhersagelinie: zusammenbau und loeschen laufen nur ueber die
// kacheln, die die linie beruehrt -- nicht ueber ihr ganzes rechteck, das
// bei naher kamera fast der ganze schirm ist.
in vec2 a_corner;        // 0 / 1, -1 / +1 (das eck-quad der segmente)
in vec2 i_tile;          // linke untere ecke, GL-px

uniform vec2 u_viewport;
uniform float u_tile;

void main() {
    vec2 p = i_tile + vec2(a_corner.x, a_corner.y * 0.5 + 0.5) * u_tile;
    gl_Position = vec4(p / u_viewport * 2.0 - 1.0, 0.0, 1.0);
}
