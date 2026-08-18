#version 330

// EIN signed-distance-field-shader fuer praktisch die gesamte UI-flaeche.
//
// Deckt ab: abgerundete rechtecke (radius pro ecke), rahmen, schlagschatten,
// vertikaler farbverlauf -- und ueber radius = min(halbe breite, halbe hoehe)
// zusaetzlich kreise, ringe und kreisboegen (v_arc), also die grundform des
// attitude-rings. Damit braucht die UI genau eine geometrie-pipeline statt
// je einer fuer panel, button, kreis und bogen.
//
// Die form-parameter kommen als flat-varyings aus der (instanzierten)
// vertex-stufe -- pro instanz konstant, gleiche float32-genauigkeit wie
// zuvor als uniforms, exakt dieselbe rechnung.
//
// Kantenglaettung: das rechteck wird in PIXELN aufgespannt, der SDF liefert
// deshalb direkt einen pixel-abstand. smoothstep ueber +-0.5 px ergibt exakt
// eine pixel breite kante, ohne fwidth und ohne von der viewport-groesse
// abzuhaengen.

in vec2 v_local;
in vec2 v_half;
flat in vec4 v_radius;         // eckradien in pixeln: oben-links, oben-rechts, unten-rechts, unten-links
flat in vec4 v_fill;
flat in vec4 v_fill2;          // zweite verlaufsfarbe (unten)
flat in float v_gradient;      // 0 = einfarbig, 1 = vertikaler verlauf
flat in vec4 v_border_color;
flat in float v_border_width;
flat in vec4 v_shadow_color;
flat in vec2 v_shadow_offset;  // pixel, ortho-konvention (y nach oben)
flat in float v_shadow_softness;
flat in vec2 v_arc;            // startwinkel, ueberstrichener winkel (radiant); >= TAU = volle form

out vec4 fragColor;

const float TAU = 6.28318530718;

// Radius der ecke, in deren quadrant p faellt.
float corner_radius(vec2 p) {
    if (p.y >= 0.0) {
        return p.x < 0.0 ? v_radius.x : v_radius.y;
    }
    return p.x < 0.0 ? v_radius.w : v_radius.z;
}

// Standard-SDF eines abgerundeten rechtecks (negativ = innen).
float rounded_box(vec2 p, vec2 half_size, float radius) {
    vec2 q = abs(p) - half_size + radius;
    return min(max(q.x, q.y), 0.0) + length(max(q, vec2(0.0))) - radius;
}

float shape_distance(vec2 p) {
    float limit = min(v_half.x, v_half.y);
    float radius = clamp(corner_radius(p), 0.0, limit);
    return rounded_box(p, v_half, radius);
}

// Winkelmaske fuer kreisboegen. Die weiche kante wird ueber den radius
// skaliert, damit sie am aeusseren rand des bogens genauso breit ist wie
// innen (ein fester winkel-epsilon waere aussen viel zu weich).
float arc_mask(vec2 p) {
    if (v_arc.y >= TAU) {
        return 1.0;
    }
    float angle = atan(p.y, p.x);
    float delta = mod(angle - v_arc.x, TAU);
    float aa = 1.5 / max(length(p), 1.0);
    return smoothstep(0.0, aa, delta) * smoothstep(0.0, aa, v_arc.y - delta);
}

// "source over destination", beide seiten NICHT vormultipliziert -- passend
// zum globalen blend_func (SRC_ALPHA, ONE_MINUS_SRC_ALPHA).
vec4 over(vec4 src, vec4 dst) {
    float a = src.a + dst.a * (1.0 - src.a);
    if (a <= 0.0001) {
        return vec4(0.0);
    }
    vec3 rgb = (src.rgb * src.a + dst.rgb * dst.a * (1.0 - src.a)) / a;
    return vec4(rgb, a);
}

void main() {
    const float AA = 0.5;

    float mask = arc_mask(v_local);
    float d = shape_distance(v_local);

    // Deckungsgrad der gesamten form und der um die rahmenbreite
    // geschrumpften innenflaeche. Die differenz ist exakt der rahmen.
    float outer = (1.0 - smoothstep(-AA, AA, d)) * mask;
    float inner = (1.0 - smoothstep(-AA, AA, d + max(v_border_width, 0.0))) * mask;

    vec4 result = vec4(0.0);

    // Schlagschatten liegt HINTER der flaeche und wird deshalb zuerst
    // aufgebaut. Er benutzt dieselbe form, nur versetzt und weichgezeichnet.
    if (v_shadow_color.a > 0.0) {
        vec2 sp = v_local - v_shadow_offset;
        float sd = shape_distance(sp);
        float soft = max(v_shadow_softness, 0.5);
        float coverage = 1.0 - smoothstep(-soft, soft, sd);
        result = vec4(v_shadow_color.rgb, v_shadow_color.a * coverage);
    }

    vec4 body = v_fill;
    if (v_gradient > 0.5) {
        // v_local.y laeuft von -half bis +half; oben = v_fill, unten = v_fill2.
        float t = clamp(v_local.y / max(v_half.y * 2.0, 1.0) + 0.5, 0.0, 1.0);
        body = mix(v_fill2, v_fill, t);
    }

    result = over(vec4(body.rgb, body.a * inner), result);

    if (v_border_width > 0.0 && v_border_color.a > 0.0) {
        float ring = max(outer - inner, 0.0);
        result = over(vec4(v_border_color.rgb, v_border_color.a * ring), result);
    }

    if (result.a <= 0.0) {
        discard;
    }
    fragColor = result;
}
