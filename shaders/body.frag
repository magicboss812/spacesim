#version 120

// Fragment Shader für Body-Rendering mit SDF
// Rendert: Kreis, Atmosphäre (Gradient), Glow

uniform vec2 u_center;          // Zentrum in Bildschirmkoordinaten
uniform float u_radius;         // Radius in Pixeln
uniform vec3 u_color;           // Planetenfarbe (RGB, normalisiert 0-1)
uniform float u_atmos_density;  // Atmosphärendichte (0-1000)
uniform float u_light_intensity; // Glow-Intensität (0-1000)
uniform vec2 u_screen_size;     // Bildschirmgröße für Normalisierung

varying vec2 v_texcoord;
varying vec2 v_position;

// SDF für Kreis
float sd_circle(vec2 p, vec2 center, float r) {
    return length(p - center) - r;
}

// Smooth minimum für weiche Übergänge
float smooth_circle(float d, float edge) {
    return 1.0 - smoothstep(-edge, edge, d);
}

void main() {
    // Position in Pixel-Koordinaten
    vec2 pixel_pos = (v_texcoord + 1.0) * 0.5 * u_screen_size;
    
    // Distanz zum Zentrum
    float dist = length(pixel_pos - u_center);
    
    // Normalisierte Distanz (0 = Zentrum, 1 = Radius)
    float normalized_dist = dist / u_radius;
    
    // === 1. PLANETEN-KREIS (SDF) ===
    float circle_dist = sd_circle(pixel_pos, u_center, u_radius);
    float circle_alpha = smooth_circle(circle_dist, 1.5);
    
    // === 2. ATMOSPHÄRE (Gradient) ===
    float atmos_alpha = 0.0;
    vec3 atmos_color = u_color;
    
    if (u_atmos_density > 0.0) {
        // Atmosphären-Radius (1.5x Planetenradius)
        float atmos_radius = u_radius * 1.5;
        
        // Gradient: Dicht am Planeten, transparent am Rand
        float atmos_dist = dist - u_radius;
        float atmos_range = atmos_radius - u_radius;
        
        // Normalisierter Gradient (0 = Planetenrand, 1 = Atmosphärenrand)
        float gradient = clamp(atmos_dist / atmos_range, 0.0, 1.0);
        
        // Dichte-basierte Stärke (0-1000 -> 0-1)
        float density_factor = u_atmos_density / 1000.0;
        
        // Atmosphären-Alpha: Hoch am Planetenrand, niedrig außen
        atmos_alpha = (1.0 - gradient) * density_factor * 0.6;
        
        // Atmosphärenfarbe: Leicht aufgehellt
        atmos_color = u_color + vec3(0.1, 0.1, 0.1);
    }
    
    // === 3. GLOW (Radial) ===
    float glow_alpha = 0.0;
    vec3 glow_color = u_color;
    
    if (u_light_intensity > 0.0) {
        // Glow-Radius (2x Planetenradius für weiten Glow)
        float glow_radius = u_radius * 2.0;
        
        // Exponentieller Abfall
        float glow_dist = dist / glow_radius;
        float glow_factor = u_light_intensity / 1000.0;
        
        // Glow mit exponentiellem Abfall
        glow_alpha = exp(-glow_dist * 3.0) * glow_factor * 0.8;
        
        // Glow-Farbe: Planetenfarbe
        glow_color = u_color;
    }
    
    // === 4. COMPOSITING ===
    vec3 final_color = u_color;
    float final_alpha = 0.0;
    
    // Planet (nur innerhalb des Radius)
    if (dist <= u_radius) {
        final_color = u_color;
        final_alpha = circle_alpha;
    }
    
    // Atmosphäre (außerhalb des Planeten, inkl. Randbereich)
    // Verwende >= statt > um 1-Pixel-Lücke zu vermeiden
    if (dist >= u_radius - 1.0 && atmos_alpha > 0.01) {
        final_color = mix(final_color, atmos_color, atmos_alpha);
        final_alpha = max(final_alpha, atmos_alpha);
    }
    
    // Glow (über alles)
    if (glow_alpha > 0.01) {
        final_color = mix(final_color, glow_color, glow_alpha * 0.5);
        final_alpha = max(final_alpha, glow_alpha);
    }
    
    // Alpha-Multiplikation für Blending
    gl_FragColor = vec4(final_color, final_alpha);
}
