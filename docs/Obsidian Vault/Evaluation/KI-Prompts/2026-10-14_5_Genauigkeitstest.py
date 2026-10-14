"""Genauigkeitstest der Schiffsphysik (KI-erstellt, 14.10.2026).

Nachbau des RKN4-Schritts mit Schrittverdopplung aus physics/world.py
(Toleranz 0,1 m / 0,001 m/s pro Schritt, Schrittweite halbieren bei Ablehnung,
verdoppeln bei Annahme, min. 0,01 s). Ein einziger anziehender Körper,
Vergleich mit der exakten Kepler-Bahn.
Aufruf: python genauigkeitstest.py
"""
import math

G = 6.6730831e-11

def make(mu):
    def acc(x, y):
        r2 = x*x + y*y; r = math.sqrt(r2); f = -mu/(r2*r); return f*x, f*y
    def rkn4(px, py, vx, vy, h):
        a1x, a1y = acc(px, py)
        p2x = px + vx*h*0.5 + a1x*h*h*0.125; p2y = py + vy*h*0.5 + a1y*h*h*0.125
        a2x, a2y = acc(p2x, p2y); a3x, a3y = a2x, a2y
        p4x = px + vx*h + a3x*h*h*0.5; p4y = py + vy*h + a3y*h*h*0.5
        a4x, a4y = acc(p4x, p4y)
        return (px + vx*h + (a1x+a2x+a3x)*h*h/6.0, py + vy*h + (a1y+a2y+a3y)*h*h/6.0,
                vx + (a1x+2*a2x+2*a3x+a4x)*h/6.0, vy + (a1y+2*a2y+2*a3y+a4y)*h/6.0)
    def run(px, py, vx, vy, T, maxh, minh=0.01, ptol=0.1, vtol=0.001):
        t = 0.0; hint = maxh; steps = 0
        while t < T:
            h = min(hint, T - t)
            while True:
                f = rkn4(px, py, vx, vy, h)
                a = rkn4(px, py, vx, vy, h*0.5)
                b = rkn4(*a, h*0.5)
                pe = math.hypot(b[0]-f[0], b[1]-f[1]); ve = math.hypot(b[2]-f[2], b[3]-f[3])
                if (pe <= ptol and ve <= vtol) or h <= minh:
                    px, py, vx, vy = b; t += h; steps += 1; hint = min(h*2.0, maxh); break
                h *= 0.5
        return px, py, vx, vy, steps
    return run

def kepler_pos(a, e, P, t):
    M = (2*math.pi/P*t) % (2*math.pi); E = M
    for _ in range(50):
        E -= (E - e*math.sin(E) - M)/(1 - e*math.cos(E))
    return a*(math.cos(E)-e), a*math.sqrt(1-e*e)*math.sin(E)

def case(name, mass, rp, ra, T_frac_or_orbits, maxh, orbits=True):
    mu = G*mass; run = make(mu)
    a = (rp+ra)/2; e = (ra-rp)/(ra+rp); vp = math.sqrt(mu*(2/rp-1/a)); P = 2*math.pi*math.sqrt(a**3/mu)
    T = T_frac_or_orbits*P
    px, py, vx, vy, steps = run(rp, 0.0, 0.0, vp, T, maxh)
    e0 = 0.5*vp*vp - mu/rp; e1 = 0.5*(vx*vx+vy*vy) - mu/math.hypot(px, py)
    kx, ky = kepler_pos(a, e, P, T)
    vk = math.sqrt(mu*(2/math.hypot(kx, ky) - 1/a))
    print(f"{name}: {T/86400:.1f} d, {steps} Schritte, rel. Energieänderung {(e1-e0)/abs(e0):.2e}, "
          f"Halbachse {-mu/(2*e1) - a:+.2f} m, Ortsabweichung {math.hypot(px-kx, py-ky):.1f} m, "
          f"|v|-Abweichung {abs(math.hypot(vx, vy)-vk)*1e3:.4f} mm/s")

case("Kreisbahn 300 km über der Erde, 1000 Umläufe", 5.972e24, 6671e3, 6671e3, 1000, 3000.0)
case("Transferellipse Erd- zu Marsbahn, Sonne allein, max. Schritt 1e6 s", 1.989e30, 149.6e9, 227.9e9, 0.5, 1.0e6)
