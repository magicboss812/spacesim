"""Zustände für die Aufnahmen zu 4.4 (KI-erstellt, 15.10.2026).

Fliegt die Läufe aus messlaeufe_4_4.py mit den dort gefundenen Korrekturen
noch einmal nach und hält Zeit, Startwinkel und Schiffszustand zu
vorgegebenen Zeitpunkten fest (Eingabe für aufnahmen_4_4.py).

Aufruf: python zustaende_4_4.py <spacesim-repo> <ergebnisse.json> <auftrag_in.json> <auftrag_out.json>
auftrag_in: liste mit {lauf, bezug: 'peri_J'|'abflug', offset_h, ...aufnahme-felder}
"""
import importlib.util
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO, ERG, IN, OUTF = sys.argv[1:5]
sys.argv = [sys.argv[0], REPO]
spec = importlib.util.spec_from_file_location('ml', os.path.join(HERE, 'messlaeufe_4_4.py'))
ml = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ml)

erg = json.load(open(ERG))
jobs = json.load(open(IN))
out = []
for lauf in sorted({j['lauf'] for j in jobs}):
    r = erg[lauf]
    mine = [j for j in jobs if j['lauf'] == lauf]
    for j in mine:
        if j['bezug'] == 'peri_J':
            j['_t'] = ml.T_DEP + r['jupiter']['t_peri_d'] * ml.DAY + j['offset_h'] * 3600.0
        elif j['bezug'] == 'peri_Sa':
            j['_t'] = ml.T_DEP + r['saturn']['t_peri_d'] * ml.DAY + j['offset_h'] * 3600.0
        else:
            j['_t'] = ml.T_DEP + j['offset_h'] * 3600.0
    mine.sort(key=lambda j: j['_t'])
    w, B, ship, th = ml.setup(lauf, r.get('theta0', {}).get('Saturn') if lauf in ('v1', 'v2') else None)
    J = B['Jupiter']
    var = 'direkt' if lauf == 'direkt' else ('v2' if lauf == 'v2' else 'v1')
    events = [(ml.T_DEP, 'abflug'), (ml.T_CORR1, 'k1')]
    done = set()
    k2_done = lauf in ('direkt', 'kontrolle')
    seen_peri = False
    prev_d = None
    queue = list(mine)
    while queue:
        t_next = min([e[0] for e in events if e[1] not in done] + [queue[0]['_t']])
        while w.time < t_next - 1e-6:
            if not k2_done:
                ch, d = ml.chunk_near(w, ship, J)
                if prev_d is not None and d > prev_d:
                    seen_peri = True
                prev_d = d
                if seen_peri and d > ml.R_SOI['Jupiter']:
                    ml.prograde_kick(ship, B, r['c2'], w.time)
                    k2_done = True
                    print(lauf, 'K2 bei', (w.time - ml.T_DEP) / ml.DAY, 'd', flush=True)
            else:
                ch = ml.DAY if w.time > ml.T_CORR1 else 300.0
            ml.step(w, min(ch, t_next - w.time))
        for tt, name in events:
            if name not in done and abs(w.time - tt) < 1e-3:
                if name == 'abflug':
                    ml.kick_rel(ship, B['Erde'], r['dv_E'], w.time)
                elif name == 'k1':
                    ml.prograde_kick(ship, B, r['c1'], w.time)
                done.add(name)
        while queue and abs(queue[0]['_t'] - w.time) < 1e-3:
            j = queue.pop(0)
            job = {k: v for k, v in j.items() if not k.startswith('_')}
            job.update(time=w.time, theta0=th, ship_pos=[ship.position.x, ship.position.y],
                       ship_vel=[ship.velocity.x, ship.velocity.y])
            out.append(job)
            print(lauf, 'zustand bei', (w.time - ml.T_DEP) / ml.DAY, 'd ->', job['out'], flush=True)
json.dump(out, open(OUTF, 'w'), indent=1)
