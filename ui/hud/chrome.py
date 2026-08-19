"""Die formsprache der instrumententafel -- an EINER stelle.

Vier bauteile, aus denen praktisch jede flaeche des HUDs zusammengesetzt
ist. Sie stehen hier und nicht in den einzelnen widgets, weil genau das der
unterschied zwischen "eine tafel" und "viele einzelne kaesten" ist: sobald
jedes widget seinen eigenen rahmen erfindet, zerfaellt die oberflaeche in
schwebende inseln -- der auffaelligste mangel der vorherigen fassung.

1. FASE STATT RUNDUNG. Alle ecken werden unter 45 grad geschnitten, und
   meist nur EINIGE davon. Umgesetzt ueber ein negatives vorzeichen im
   eckradius (siehe shaders/ui_rect.frag).

2. DOPPELTER RAHMEN. Eine haarfeine aussenlinie, ein schmaler spalt, dann
   der kern. Eine einzelne linie um eine fuellung sieht aus wie ein
   web-panel; die doppelte kante sieht aus wie blech mit rand.

3. NOTCH-TAB. Eine kleine versal-beschriftung, die in einem eigenen
   kaestchen AUF der panelkante reitet -- ORBITAL.INFO, SNAP.CONTROL. Das
   ist das einzelne element, das den groessten teil des wiedererkennungs-
   wertes traegt. Der punkt statt eines leerzeichens gehoert dazu.

4. TEILUNG. Gestrichelte marken entlang einer kante oder eines bogens, mit
   unterschiedlichem gewicht fuer haupt- und zwischenstriche.

KOORDINATEN wie ueberall in ui/: top-down pixel. WINKEL in den
kompass-funktionen: 0 = oben, im uhrzeigersinn.
"""

import math

from ..theme import with_alpha

#: Punkt statt leerzeichen -- die schreibweise der vorlage.
DOT = '.'


def tab_text(*words):
    """('ORBITAL', 'INFO') -> 'ORBITAL.INFO'."""
    return DOT.join(str(word).upper() for word in words)


def polar(cx, cy, radius, compass_deg):
    """Kompasswinkel -> punkt in TOP-DOWN pixeln (0 = oben, im uhrzeigersinn)."""
    angle = math.radians(float(compass_deg) - 90.0)
    return (cx + math.cos(angle) * radius, cy + math.sin(angle) * radius)


def compass_to_screen(compass_deg):
    """Kompasswinkel -> der winkel, den UIDraw.arc erwartet.

    UIDraw rechnet mathematisch (0 = nach rechts, gegen den uhrzeigersinn),
    der ring in kompassgrad (0 = oben, im uhrzeigersinn) -- und dazwischen
    liegt noch der y-wechsel der top-down-konvention. Beides zusammen ist
    genau `90 - compass`, und es steht hier einmal statt in jedem aufrufer.
    """
    return (90.0 - float(compass_deg)) % 360.0


# --------------------------------------------------------------- flaechen

def frame(ctx, x, y, w, h, cut=None, fill=None, line=None, inner_line=None,
          glow_role=None, gap=None, corners=None):
    """Der doppelt gerahmte, gefaste block -- das grundbauteil der tafel.

    corners: (oben-links, oben-rechts, unten-rechts, unten-links) als bools.
             None faset alle vier. Eine ecke scharf zu lassen ist die
             uebliche art, einen tab oder einen nachbarblock anzudocken.
    """
    palette = ctx.theme.palette
    cut = ctx.px(ctx.theme.radius.cut if cut is None else cut)
    gap = ctx.px(ctx.theme.frame_gap if gap is None else gap)
    fill = palette.panel_core if fill is None else fill
    line = palette.edge if line is None else line
    inner_line = palette.edge_inner if inner_line is None else inner_line

    def radii(size):
        value = -abs(size)
        if corners is None:
            return (value,) * 4
        return tuple(value if flag else 0.0 for flag in corners)

    # Aussenlinie: nur kontur, keine fuellung -- der spalt dazwischen bleibt
    # der szene ueberlassen und traegt damit die tiefe.
    ctx.draw.rect(
        x, y, w, h, fill=None, radius=radii(cut), border_color=line,
        border_width=ctx.theme.border_width,
        shadow=ctx.theme.glow(glow_role) if glow_role else None,
        shadow_offset=(0.0, 0.0),
        shadow_softness=ctx.px(20.0) if glow_role else 0.0,
    )
    # Kern, um den spalt eingerueckt. Seine fase ist um denselben betrag
    # kleiner, sonst laufen aussen- und innenkante an den ecken auseinander.
    inner_cut = max(0.0, cut - gap)
    ctx.draw.rect(
        x + gap, y + gap, w - gap * 2.0, h - gap * 2.0,
        fill=fill, radius=radii(inner_cut),
        border_color=inner_line, border_width=ctx.theme.border_width,
    )
    return (x + gap, y + gap, w - gap * 2.0, h - gap * 2.0)


def plate(ctx, x, y, w, h, cut=None, fill=None, line=None, corners=None):
    """Einfache gefaste flaeche -- fuer knoepfe und kacheln, wo ein
    doppelrahmen zu schwer waere."""
    palette = ctx.theme.palette
    cut = ctx.px(ctx.theme.radius.cut_sm if cut is None else cut)
    value = -abs(cut)
    radius = (value,) * 4 if corners is None else tuple(
        value if flag else 0.0 for flag in corners
    )
    ctx.draw.rect(
        x, y, w, h,
        fill=palette.panel_pill if fill is None else fill,
        radius=radius,
        border_color=palette.edge if line is None else line,
        border_width=ctx.theme.border_width,
    )


def tab(ctx, text, x, y, color=None, align='left', edge='top', width=None,
        role='tab'):
    """Die beschriftung, die AUF einer panelkante reitet.

    (x, y) ist der bezugspunkt auf der kante; align sagt, ob der tab dort
    beginnt, endet oder zentriert sitzt. Der tab ist an der zur kante
    zeigenden seite SCHARF und auf der abgewandten gefast -- so wirkt er
    angesetzt und nicht aufgeklebt.

    Gibt sein rechteck zurueck, damit ein aufrufer daran weiterbauen kann.
    """
    palette = ctx.theme.palette
    color = palette.text_muted if color is None else color
    label = str(text)
    text_w, text_h = ctx.text.measure(label, role)
    pad_x = ctx.px(7)
    height = text_h + ctx.px(4)
    box_w = (text_w + pad_x * 2.0) if width is None else ctx.px(width)

    left = float(x)
    if align == 'center':
        left -= box_w * 0.5
    elif align == 'right':
        left -= box_w

    # AUSSERHALB der kante, nicht darauf: (x, y) ist die kante selbst, der
    # tab sitzt daneben. Legte man ihn nach innen, verdeckte er die erste
    # zeile des blocks -- bei der zeitraffer-leiste waren das die beiden
    # ersten stufen.
    top = float(y) - height if edge == 'top' else float(y)
    cut = ctx.px(ctx.theme.radius.cut_sm)
    # Zur kante hin scharf, von ihr weg gefast.
    if edge == 'top':
        radius = (-cut, -cut, 0.0, 0.0)
    else:
        radius = (0.0, 0.0, -cut, -cut)

    ctx.draw.rect(
        left, top, box_w, height, fill=palette.panel_popup, radius=radius,
        border_color=palette.edge, border_width=ctx.theme.border_width,
    )
    ctx.text.draw(label, left + box_w * 0.5, top + height * 0.5, role=role,
                  color=color, align='center', valign='middle')
    return (left, top, box_w, height)


# ---------------------------------------------------------------- teilung

def ruler(ctx, x, y, length, color, count=12, major_every=4, vertical=False,
          major=7.0, minor=4.0, width=1.0):
    """Gerade teilung entlang einer kante."""
    if count <= 1:
        return
    step = float(length) / float(count - 1)
    for index in range(count):
        is_major = (index % max(1, major_every)) == 0
        size = ctx.px(major if is_major else minor)
        shade = color if is_major else with_alpha(color, 0.45)
        if vertical:
            ctx.draw.rect(x, y + index * step, size, ctx.px(width), fill=shade)
        else:
            ctx.draw.rect(x + index * step, y, ctx.px(width), size, fill=shade)


def arc_ruler(ctx, cx, cy, radius, color, start_deg, end_deg, count=13,
              major_every=4, major=9.0, minor=5.0, width=1.4, inward=True):
    """Teilung entlang eines kreisbogens, in kompassgrad.

    inward=True zieht die striche nach innen (zur ringmitte), sonst nach
    aussen. Zwei ruler auf demselben radius, einer nach innen und einer
    nach aussen, ergeben die doppelte teilung der vorlage.
    """
    if count <= 1:
        return
    span = (float(end_deg) - float(start_deg)) / float(count - 1)
    for index in range(count):
        deg = float(start_deg) + span * index
        is_major = (index % max(1, major_every)) == 0
        size = ctx.px(major if is_major else minor)
        shade = color if is_major else with_alpha(color, 0.45)
        outer = float(radius)
        inner = outer - size if inward else outer + size
        x0, y0 = polar(cx, cy, outer, deg)
        x1, y1 = polar(cx, cy, inner, deg)
        ctx.draw.line(x0, y0, x1, y1, shade,
                      width=max(1.0, ctx.px(width if is_major else width * 0.8)))


# ------------------------------------------------------------ segmentbogen

def segment_arc(ctx, cx, cy, radius, thickness, start_deg, end_deg, fraction,
                color, count=14, gap_deg=1.6, empty=None, bidirectional=False):
    """Ein bogen aus EINZELNEN zellen -- die anzeigeform der vorlage.

    Ein durchgezogener balken sagt "prozent", ein zellenbogen sagt "stufe":
    er ist auf einen blick abzaehlbar und aendert sich sichtbar sprunghaft.
    Genau deshalb benutzt die vorlage ihn fuer schub und raffung.

    fraction laeuft bei bidirectional=False von 0 (nichts) bis 1 (voll) und
    fuellt vom ANFANG des bogens her. Bei bidirectional=True laeuft sie von
    -1 bis +1 und fuellt von der MITTE aus nach beiden seiten -- so zeigt
    ein bogen ein vorzeichenbehaftetes mass (steigen/sinken), ohne dass man
    die nulllage suchen muss.
    """
    palette = ctx.theme.palette
    empty = palette.edge_inner if empty is None else empty
    count = max(1, int(count))
    span = (float(end_deg) - float(start_deg)) / count
    cell = abs(span) - float(gap_deg)
    if cell <= 0.0:
        cell = abs(span) * 0.7

    if fraction is None:
        lit_lo = lit_hi = -1
    elif bidirectional:
        middle = count * 0.5
        reach = middle * max(-1.0, min(1.0, float(fraction)))
        lit_lo = int(math.floor(min(middle, middle + reach)))
        lit_hi = int(math.ceil(max(middle, middle + reach))) - 1
    else:
        lit_lo = 0
        lit_hi = int(round(count * max(0.0, min(1.0, float(fraction))))) - 1

    for index in range(count):
        deg = float(start_deg) + span * index
        lo = min(deg, deg + span)
        lit = lit_lo <= index <= lit_hi
        ctx.draw.arc(
            cx, cy, float(radius), float(thickness),
            color if lit else empty,
            compass_to_screen(lo + abs(span)), cell,
        )


def bar_cells(ctx, x, y, w, h, fraction, color, count=10, gap=2.0,
              empty=None, vertical=False):
    """Dasselbe geradlinig: ein balken aus einzelnen zellen."""
    palette = ctx.theme.palette
    empty = palette.edge_inner if empty is None else empty
    count = max(1, int(count))
    gap_px = ctx.px(gap)
    lit = -1 if fraction is None else int(
        round(count * max(0.0, min(1.0, float(fraction))))
    ) - 1
    cut = -ctx.px(2.0)
    if vertical:
        cell = (h - gap_px * (count - 1)) / count
        for index in range(count):
            # Von UNTEN fuellen: ein senkrechter pegel waechst nach oben.
            top = y + h - (index + 1) * cell - index * gap_px
            ctx.draw.rect(x, top, w, cell,
                          fill=color if index <= lit else empty,
                          radius=(cut, 0.0, cut, 0.0))
    else:
        cell = (w - gap_px * (count - 1)) / count
        for index in range(count):
            left = x + index * (cell + gap_px)
            ctx.draw.rect(left, y, cell, h,
                          fill=color if index <= lit else empty,
                          radius=(cut, 0.0, cut, 0.0))
