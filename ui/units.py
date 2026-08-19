"""Kompakte zahlenformatierung fuer das HUD.

Reine funktionen: kein GL, kein pygame, kein zustand. Damit sind sie ohne
fenster testbar -- das ist der grund, warum sie ein eigenes modul sind und
nicht in der zeichenschicht liegen.

INVARIANTE: das projekt rechnet ausschliesslich in SI (meter, sekunden,
kilogramm). Diese funktionen formatieren nur FUER DIE ANZEIGE. Keine der
hier erzeugten einheiten (AU, tage, erdmassen) darf jemals in gespeicherten
zustand zurueckfliessen.
"""

import math

_MINUTE_S = 60.0
_HOUR_S = 3600.0
_DAY_S = 86400.0
_YEAR_S = 365.25 * _DAY_S


def _finite(value):
    """Wandelt beliebige eingaben in einen endlichen float oder None."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _scaled(value, digits, ladder, base_unit):
    """Waehlt die groesste einheit der leiter, die value >= 1 laesst."""
    magnitude = abs(value)
    for threshold, suffix in ladder:
        if magnitude >= threshold:
            return f"{value / threshold:.{digits}f}{suffix}"
    return f"{value:.0f}{base_unit}"


_DISTANCE_LADDER = (
    (1e12, 'Tm'),
    (1e9, 'Gm'),
    (1e6, 'Mm'),
    (1e3, 'km'),
)

_SPEED_LADDER = (
    (1e6, 'Mm/s'),
    (1e3, 'km/s'),
)

_MASS_LADDER = (
    (1e12, 'Gt'),
    (1e9, 'Mt'),
    (1e6, 'kt'),
    (1e3, 't'),
)


def distance(meters, digits=2, placeholder='--'):
    """Strecke in m / km / Mm / Gm / Tm."""
    value = _finite(meters)
    if value is None:
        return placeholder
    return _scaled(value, digits, _DISTANCE_LADDER, 'm')


def altitude(meters, placeholder='--'):
    """Hoehe ueber grund. Negative werte sind gueltig (unter der oberflaeche)."""
    value = _finite(meters)
    if value is None:
        return placeholder
    sign = '-' if value < 0 else ''
    return sign + distance(abs(value), placeholder=placeholder)


def speed(meters_per_second, digits=2, placeholder='--'):
    """Geschwindigkeit in m/s / km/s / Mm/s."""
    value = _finite(meters_per_second)
    if value is None:
        return placeholder
    magnitude = abs(value)
    for threshold, suffix in _SPEED_LADDER:
        if magnitude >= threshold:
            return f"{value / threshold:.{digits}f}{suffix}"
    return f"{value:.1f}m/s"


def delta_v(meters_per_second, placeholder='--'):
    """Delta-v. Bewusst IMMER in m/s -- das ist die einheit, in der man
    manoever plant, und ein umschalten auf km/s macht sie unvergleichbar."""
    value = _finite(meters_per_second)
    if value is None:
        return placeholder
    if abs(value) >= 1e4:
        return f"{value:,.0f}m/s".replace(',', ' ')
    return f"{value:.1f}m/s"


def duration(seconds, placeholder='--'):
    """Zeitspanne als '1y 23d 04:05:06'.

    Jahre und tage nur wenn sie ungleich null sind; stunden/minuten/sekunden
    immer zweistellig, damit die spalte beim mitlaufen nicht springt.
    """
    value = _finite(seconds)
    if value is None:
        return placeholder

    sign = '-' if value < 0 else ''
    remaining = abs(value)

    years = int(remaining // _YEAR_S)
    remaining -= years * _YEAR_S
    days = int(remaining // _DAY_S)
    remaining -= days * _DAY_S
    hours = int(remaining // _HOUR_S)
    remaining -= hours * _HOUR_S
    minutes = int(remaining // _MINUTE_S)
    remaining -= minutes * _MINUTE_S
    secs = int(remaining)

    clock = f"{hours:02d}:{minutes:02d}:{secs:02d}"
    if years:
        return f"{sign}{years}y {days}d {clock}"
    if days:
        return f"{sign}{days}d {clock}"
    return f"{sign}{clock}"


def countdown(seconds, placeholder='--'):
    """Wie duration, aber mit fuehrendem 'T-' / 'T+' fuer manoever-timer."""
    value = _finite(seconds)
    if value is None:
        return placeholder
    marker = 'T-' if value >= 0 else 'T+'
    return marker + duration(abs(value))


def mass(kilograms, digits=2, placeholder='--'):
    """Masse in kg / t / kt / Mt / Gt, darueber wissenschaftlich in kg.

    Bewusst KEINE erdmassen: das uebliche zeichen dafuer (M⊕, U+2295) fehlt
    in vielen UI-schriften und wuerde als kaestchen erscheinen. Ausserdem
    bleibt die anzeige so durchgehend SI -- passend zur projekt-invariante.
    """
    value = _finite(kilograms)
    if value is None:
        return placeholder
    if abs(value) >= 1e15:
        return f"{value:.{digits}e}kg"
    return _scaled(value, digits, _MASS_LADDER, 'kg')


def angle(radians, digits=1, placeholder='--'):
    """Winkel in grad, normalisiert auf [0, 360)."""
    value = _finite(radians)
    if value is None:
        return placeholder
    degrees = math.degrees(value) % 360.0
    return f"{degrees:.{digits}f}°"


def signed_angle(radians, digits=1, placeholder='--'):
    """Winkel in grad, normalisiert auf (-180, 180] -- fuer abweichungen."""
    value = _finite(radians)
    if value is None:
        return placeholder
    degrees = (math.degrees(value) + 180.0) % 360.0 - 180.0
    return f"{degrees:+.{digits}f}°"


def eccentricity(value, placeholder='--'):
    """Exzentrizitaet: dimensionslos, drei nachkommastellen."""
    number = _finite(value)
    if number is None:
        return placeholder
    return f"{number:.3f}"


def scientific(value, digits=2, placeholder='--'):
    """Wissenschaftliche notation -- fuer kamera-skala und rohwerte."""
    number = _finite(value)
    if number is None:
        return placeholder
    return f"{number:.{digits}e}"


def time_warp(factor, placeholder='--'):
    """Zeitraffer-faktor als '1x' / '1 000x' / '1.0e6x'."""
    number = _finite(factor)
    if number is None:
        return placeholder
    if abs(number) >= 1e5:
        return f"{number:.1e}x"
    if number == int(number):
        return f"{int(number):,}x".replace(',', ' ')
    return f"{number:.1f}x"


# --------------------------------------------------- wert und einheit getrennt

def _split_scaled(value, digits, ladder, base_unit, base_digits=0):
    magnitude = abs(value)
    for threshold, suffix in ladder:
        if magnitude >= threshold:
            return (f"{value / threshold:.{digits}f}", suffix)
    return (f"{value:.{base_digits}f}", base_unit)


def split_distance(meters, digits=2, placeholder='--'):
    """('5.36', 'Gm') statt '5.36Gm'.

    Die instrumententafel setzt zahl und einheit in VERSCHIEDENEN groessen
    -- ein 25-px-messwert neben einer 10-px-einheit. Genau dieser kontrast
    macht die anzeige ablesbar; zusammengesetzt liesse er sich nicht
    herstellen, ohne die zeichenkette wieder auseinanderzunehmen.
    """
    value = _finite(meters)
    if value is None:
        return (placeholder, '')
    return _split_scaled(value, digits, _DISTANCE_LADDER, 'm')


def split_speed(meters_per_second, digits=2, placeholder='--'):
    """('11.69', 'km/s') statt '11.69km/s'."""
    value = _finite(meters_per_second)
    if value is None:
        return (placeholder, '')
    return _split_scaled(value, digits, _SPEED_LADDER, 'm/s', base_digits=0)
