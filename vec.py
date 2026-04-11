import math

G = 6.6730831e-11

# Alle Vektoroperatoren in einer Class Datei, damit ich sie 1. von überall aufrufen kann und 2. die Performance verbessern kann
# Self und oder repräsentieren dabei, dass ich soz. selbst die Werte der Vektoren bestimme

class Vec2:
    __slots__ = ('x', 'y')
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)
        
    def __sub__(self, other):
        """vektor-subtraktion."""
        if isinstance(other, Vec2):
            return Vec2(self.x - other.x, self.y - other.y)
        return NotImplemented
    
    def __add__(self, other):
        """vektor-addition."""
        if isinstance(other, Vec2):
            return Vec2(self.x + other.x, self.y + other.y)
        return NotImplemented
    
    def __mul__(self, scalar):
        """skalare multiplikation (self * scalar)."""
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        s = float(scalar)
        return Vec2(self.x * s, self.y * s)
    def __iadd__(self, other):
        """in-place vektor-addition."""
        if isinstance(other, Vec2):
            self.x += other.x
            self.y += other.y
            return self
        return NotImplemented
    
    def __isub(self, other):
        """in-place vektor-subtraktion."""
        if isinstance(other, Vec2):
            self.x -= other.x
            self.y -= other.y
            return self
        return NotImplemented
    
    def __imul__(self, scalar):
        """in-place skalare multiplikation."""
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        s = float(scalar)
        self.x *= s
        self.y *= s
        return self

    def __rmul__(self, scalar):
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        """skalare multiplikation (scalar * self)."""
        s = float(scalar)
        return Vec2(self.x * s, self.y * s)
    
    def __truediv__(self, scalar):
        """skalare division."""
        s = float(scalar)
        if s == 0.0:
            raise ValueError("Division by zero")
        return Vec2(self.x / s, self.y / s)
    
    def __neg__(self):
        """negation."""
        return Vec2(-self.x, -self.y)
    
    def __repr__(self):
        return f"Vec2({self.x}, {self.y})"
    
    def magnitude_squared(self):
        """quadratische länge für performance-kritischen code."""
        return self.x * self.x + self.y * self.y
    
    def magnitude(self):
        """betrag (länge) des vektors."""
        return math.sqrt(self.magnitude_squared())
    
    def normalize(self):
        """gibt normalisierten (einheits-)vektor zurück."""
        m2 = self.magnitude_squared()
        if m2 < 1e-30:
            return Vec2(0.0, 0.0)
        mag = math.sqrt(m2)
        return Vec2(self.x / mag, self.y / mag)
    
    def dot(self, other):
        """skalarprodukt."""
        if isinstance(other, Vec2):
            return self.x * other.x + self.y * other.y
        return NotImplemented
    
    def copy(self):
        """erstellt eine kopie dieses vektors."""
        return Vec2(self.x, self.y)
    
    def to_tuple(self):
        """in (x, y)-tuple umwandeln."""
        return (self.x, self.y)
    
    @staticmethod
    def from_tuple(t):
        """erstellt Vec2 aus tuple oder liste."""
        return Vec2(float(t[0]), float(t[1]))
    
    def distance_squared_to(self, other):
        """quadratische distanz zu einem anderen vektor."""
        if isinstance(other, Vec2):
            dx = self.x - other.x
            dy = self.y - other.y
            return dx * dx + dy * dy
        return NotImplemented
    
    def clear(self):
        """vektor auf null zurücksetzen."""
        self.x = 0.0
        self.y = 0.0
        return self
    def set(self, x, y):
        """vektor-komponenten setzen."""
        self.x = float(x)
        self.y = float(y)
        return self

def vec(x, y):
    """bequemlichkeitsfunktion zum erstellen von Vec2."""
    return Vec2(x, y)
