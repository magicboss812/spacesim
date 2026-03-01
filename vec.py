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
        """Vector subtraction."""
        if isinstance(other, Vec2):
            return Vec2(self.x - other.x, self.y - other.y)
        return NotImplemented
    
    def __add__(self, other):
        """Vector addition."""
        if isinstance(other, Vec2):
            return Vec2(self.x + other.x, self.y + other.y)
        return NotImplemented
    
    def __mul__(self, scalar):
        """Scalar multiplication (self * scalar)."""
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        s = float(scalar)
        return Vec2(self.x * s, self.y * s)
    def __iadd__(self, other):
        """In-place vector addition."""
        if isinstance(other, Vec2):
            self.x += other.x
            self.y += other.y
            return self
        return NotImplemented
    
    def __isub(self, other):
        """In-place vector subtraction."""
        if isinstance(other, Vec2):
            self.x -= other.x
            self.y -= other.y
            return self
        return NotImplemented
    
    def __imul__(self, scalar):
        """In-place scalar multiplication."""
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        s = float(scalar)
        self.x *= s
        self.y *= s
        return self

    def __rmul__(self, scalar):
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        """Scalar multiplication (scalar * self)."""
        s = float(scalar)
        return Vec2(self.x * s, self.y * s)
    
    def __truediv__(self, scalar):
        """Scalar division."""
        s = float(scalar)
        if s == 0.0:
            raise ValueError("Division by zero")
        return Vec2(self.x / s, self.y / s)
    
    def __neg__(self):
        """Negation."""
        return Vec2(-self.x, -self.y)
    
    def __repr__(self):
        return f"Vec2({self.x}, {self.y})"
    
    def magnitude_squared(self):
        """Squared magnitude for performance-critical code."""
        return self.x * self.x + self.y * self.y
    
    def magnitude(self):
        """Magnitude (length) of the vector."""
        return math.sqrt(self.magnitude_squared())
    
    def normalize(self):
        """Return normalized (unit) vector."""
        m2 = self.magnitude_squared()
        if m2 < 1e-30:
            return Vec2(0.0, 0.0)
        mag = math.sqrt(m2)
        return Vec2(self.x / mag, self.y / mag)
    
    def dot(self, other):
        """Dot product."""
        if isinstance(other, Vec2):
            return self.x * other.x + self.y * other.y
        return NotImplemented
    
    def copy(self):
        """Create a copy of this vector."""
        return Vec2(self.x, self.y)
    
    def to_tuple(self):
        """Convert to (x, y) tuple."""
        return (self.x, self.y)
    
    @staticmethod
    def from_tuple(t):
        """Create Vec2 from tuple or list."""
        return Vec2(float(t[0]), float(t[1]))
    
    def distance_squared_to(self, other):
        """Squared distance to another vector."""
        if isinstance(other, Vec2):
            dx = self.x - other.x
            dy = self.y - other.y
            return dx * dx + dy * dy
        return NotImplemented
    
    def clear(self):
        """Reset vector to zero."""
        self.x = 0.0
        self.y = 0.0
        return self
    def set(self, x, y):
        """Set vector components."""
        self.x = float(x)
        self.y = float(y)
        return self

def vec(x, y):
    """Convenience function to create Vec2."""
    return Vec2(x, y)
