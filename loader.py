import json
from vec import Vec2
from bodies import body, schiff

class SystemLoader:
    """Lädt ein Planetensystem aus einer JSON-Datei."""
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None
    
    def load(self):
        """Lädt die JSON-Datei und gibt eine Liste von body-Objekten zurück."""
        with open(self.filepath, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        bodies = []
        body_refs = {}  # Für Parent-Referenzen (Monde)
        
        # Erster Durchlauf: Alle Körper erstellen
        for entry in self.data.get("bodies", []):
            b = self._create_body(entry)
            bodies.append(b)
            body_refs[entry["name"]] = b
        
        # Zweiter Durchlauf: Parent-Referenzen auflösen (is_moon_of)
        for i, entry in enumerate(self.data.get("bodies", [])):
            if "is_moon_of" in entry and entry["is_moon_of"]:
                bodies[i].is_moon_of = body_refs.get(entry["is_moon_of"])
                bodies[i].scripted_orbit = True
        
        return bodies
    
    def hex_to_rgb(self, hex_color):
        """Wandelt Hex-String '#RRGGBB' in RGB-Tupel (r, g, b) um."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def _create_body(self, entry):
        """Erstellt ein einzelnes body- oder schiff-Objekt aus einem JSON-Eintrag."""
        
        # Wenn is_ship=True, verwende die schiff-Klasse
        if entry.get("is_ship", False):
            print(f"LOADER: Creating ship {entry['name']} with position={entry['position']}, velocity={entry.get('velocity', 'NOT FOUND')}")
            return schiff(
                name=entry["name"],
                position=Vec2(entry["position"][0], entry["position"][1]),
                velocity=Vec2(entry["velocity"][0], entry["velocity"][1]) if "velocity" in entry else Vec2(0, 0),
                color=self.hex_to_rgb(entry["color"]) if "color" in entry else (255, 255, 255)
            )
        
        # Ansonsten erstelle einen normalen body
        return body(
            name=entry["name"],
            mass=entry["mass"],
            radius=entry["radius"],
            position=Vec2(entry["position"][0], entry["position"][1]),
            velocity=Vec2(entry["velocity"][0], entry["velocity"][1]),
            fixed=entry.get("fixed", False),
            semi_major_axis=entry.get("semi_major_axis") if "semi_major_axis" in entry else 0.0,
            eccentricity=entry.get("eccentricity") if "eccentricity" in entry else 0.0,
            theta0=entry.get("theta0", 0.0) if "theta0" in entry else 0.0,
            is_moon_of=None,
            is_ship=entry.get("is_ship", False) if "is_ship" in entry else False,
            color=self.hex_to_rgb(entry["color"]) if "color" in entry else (255, 255, 255),
            atmosphere_color=self.hex_to_rgb(entry["atmosphere_color"]) if "atmosphere_color" in entry else (255, 255, 255),
            has_atmosphere=entry.get("has_atmosphere", False),
            atmos_density=entry.get("atmos_density", 0.0),
            light_intensity=entry.get("light_intensity", 0.0)
        )
