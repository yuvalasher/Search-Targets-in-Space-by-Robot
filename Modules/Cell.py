from dataclasses import dataclass
from utils import Location

@dataclass
class Cell:
    location: Location
    is_target: bool

