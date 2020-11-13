from typing import List, Tuple
from utils import Location, CONFIG_PATH
from Cell import Cell
from dataclasses import dataclass
import numpy as np
from configparser import ConfigParser

config = ConfigParser()
config.read(CONFIG_PATH)
np.random.seed(config['UTILS'].getint(('SEED')))

@dataclass
class Area:
    """
    Class which represents Area of cells and agents which produce signals
    Area is squared, equal size of length and width (nxn)
    :param t_interval: how frequent the data generates. 0 = no updates
    """
    num_cells: int
    t_interval: int
    pta: float
    alpha: float
    cells: List[Cell]
    # agent: Agent(location=agent_location, )

    @staticmethod
    def calculate_distance(a_location: Location, b_location: Location):
        return np.sqrt(np.sum(
            [np.power(a_coordinate - b_coordinate, 2) for a_coordinate, b_coordinate in zip(a_location, b_location)]))

    @staticmethod
    def generate_targets(cells_locations: List[Location], num_targets: int) -> List[Location]:
        """
        Randomize num_targets cells
        """
        return [(np.random.randint(len(cells_locations)), np.random.randint(len(cells_locations))) for _ in range(num_targets)]