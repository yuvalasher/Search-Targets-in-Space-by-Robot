from typing import List, Tuple
from utils import Location, CONFIG_PATH
from dataclasses import dataclass
import numpy as np
from configparser import ConfigParser

config = ConfigParser()
config.read(CONFIG_PATH)


# np.random.seed(config['UTILS'].getint(('SEED')))

@dataclass
class Area:
    """
    Class which represents Area of cells and agents which produce signals
    Area is squared, equal size of length and width (nxn)
    :param t_interval: how frequent the data generates. 0 = no updates
    """
    num_cells_axis: int
    t_interval: int
    pta: float
    alpha: float
    targets_locations: List[Location]
    cells: np.array = None

    # agent: Agent

    def __post_init__(self):
        """
        Initialize self.cells as zeros matrix NxN and assigns 1 in targets locations
        The values of the matrix (1 locations) depended on targets_locations, so doing it after initialization
        Converting List of Locations (targets' coordinates) to One Hot Encoding vector (NXN, 1)
        of "1" if cell contains target or "0" if cell do not contain
        """
        self.cells = np.zeros((self.num_cells_axis, self.num_cells_axis))
        np.put(self.cells,
               np.ravel_multi_index(np.array(self.targets_locations).T, self.cells.shape), 1)

    @staticmethod
    def calculate_distance(agent_location: Location, cells_indices: Tuple[np.array, np.array]) -> np.array:
        """
        Vectorized distance calculation
        Receiving indices of the matrix: 2 numpy arrays (horizontal & vertical) from np.indices(matrix.shape, sparse=True)
        """
        return np.sqrt(np.power(cells_indices[0] - np.array([agent_location[0]]), 2) + np.power(
            cells_indices[1] - np.array([agent_location[0]]), 2))

    @staticmethod
    def generate_targets(num_targets: int) -> List[Location]:
        """
        Randomize num_targets cells - without duplications of targets in cells
        """
        targets_locations = []
        num_generated_targets = 0
        while num_generated_targets != num_targets:
            location = tuple(np.random.choice(config['PARAMS'].getint('N'), 2))
            if location not in targets_locations:
                targets_locations.append(location)
                num_generated_targets += 1
        return targets_locations
