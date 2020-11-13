from typing import List, Union
from utils import Location, CONFIG_PATH, CSV_COLUMNS
from Area import Area
from Agent import Agent
from dataclasses import dataclass
from scipy.stats import bernoulli
from pathlib import Path
import numpy as np
import pandas as pd
import configparser
from configparser import ConfigParser
from tqdm import tqdm

config = ConfigParser()
config.read(CONFIG_PATH)
np.random.seed(config['UTILS'].getint(('SEED')))


@dataclass
class DataGenerator:
    """
    Responsible to generate data for all the variables
    :param path: The Location of the data will be written to
    """
    csv_path: Path

    @staticmethod
    def param_validation(params: configparser.SectionProxy) -> None:
        """
        Validation of the params definitions from config.cfg. validates relations between objects, as the agent or the targets are in the Area
        If not valid, raise InvalidParamsException
        """
        pass

    def simulate_data(self, area: Area, agent: Agent, num_rounds: int) -> None:
        """
        Running the simulation of the data - based on values of pta, alpha,
        Writing to a csv file the simulation by values:
        t, cell_id, a (alarm sent), x (alarm received), s (is exists target - area.targets)
        """
        df = pd.DataFrame(columns=CSV_COLUMNS)
        for tick in tqdm(range(num_rounds)):
            for cell in area.cells:
                p = area.pta
                if not cell.is_target:
                    p *= area.alpha
                a = self.realization_by_bernoulli(p)
                if a:
                    x = self.realization_by_bernoulli(
                        p=agent.calculate_probability_of_agent_receiving_signal_from_cell(area=area,
                            target_location=cell.location))
                else:
                    x = 0
                df = df.append({'t': tick,
                                'cell_location': cell.location,
                                'a': a,
                                'x': x,
                                's': int(cell.is_target)}, ignore_index=True)

            if tick % 100 == 0 or tick == num_rounds:
                df.to_csv(self.csv_path, index=False)

    def realization_by_bernoulli(self, p: float, num_values: Union[None, int] = None) -> Union[int, List[int]]:
        """
        Generates num_values data points of the Bernoulli distribution with probability p
        Used for experiment of existence of the agents in the Area, sending true alarm (pta) & false alarm(pfa)
        if num_values not supplied, will produce one sample as int
        :return: List with length of num_values with values 0 or 1 by the experiment result
        """
        if num_values:
            return bernoulli.rvs(p, size=num_values)
        return bernoulli.rvs(p)
