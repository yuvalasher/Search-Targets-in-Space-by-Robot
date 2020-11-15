from typing import List, Tuple, Union
from utils import Location, CONFIG_PATH
from Area import Area
import Agent
from dataclasses import dataclass
from scipy.stats import bernoulli
from pathlib import Path
import numpy as np
import configparser
from configparser import ConfigParser
from tabulate import tabulate

config = ConfigParser()
config.read(CONFIG_PATH)
np.random.seed(config['UTILS'].getint(('SEED')))


@dataclass
class DataGenerator:
    """
    Responsible to generate data for all the variables
    :param path: The Location of the data will be written to
    """

    @staticmethod
    def param_validation(params: configparser.SectionProxy) -> None:
        """
        Validation of the params definitions from config.cfg. validates relations between objects, as the agent or the targets are in the Area
        If not valid, raise InvalidParamsException
        """
        N = params.getint('N')
        num_cells = N * N
        target_locations = eval(params['TARGET_LOCATIONS'])
        agent_position = eval(params['AGENT_POSITION'])

        if target_locations is None:
            assert params.getint('NUM_TARGETS') <= num_cells
        assert len(target_locations) <= num_cells
        assert all([target_location[0] <= N - 1 and target_location[1] <= N - 1 for target_location in target_locations])
        if isinstance(agent_position, tuple):
            assert agent_position[0] <= N - 1
            assert agent_position[1] <= N - 1
        assert 0 < params.getfloat('pta') <= 1
        assert 0 < params.getfloat('alpha') <= 1
        assert 0 < params.getfloat('INITIAL_PRIOR_P') < 1

    @staticmethod
    def simulate_data(area: Area, agent: Agent) -> Tuple[np.array]:
        """
        Running the simulation of the data - based on values of pta, alpha,
        a (alarm sent), x (alarm received), s (is exists target - area.targets)
        Generator of events
        """
        a_prob_matrix = np.where(area.cells == 1, area.pta, area.alpha * area.pta)
        a_matrix = np.vectorize(DataGenerator.realization_by_bernoulli)(a_prob_matrix)
        x_prob_matrix = np.where(a_matrix == 0, 0, agent.calculate_probability_of_agent_receiving_signal_from_cell(area=area))
        x_matrix = np.vectorize(DataGenerator.realization_by_bernoulli)(x_prob_matrix)
        yield a_matrix, x_matrix, area.cells

    @staticmethod
    def realization_by_bernoulli(p: float, num_values: Union[None, int] = None) -> Union[int, List[int]]:
        """
        Generates num_values data points of the Bernoulli distribution with probability p
        Used for experiment of existence of the agents in the Area, sending true alarm (pta) & false alarm(pfa)
        if num_values not supplied, will produce one sample as int
        :return: List with length of num_values with values 0 or 1 by the experiment result
        """
        if num_values:
            return bernoulli.rvs(p, size=num_values)
        return bernoulli.rvs(p)

    @staticmethod
    def tabulate_matrix(matrix: np.array):
        print(tabulate(matrix, headers=[*list(range(config['PARAMS'].getint('N')))], tablefmt='github', showindex=True))
