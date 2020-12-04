from typing import List, Tuple, Union, Any
from utils import Location, CONFIG_PATH
from utils import save_hdf5_to_file, load_hdf5_file
from Area import Area
import Agent
from dataclasses import dataclass
from scipy.stats import bernoulli
from pathlib import Path
import numpy as np
import configparser
from configparser import ConfigParser
from tabulate import tabulate
from tqdm import tqdm

config = ConfigParser()
config.read(CONFIG_PATH)


# np.random.seed(config['UTILS'].getint(('SEED')))

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
        else:
            assert len(target_locations) <= num_cells
            assert all(
                [target_location[0] <= N - 1 and target_location[1] <= N - 1 for target_location in target_locations])
        if isinstance(agent_position, tuple):
            assert agent_position[0] <= N - 1
            assert agent_position[1] <= N - 1
        assert 0 < params.getfloat('pta') <= 1
        assert 0 < params.getfloat('alpha') <= 1
        assert 0 < params.getfloat('INITIAL_PRIOR_P') < 1

    @staticmethod
    def _calculate_syntatic_data(area: Area, agent: Agent):
        """
        Running the simulation of the data - based on values of pta, alpha,
        a (alarm sent), x (alarm received), s (is exists target - area.targets)
        """
        a_prob_matrix = np.where(area.cells == 1, area.pta, area.alpha * area.pta)
        a_matrix = np.vectorize(DataGenerator.realization_by_bernoulli)(a_prob_matrix)
        x_prob_matrix = np.where(a_matrix == 0, 0,
                                 agent.calculate_probability_of_agent_receiving_signal_from_cell(area=area))
        x_matrix = np.vectorize(DataGenerator.realization_by_bernoulli)(x_prob_matrix)
        return a_matrix, x_matrix, area.cells

    @staticmethod
    def simulate_data_generator(area: Area, agent: Agent) -> Tuple[np.array, np.array, np.array]:
        """
        Generator of events a, x, cells (placing targets and non-targets)
        """
        a_matrix, x_matrix, area.cells = DataGenerator._calculate_syntatic_data(agent=agent, area=area)
        yield a_matrix, x_matrix, area.cells

    @staticmethod
    def simulate_agent_receive_signals(area: Area, agent: Agent) -> Tuple[np.array, np.array]:
        """
        Generator of events a, x, cells (placing targets and non-targets)
        return x matrix and distance matrix from the agent location as vectors (N * N,)
        """
        a_matrix, x_matrix, area.cells = DataGenerator._calculate_syntatic_data(agent=agent, area=area)
        return x_matrix.reshape(-1, 1)

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

    def _get_generated_params_for_sequence(self) -> Tuple[Any, Area, int, List[Location]]:
        """
        Used for generating the parameters for the X data generation as agent location, number of targets,
        location of the targets, etc.
        :return: agent, area, num_of_targets, targets_location
        """
        params = config["PARAMS"]
        targets_locations = Area.generate_targets(params.getint('NUM_TARGETS'))
        area = Area(num_cells_axis=params.getint('N'), t_interval=params.getint('T_INTERVAL'),
                    pta=params.getfloat('pta'), alpha=params.getfloat('alpha'),
                    targets_locations=targets_locations)

        agent = Agent.Agent(current_location=eval(params['AGENT_POSITION']),
                            lambda_strength=params.getint('LAMBDA_SENSOR'),
                            p_S=Agent.Agent.get_p_S_from_initial_prior(prior=params.getfloat('INITIAL_PRIOR_P'),
                                                                       area=area),
                            entropy_updates=[], information_gain_updates=[], converge_iter=200 - 1, p_S_history=[])
        return agent, area, len(targets_locations), targets_locations

    # def _get_X_data(self) -> np.array:
    #     """
    #     X is a vector with size of NXN, which represents the signals which the agent receive. X generated from many parameters
    #     as lambda (agent's sensor strength), pta, alpha and the location (Bernoulli distributions of the probability for
    #     a single cell will generate alarm, false alarm and the distance of the cell from the agent)
    #     """
    #     pass

    def _get_sequence(self, seq_len: int, num_features: int, area: Area, agent: Agent) -> Tuple[np.array, np.array]:
        """
        Generation of num_sequences for Time Series-Deep Learning model (RNN / LSTM / GRU)
        Each sequence is a new Area with agent and targets
        :return: X - shape: (NXN, num_features) ; y - shape: (NXN)
        """
        data = np.empty((seq_len, area.num_cells_axis * area.num_cells_axis, num_features))
        rij = area.calculate_distance(agent_location=agent.current_location,
                                      cells_indices=np.indices(area.cells.shape, sparse=True)).reshape(-1, 1)
        for seq_num in range(seq_len):
            X = DataGenerator.simulate_agent_receive_signals(agent=agent, area=area)
            data[seq_num] = np.concatenate((X, rij), axis=1)
        y = area.cells.reshape(-1)
        return data, y

    def time_series_data_generation(self, num_sequences: int = 1000, seq_len: int = 100, num_features: int = 2,
                                    train_ratio: float = 0.7, validation_ratio: float = 0.15):
        """
        Running the data generation - Creation of num_sequences (each sequence with seq_len length)
        Features are:   1) The X matrix (signal received from the target)
                        2) The distance of the agent from the cell
        """
        area_size = config['PARAMS'].getint('N') * config['PARAMS'].getint('N')
        X = np.empty((num_sequences, seq_len, area_size, num_features))
        y = np.empty((num_sequences, area_size))
        for seq_num in tqdm(range(num_sequences)):
            agent, area, num_targets, targets_locations = self._get_generated_params_for_sequence()
            X[seq_num], y[seq_num] = self._get_sequence(seq_len=seq_len, area=area, agent=agent,
                                                        num_features=num_features)
        save_hdf5_to_file('X', X)
        save_hdf5_to_file('y', y)


if __name__ == '__main__':
    data_generator = DataGenerator()
    data_generator.time_series_data_generation(num_sequences=10000, seq_len=100)
