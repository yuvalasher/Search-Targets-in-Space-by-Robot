from typing import List, Tuple, Dict
from dataclasses import dataclass
from utils import Location, CONFIG_PATH
from utils import save_pickle_object, NOISE
import Area
from DataGenerator import DataGenerator
import numpy as np
from copy import deepcopy
from configparser import ConfigParser
import seaborn as sns
import matplotlib.pyplot as plt

config = ConfigParser()
config.read(CONFIG_PATH)
N = config['PARAMS'].getint('N')
sns.set(rc={'figure.figsize': (7.7, 4.27)})


@dataclass
class Agent:
    """
    p_S = {'t': np.array (probabilities),
           't - 1': np.array (probabilities)}
    """
    current_location: Location
    lambda_strength: float
    p_S: Dict[str, np.array]
    entropy_updates: List[float]
    information_gain_updates: List[float]
    converge_iter: int
    p_S_history: List[np.array]

    def bayesian_update(self, area: Area, until_convergence: bool = True, verbose: bool = False, stop_after_iteration: int = 1000) -> None:
        """
        Running the bayesian updating on a given matrix generated from generator until convergence
        :param until_convergence: Run updating until convergence of all the targets (and only them) or until infinity (searching targets all the tine)
        :param verbose: True if want to show the probabilities map of the agent through the updates
        """
        t: int = 0
        converge_status = False
        converged = False
        convergence_threshold: float = config['PARAMS'].getfloat('P_THRESHOLD')
        while not converge_status:

            self.p_S_history.append(self.p_S["t"])
            if self._check_convergence(until_convergence=until_convergence,
                                    convergence_threshold=convergence_threshold,
                                    targets_locations=area.targets_locations):

                if not converged:
                    self.converge_iter = t
                    converged = True
                if until_convergence:
                    converge_status = True

            for a, x, s in DataGenerator.simulate_data(area=area, agent=self):
                self._update_probability_of_target_existence(area=area, evidence=x)
                self._calculate_metrics()

                if verbose and t % 50 == 0 and t > 0:
                    print('Number of converged targets: {}'.format(
                        len(np.where(np.array(list(self.p_S['t'])) > convergence_threshold)[0])))
                    print(
                        't: {}; Targets Cells Probs: {}'.format(t, [(location, self.p_S['t'][location]) for location in
                                                                    area.targets_locations]))
                # if verbose and t % 50 == 0 and t > 0:
                    # self._plot_target_searching(area=area, t=t)
                    # self._plot_metrics(t=t)
                t += 1
            if t >= stop_after_iteration:
                break
        self._done_convergence(area=area, t_end=t)

    def _done_convergence(self, area: Area, t_end: int):
        print('\n*********** Convergence is Done ! t = {} ***********'.format(t_end))
        print('Targets Cells Probabilities: {}'.format(
            [self.p_S['t'][location] for location in area.targets_locations]))
        DataGenerator.tabulate_matrix(self.p_S['t'])
        # self._plot_target_searching(area=area, t=t_end)
        # self._plot_metrics(t=t_end)
        save_pickle_object(obj_name='p_S_converged', obj=self.p_S['t'])

    def _update_probability_of_target_existence(self, area: Area, evidence: np.array) -> None:
        """
        Bayesian updating for p_S (prior) based on the evidence -> posterior
        Evidence is the X matrix (NXN)
        evidence: The signal received from a cell (x(ci,t)). A signal can be True Alarm (TA) or False Alarm (FA)
        """

        def update_if_x_is_1(p_S_t_minus_1):
            p_S_t_minus_1_ta = p_S_t_minus_1 * area.pta
            return (p_S_t_minus_1_ta) / (p_S_t_minus_1_ta + (1 - p_S_t_minus_1) * area.alpha * area.pta)

        def update_if_x_is_0(p_S_t_minus_1):
            p_x_receive_signal = self.calculate_probability_of_agent_receiving_signal_from_cell(area=area)
            numerator = p_S_t_minus_1 * ((1 - area.pta) + area.pta * (1 - p_x_receive_signal))
            denominator = numerator + (1 - p_S_t_minus_1) * (
                    (1 - area.alpha * area.pta) + area.alpha * area.pta * (
                    1 - p_x_receive_signal))
            return numerator / denominator

        p_S_t_minus_1 = deepcopy(self.p_S['t-1'])
        self.p_S['t-1'] = deepcopy(self.p_S['t'])
        self.p_S['t'] = np.where(evidence == 1, update_if_x_is_1(p_S_t_minus_1), update_if_x_is_0(p_S_t_minus_1))

    def _check_convergence(self, until_convergence: bool, convergence_threshold: float,
                           targets_locations: List[Location]) -> bool:
        """
        Checking if the cells with the targets got convergence - the probability of the cells are bigger than P_THRESHOLD
        and that there is only 3 targets identified (No False Negative)
        The bigger value of alpha, the time until convergence will be higher (pta=1, alpha = 0.9 -> t=2388;
        pta=1, alpha = 0.5 -> t=145)
        :return: True if all the target cells' probability is bigger than the convergence threshold else False
        """
        # if not until_convergence:
        #     return False
        targets_locations_probs = [self.p_S['t'][location] for location in targets_locations]
        num_of_targets_converged = len(np.where(np.array(targets_locations_probs) > convergence_threshold)[0])
        num_of_converged_cells = len(np.where(np.array(list(self.p_S['t'])) > convergence_threshold)[0])
        return False if (num_of_targets_converged != len(targets_locations) or num_of_converged_cells != len(
            targets_locations)) else True

    def calculate_probability_of_agent_receiving_signal_from_cell(self, area: Area) -> np.array:
        """
        Calculate P(X(ci, t)) based on the Pta, P(S(ci,t-1)), alpha
        This function will be run by main num_cells times
        rij - the distance between target to agent
        Pr(signal received by the sensor / alarm sent from ci) = exp(-rij/lambda)
        :return: P(X(ci, t))
        """
        rij = area.calculate_distance(agent_location=self.current_location,
                                      cells_indices=np.indices(area.cells.shape, sparse=True))

        return np.exp(-(rij / self.lambda_strength))

    def _plot_target_searching(self, area: Area, t: int):
        arrays_names = ["Area Cells", "Agent's Cells Probabilities"]
        arrays = [area.cells, self.p_S['t']]
        fig, axes = plt.subplots(2, 1)
        for idx, (array, ax, array_name) in enumerate(zip(arrays, axes, arrays_names)):
            ax.set_title('{} - t: {}'.format(array_name, t))
            _ = sns.heatmap(array, cmap=sns.cubehelix_palette(1000, hue=0.05, rot=0, light=0.9, dark=0), cbar=False,
                            ax=ax)
        plt.pause(1e-12)

    def _calculate_entropy(self) -> float:
        """
        Calculate the entropy by each cell and summed through all the cells (N * N)
        Maximum entropy is uniform (0.5, 0.5) -> entropy: 1 for each cell and summed to entropy of 1 * N * N
        Adding very small const to the probabilities to ignore situation of log of 0 (not defined)
        """
        return float(np.sum(-self.p_S['t'] * np.log2((self.p_S['t'] + NOISE)) - (1 - (self.p_S['t'] - NOISE)) * np.log2(1 - (self.p_S['t'] - NOISE))))

    def _calculate_information_gain_KL(self) -> float:
        """
        Calculating the information gain by Kullback-Leibler divergence - the information gained between 2 timestamps -
        What is the information this round (timestamp) contributed (distribution of "t-1" VS the distribution of "t")
        """
        return float(np.sum(self.p_S['t'] * (np.log2((self.p_S['t'] + NOISE) / (self.p_S['t-1'] + NOISE))) +
                            (1 - self.p_S['t']) * (np.log2((1 - (self.p_S['t'] - NOISE)) / (1 - (self.p_S['t-1'] - NOISE))))))

    def _calculate_metrics(self) -> None:
        self.entropy_updates.append(self._calculate_entropy())
        self.information_gain_updates.append(self._calculate_information_gain_KL())

    def get_metrics(self) -> Tuple[List[float], List[float], int]:
        return self.entropy_updates, self.information_gain_updates, self.converge_iter

    def get_prob_history(self, location) -> List[float]:

        ret_prob = []
        for t in range(len(self.p_S_history)):
            ret_prob.append(self.p_S_history[t][location])
        return ret_prob

    def _plot_metrics(self, t:int):
        fig, ax = plt.subplots()
        ax.plot(list(range(len(self.entropy_updates))), self.entropy_updates, label='Entropy')
        ax.plot(list(range(len(self.information_gain_updates))), self.information_gain_updates, label='Information Gain')
        ax.legend()
        ax.set_title('Metrics Through Time - t: {}'.format(t))
        plt.xlabel('time (s)')
        plt.ylabel('value')
        plt.show()

    @staticmethod
    def get_p_S_from_initial_prior(prior: float, area: Area) -> Dict[str, np.array]:
        """
        Build p(x(ci,t) & p(x(ci,t-1) as Dict
        """
        return {'t': np.full((area.num_cells_axis, area.num_cells_axis), prior),
                't-1': np.full((area.num_cells_axis, area.num_cells_axis), prior)}
