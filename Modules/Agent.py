from typing import List, Tuple, Dict
from dataclasses import dataclass
from utils import Location, CONFIG_PATH
import Area
import pandas as pd
import numpy as np
from copy import deepcopy
import pickle


@dataclass
class Agent:
    """
    p_X = {'t': {(0,0): probability_value, (0,1): probability_value},...},
           't - 1': {(0,0): probability_value, (0,1): probability_value, ...}}
    """
    current_location: Location
    lambda_strength: float
    p_X: Dict[str, Dict[Location, float]]

    def bayesian_update(self, df: pd.DataFrame, area: Area, convergence_threshold: float,
                        targets_locations: List[Location]) -> None:
        """
        Running the bayesian updating on a given dataframe as data
        """
        for idx, (t, cell_location, a, x, s) in df.iterrows():
            self.update_probability_of_target_existence(area=area, cell_location=eval(cell_location), evidence=x)
            if self.check_convergence(convergence_threshold=convergence_threshold, targets_locations=targets_locations):
                if idx % 400 == 0:
                    print('\n*********** Convergence is Done ! t = {} ***********'.format(t))
                    print('Targets Cells Probabilities: {}'.format([self.p_X['t'][location] for location in targets_locations]))
                    print(self.p_X['t'])
                    print('Number of converged targets: {}'.format(len(np.where(np.array(list(self.p_X['t'].values())) > 0.95)[0])))
                    self.save_pickle_object(obj_name='p_X_converged', obj=self.p_X['t'])
                    # break
            if idx % 400 == 0:
                print('t: {}; Targets Cells Probabilities: {}'.format(t, [self.p_X['t'][location] for location in targets_locations]))

    def check_convergence(self, convergence_threshold: float, targets_locations: List[Location]) -> bool:
        """
        Checking if the cells with the targets got convergence - the probability of the cells are bigger than P_THRESHOLD
        :return: True if all the target cells' probability is bigger than the convergence threshold else False
        """
        targets_locations_probs = [self.p_X['t'][location] for location in targets_locations]
        return True if len(np.where(np.array(targets_locations_probs) > convergence_threshold)[0]) == len(
            targets_locations) else False

    def update_probability_of_target_existence(self, area: Area, cell_location: Location, evidence: int) -> None:
        """
        Bayesian updating for p_X (prior) based on the evidence -> posterior
        evidence: The signal received from a cell (x(ci,t)). A signal can be True Alarm (TA) or False Alarm (FA)
        """
        p_X_t_minus_1 = deepcopy(self.p_X['t-1'])
        self.p_X['t-1'] = deepcopy(self.p_X['t'])
        p_X_t_minus_1_cell = p_X_t_minus_1[cell_location]
        p_X_t_minus_1_ta = p_X_t_minus_1_cell * area.pta
        if evidence == 1:
            self.p_X['t'][cell_location] = (p_X_t_minus_1_ta) / (
                    p_X_t_minus_1_ta + (1 - p_X_t_minus_1[cell_location]) * area.alpha * area.pta)

        else:
            p_a_receive_signal = self.calculate_probability_of_agent_receiving_signal_from_cell(area=area,
                                                                                                target_location=cell_location)
            numerator = p_X_t_minus_1_cell * ((1 - area.pta) + area.pta * (1 - p_a_receive_signal))
            denominator = numerator + (1 - p_X_t_minus_1_cell) * (
                    (1 - area.alpha * area.pta) + area.alpha * area.pta * (
                    1 - p_a_receive_signal))
            self.p_X['t'][cell_location] = numerator / denominator

    def calculate_probability_of_agent_receiving_signal_from_cell(self, area: Area, target_location: Location) -> float:
        """
        Calculate P(X(ci, t)) based on the Pta, P(S(ci,t-1)), alpha
        This function will be run by main num_cells times
        rij - the distance between target to agent
        Pr(signal received by the sensor / alarm sent from ci) = exp(-rij/lambda)
        :return: P(X(ci, t))
        """
        rij = area.calculate_distance(a_location=self.current_location,
                                      b_location=target_location)

        return np.exp(-(rij / self.lambda_strength))

    @staticmethod
    def get_p_X_from_initial_prior(prior: float) -> Dict[str, float]:
        """
        Build p(x(ci,t) & p(x(ci,t-1) as Dict
        """
        return {'t': prior, 't-1': prior}

    @staticmethod
    def save_pickle_object(obj_name: str, obj: object):
        with open('Pickles/{}.pkl'.format(obj_name), 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_pickle_object(obj_name: str):
        with open('Pickles/{}.pkl'.format(obj_name), 'rb') as f:
            return pickle.load(f)
