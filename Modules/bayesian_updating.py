from Area import Area
from Agent import Agent
from DataGenerator import DataGenerator
from utils import Location, CONFIG_PATH
from configparser import ConfigParser
from datetime import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import random
from typing import Dict, Any

config = ConfigParser()
config.read(CONFIG_PATH)

def bayesian_updating_iteration(params, until_convergence: bool = True, sensitivity_analysis: bool=False):
    start_time = dt.now()
    DataGenerator.param_validation(params=params)
    targets_locations = eval(params['TARGET_LOCATIONS'])

    if targets_locations is None and params.getint('NUM_TARGETS'):
        targets_locations = Area.generate_targets(num_targets=params.getint('NUM_TARGETS'))

    area = Area(num_cells_axis=params.getint('N'), t_interval=params.getint('T_INTERVAL'),
                pta=params.getfloat('pta'), alpha=params.getfloat('alpha'),
                targets_locations=targets_locations)

    agent = Agent(current_location=eval(params['AGENT_POSITION']), lambda_strength=params.getint('LAMBDA_SENSOR'),
                  p_S=Agent.get_p_S_from_initial_prior(prior=params.getfloat('INITIAL_PRIOR_P'), area=area),
                  entropy_updates=[], information_gain_updates=[], converge_iter=200 - 1, p_S_history=[])
    """
    2 Running modes: until convergence (all the targets identified and not False Positives) or until infinity
    """
    if not sensitivity_analysis:
        agent.bayesian_update(area=area, until_convergence=until_convergence, verbose=True, stop_after_iteration=200)
        print('Bayesian Updating Done in {}'.format(dt.now() - start_time))

    else:
        target_prob = agent.get_prob_history(targets_locations[2])
        # non_target_loc = random.choice([loc for loc in zip(range(params.getint('N')), range(params.getint('N'))) if loc not in targets_locations])
        # Randomly selected non targets
        list_of_loc = [(9, 10), (7, 15), (1, 16)]
        agent_pos = eval(params['AGENT_POSITION'])
        non_target_prob = {}
        for loc in list_of_loc:
            dist = np.round(np.sqrt(np.power(loc[0] - agent_pos[0], 2) + np.power(loc[1] - agent_pos[1], 2)))
            non_target_prob[dist] = agent.get_prob_history(loc)

        ent_upd, info_gain_upd, conv_iter  = agent.get_metrics()

        return ent_upd, info_gain_upd, conv_iter, target_prob, non_target_prob


if __name__ == '__main__':
    params = config["PARAMS"]
    bayesian_updating_iteration(params=params, until_convergence=True)