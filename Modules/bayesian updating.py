from Area import Area
from Agent import Agent
from DataGenerator import DataGenerator
from utils import Location, CONFIG_PATH
from itertools import product
from configparser import ConfigParser
from datetime import datetime as dt

config = ConfigParser()
config.read(CONFIG_PATH)

if __name__ == '__main__':
    start_time = dt.now()
    params = config['PARAMS']
    DataGenerator.param_validation(params=params)
    cells_locations = [(a, b) for a, b in product(range(params.getint('N')), repeat=2)]
    targets_locations = eval(params['TARGET_LOCATIONS'])
    if targets_locations is None and params.getint('NUM_TARGETS'):
        targets_locations = Area.generate_targets(cells_locations=cells_locations,
                                                  num_targets=params.getint('NUM_TARGETS'))

    area = Area(num_cells_axis=params.getint('N'), t_interval=params.getint('T_INTERVAL'),
                pta=params.getfloat('pta'), alpha=params.getfloat('alpha'),
                targets_locations=targets_locations)

    agent = Agent(current_location=eval(params['AGENT_POSITION']), lambda_strength=params.getint('LAMBDA_SENSOR'),
                  p_S=Agent.get_p_S_from_initial_prior(prior=params.getfloat('INITIAL_PRIOR_P'), area=area))
    """
    2 Running modes: until convergence (all the targets identified and not False Positives) or until infinity
    """
    agent.bayesian_update(area=area, until_convergence=True, verbose=True)
    print('Bayesian Updating Done in {}'.format(dt.now() - start_time))
