from Area import Area
from Agent import Agent
from Cell import Cell
from DataGenerator import DataGenerator
from utils import Location, CONFIG_PATH, CSV_DATA_PATH
from itertools import product
from configparser import ConfigParser
from datetime import datetime as dt
import pandas as pd

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
    print('Initial Prior: {}'.format(
        Agent.get_p_X_from_initial_prior(dict.fromkeys(cells_locations, params.getfloat('INITIAL_PRIOR_P')))))
    agent = Agent(current_location=eval(params['AGENT_POSITION']), lambda_strength=params.getint('LAMBDA_SENSOR'),
                  p_X=Agent.get_p_X_from_initial_prior(
                      dict.fromkeys(cells_locations, params.getfloat('INITIAL_PRIOR_P'))))

    area = Area(num_cells=params.getint('N') * params.getint('N'), t_interval=params.getint('T_INTERVAL'),
                pta=params.getfloat('pta'), alpha=params.getfloat('alpha'),
                cells=[Cell(location=t_location, is_target=t_location in targets_locations) for t_location in
                       cells_locations])

    # TODO - run on the dataframe and call agent.update_probability_of_target_existence
    df = pd.read_csv(CSV_DATA_PATH)
    agent.bayesian_update(df=df, area=area, convergence_threshold=params.getfloat('P_THRESHOLD'),
                          targets_locations=targets_locations)
    print('Bayesian Updating Done in {}'.format(dt.now() - start_time))
