from Area import Area
from Agent import Agent
from Cell import Cell
from DataGenerator import DataGenerator
from utils import Location, CONFIG_PATH, CSV_DATA_PATH
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
        targets_locations = Area.generate_targets(cells_locations=cells_locations, num_targets=params.getint('NUM_TARGETS'))

    agent = Agent(current_location=eval(params['AGENT_POSITION']), lambda_strength=params.getint('LAMBDA_SENSOR'),
                  p_X=Agent.get_p_X_from_initial_prior(
                      dict.fromkeys(cells_locations, params.getfloat('INITIAL_PRIOR_P'))))

    area = Area(num_cells=params.getint('N') * params.getint('N'), t_interval=params.getint('T_INTERVAL'),
                pta=params.getfloat('pta'), alpha=params.getfloat('alpha'),
                cells=[Cell(location=t_location, is_target=t_location in targets_locations) for t_location in cells_locations])
    data_generator = DataGenerator(csv_path=CSV_DATA_PATH)
    data_generator.simulate_data(area=area, agent=agent, num_rounds=params.getint('NUM_ROUNDS'))
    print('Data Generator Done in {}'.format(dt.now() - start_time))
