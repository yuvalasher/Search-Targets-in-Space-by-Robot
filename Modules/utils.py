from typing import Tuple
from pathlib import Path

Location = Tuple[int, int]
CONFIG_PATH = Path('config.cfg')
CSV_DATA_PATH = Path('Data/data_V2_300_epochs.csv')
CSV_COLUMNS = ['t', 'cell_location', 'a', 'x', 's']