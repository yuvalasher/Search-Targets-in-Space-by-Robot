from typing import Tuple
from pathlib import Path

# AgentGRU Training Consts
NUM_EPOCHS: int = 100
TRAIN_RATIO: float = 0.7
VALIDATION_RATIO: float = 0.15
TEST_RATIO: float = 0.15
hidden_dim: int = 100
num_layers: int = 1
lr: float = 1e-3
BATCH_SIZE: int = 50
PRINT_EVERY: int = 1
SAVE_EVERY: int = 10

Location = Tuple[int, int]
CONFIG_PATH: Path = Path('config.cfg')
NOISE: float = 1e-8
FILES_PATH = r"C:\Users\asher\OneDrive\Documents\Data Science Degree\3th Year\Curiosity\Search-Targets-in-Space-by-Robot\Modules"

# Early Stopping Consts
MIN_IMPROVEMENT: float = 1e-3
PATIENT_NUM_EPOCHS: int = 10