from typing import Tuple
from pathlib import Path

# AgentGRU Training Consts
NUM_EPOCHS: int = 30
train_ratio: float = 0.7
validation_ratio: float = 0.15
test_ratio: float = 0.15
hidden_dim: int = 32
num_layers: int = 2
lr: float = 1e-3
batch_size: int = 256
PRINT_EVERY: int = 5
SAVE_EVERY: int = 5

Location = Tuple[int, int]
CONFIG_PATH: Path = Path('config.cfg')
NOISE: float = 1e-8
FILES_PATH = r"C:\Users\asher\OneDrive\Documents\Data Science Degree\3th Year\Curiosity\Search-Targets-in-Space-by-Robot\Modules"

# Early Stopping Consts
MIN_IMPROVEMENT: float = 1e-3
PATIENT_NUM_EPOCHS: int = 10