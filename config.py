from pathlib import Path

MODE = "data/train"
file_path = Path(__file__).parent

ORI_PATH = file_path / MODE / "ori/"
MODE_OF_ACTION_PATH = file_path / MODE / "OldCompoundsMoA.csv"
SAVE_CLEAN_PATH = file_path / MODE / "cleaned/"
SAVE_FEATURE_PATH = file_path / MODE / "featured/"

DATA_PATH = file_path / MODE

SAVE_FINAL_RESULT_PATH = file_path / MODE / "final_result/"
