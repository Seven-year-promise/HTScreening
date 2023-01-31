from pathlib import Path

file_path = Path(__file__).parent

TRAIN_ORI_PATH = file_path / "data/train/ori/"
TRAIN_MODE_OF_ACTION_PATH = file_path / "data/train/OldCompoundsMoA.csv"
TRAIN_SAVE_CLEAN_PATH = file_path / "data/train/cleaned/"
TRAIN_SAVE_FEATURE_PATH = file_path / "data/train/featured/"

TRAIN_DATA_PATH = file_path / "data/train/"

TRAIN_SAVE_FINAL_RESULT_PATH = file_path / "data/train/final_result/"


TEST_ORI_PATH = file_path / "data/test/ori/"
TEST_MODE_OF_ACTION_PATH = file_path / "data/test/OldCompoundsMoA.csv"
TEST_SAVE_CLEAN_PATH = file_path / "data/test/cleaned/"
TEST_SAVE_FEATURE_PATH = file_path / "data/test/featured/"

TEST_DATA_PATH = file_path / "data/test/"

TEST_SAVE_FINAL_RESULT_PATH = file_path / "data/test/final_result/"
