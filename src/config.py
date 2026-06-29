from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

HATECHECK_HF_ID = "Paul/hatecheck-polish"

PTASZYNSKI_MODEL_ID = "ptaszynski/bert-base-polish-cyberbullying"
TRELBERT_MODEL_ID = "deepsense-ai/trelbert"

BATCH_SIZE = 32
MAX_LENGTH = 256
SEED = 42

FEMININE_NAMES = ["Julia", "Maja", "Hanna", "Zofia", "Wiktoria"]
MASCULINE_NAMES = ["Jakub", "Antoni", "Szymon", "Jan", "Filip"]
NEUTRAL_PREFIX = "Osoba"

PREFIX_FORMAT = "{name}: {text}"

