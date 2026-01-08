import os

# ---- Project paths ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

DATA_DIR = os.path.join(PROJECT_DIR, "data")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
VISUALS_DIR = os.path.join(PROJECT_DIR, "visuals")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(VISUALS_DIR, exist_ok=True)

# Where to save the trained model pipeline
DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, "imdb_best_model.joblib")

# Optional: where to save metrics as JSON
METRICS_PATH = os.path.join(DATA_DIR, "imdb_metrics.json")

# ---- ML settings ----
RANDOM_STATE = 42
TEST_SIZE = 0.2          # validation size (from train split)
MAX_FEATURES = 20000
NGRAM_RANGE = (1, 2)     # unigrams + bigrams

# Neutral / uncertain prediction threshold
# If max(proba) < NEUTRAL_THRESHOLD -> label as NEUTRAL / UNCERTAIN
NEUTRAL_THRESHOLD = 0.6