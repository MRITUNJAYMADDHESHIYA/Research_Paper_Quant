

DATA_PATH = "C:/Users/Mritunjay Maddhesiya/OneDrive/Desktop/Research_Paper/11_Forecasting_Position_Model/data/XAU_Daily_20_26.csv"

DATE_COLUMN = "Date"

FEATURES = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]
TARGET      = "Return"

WINDOW_SIZE = 100
TRAIN_RATIO = 0.80
EPOCHS      = 30
BATCH_SIZE  = 32
LEARNING_RATE = 0.001

PREDICTION_THRESHOLD = 0.001   
TRANSACTION_COST     = 0.0005
INITIAL_CAPITAL      = 100000
RANDOM_SEED          = 42