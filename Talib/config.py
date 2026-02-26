# Константы для приложения Talib (OHLCV, метки рангов, классификаторы).

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]

RANK_LABEL_PREFIX = "rank_"

CLASSIFIER_CHOICES = [
    "Random Forest",
    "SVM",
    "XGBoost",
    "Voting (RF+SVC+LR)",
    "Stacking (RF,SVC,LR)",
    "Stacking (RF,SVC,XGB meta)",
]

USE_TIME_FEATURES = False
USE_RETURN_REGRESSOR = True
USE_VOLATILITY_REGIME = True

YFINANCE_PERIODS = ["1mo", "3mo", "6mo", "1y", "2y", "5y"]
YFINANCE_INTERVALS = ["1d", "1h", "5m", "15m"]

DEFAULT_WINDOW_LEN = 20
DEFAULT_HORIZON = 5
DEFAULT_N_QUANTILES = 5
DEFAULT_TRAIN_RATIO = 0.7

DEFAULT_COMMISSION_PCT = 0.001
DEFAULT_SLIPPAGE_PCT = 0.0005

DEFAULT_MIN_RETURN_THRESHOLD = 0.003
DEFAULT_MIN_CONFIDENCE = 0.5
DEFAULT_HOLD_BARS = None

LONG_RANK_INDEX = 0
SHORT_RANK_INDEX = -1

LSTM_EPOCHS = 50
LSTM_BATCH_SIZE = 32
LSTM_UNITS = 64

# Порог разницы train_acc - test_acc для предупреждения о переобучении
OVERFIT_ACC_DIFF_THRESHOLD = 0.15
