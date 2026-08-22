import os

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_absolute_error,mean_squared_error)

import matplotlib.pyplot as plt

from data_loader import DataLoader
from sequence_builder import SequenceBuilder

from models.lstm_model import LSTMModel
from models.bilstm_model import BiLSTMModel
from models.attention_bilstm import (AttentionBiLSTMModel)

from strategy.prediction_strategy import (PredictionStrategy)
from backtest.engine import (BacktestEngine)

import config


############### Data ###########################
print("\nLoading data...")
loader = DataLoader(config.DATA_PATH)
df = loader.load()
df = loader.prepare(df)
print("\nData shape:")
print(df.shape)
print("\nColumns:")
print(df.columns.tolist())


# =========================================================
# 2. FEATURES
# =========================================================

features = config.FEATURES
target = config.TARGET
X_raw = df[features].values
y_raw = df[target].values


# =========================================================
# 3. TRAIN / TEST SPLIT
# =========================================================

split = int(len(df) *config.TRAIN_RATIO)

print(
    "\nTrain rows:",
    split
)

print(
    "Test rows:",
    len(df) - split
)


# =========================================================
# 4. SCALE FEATURES
# =========================================================

scaler = StandardScaler()

scaler.fit(
    X_raw[:split]
)

X_scaled = scaler.transform(
    X_raw
)


# =========================================================
# 5. CREATE SEQUENCES
# =========================================================

builder = SequenceBuilder(
    config.WINDOW_SIZE
)

X, y = builder.create_sequences(
    X_scaled,
    y_raw
)


# =========================================================
# 6. SEQUENCE SPLIT
# =========================================================

sequence_split = (
    split -
    config.WINDOW_SIZE
)

X_train = X[
    :sequence_split
]

y_train = y[
    :sequence_split
]

X_test = X[
    sequence_split:
]

y_test = y[
    sequence_split:
]


print(
    "\nX_train:",
    X_train.shape
)

print(
    "X_test:",
    X_test.shape
)


# =========================================================
# 7. MODELS
# =========================================================

models = {

    "LSTM":
        LSTMModel(
            config.WINDOW_SIZE,
            len(features)
        ),

    "BiLSTM":
        BiLSTMModel(
            config.WINDOW_SIZE,
            len(features)
        ),

    "Attention_BiLSTM":
        AttentionBiLSTMModel(
            config.WINDOW_SIZE,
            len(features)
        )

}


# =========================================================
# 8. STRATEGY
# =========================================================

strategy = PredictionStrategy(

    threshold=
        config.PREDICTION_THRESHOLD

)


# =========================================================
# 9. BACKTEST ENGINE
# =========================================================

backtester = BacktestEngine(

    initial_capital=
        config.INITIAL_CAPITAL,

    transaction_cost=
        config.TRANSACTION_COST

)


all_results = {}


# =========================================================
# 10. TRAIN EACH MODEL
# =========================================================

for model_name, model in models.items():

    print("\n")
    print("=" * 60)

    print(
        "TRAINING:",
        model_name
    )

    print("=" * 60)


    # -----------------------------------------------------
    # Train
    # -----------------------------------------------------

    history = model.fit(

        X_train,

        y_train,

        epochs=config.EPOCHS,

        batch_size=config.BATCH_SIZE,

        validation_data=(
            X_test,
            y_test
        )

    )


    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    predictions = model.predict(
        X_test
    )


    # -----------------------------------------------------
    # ML Metrics
    # -----------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )


    # Direction accuracy

    actual_direction = (
        y_test > 0
    )

    predicted_direction = (
        predictions > 0
    )

    direction_accuracy = (
        actual_direction ==
        predicted_direction
    ).mean()


    # -----------------------------------------------------
    # Signals
    # -----------------------------------------------------

    signals = np.array([

        strategy.generate_signal(
            pred
        )

        for pred in predictions

    ])


    # -----------------------------------------------------
    # Backtest
    # -----------------------------------------------------

    backtest_result = backtester.run(

        actual_returns=y_test,

        signals=signals

    )


    metrics = backtester.metrics(
        backtest_result
    )


    # -----------------------------------------------------
    # Store
    # -----------------------------------------------------

    all_results[
        model_name
    ] = {

        "MAE":
            mae,

        "RMSE":
            rmse,

        "Direction Accuracy":
            direction_accuracy,

        "Sharpe":
            metrics["Sharpe Ratio"],

        "Total Return":
            metrics["Total Return"],

        "Max Drawdown":
            metrics["Max Drawdown"],

        "Trades":
            metrics["Trades"],

        "Predictions":
            predictions,

        "Backtest":
            backtest_result

    }


    # -----------------------------------------------------
    # Save model
    # -----------------------------------------------------

    os.makedirs(
        "saved_models",
        exist_ok=True
    )

    model.save(
        f"saved_models/{model_name}.keras"
    )


    # -----------------------------------------------------
    # Print results
    # -----------------------------------------------------

    print("\nResults:")

    print(
        "MAE:",
        mae
    )

    print(
        "RMSE:",
        rmse
    )

    print(
        "Direction Accuracy:",
        direction_accuracy
    )

    print(
        "Sharpe:",
        metrics["Sharpe Ratio"]
    )

    print(
        "Total Return:",
        metrics["Total Return"]
    )

    print(
        "Max Drawdown:",
        metrics["Max Drawdown"]
    )

    print(
        "Trades:",
        metrics["Trades"]
    )


# =========================================================
# 11. MODEL COMPARISON
# =========================================================

comparison = []

for model_name, result in all_results.items():

    comparison.append({

        "Model":
            model_name,

        "MAE":
            result["MAE"],

        "RMSE":
            result["RMSE"],

        "Direction Accuracy":
            result["Direction Accuracy"],

        "Sharpe":
            result["Sharpe"],

        "Total Return":
            result["Total Return"],

        "Max Drawdown":
            result["Max Drawdown"],

        "Trades":
            result["Trades"]

    })


comparison_df = pd.DataFrame(
    comparison
)


print("\n")
print("=" * 70)

print("MODEL COMPARISON")

print("=" * 70)

print(
    comparison_df.to_string(
        index=False
    )
)


comparison_df.to_csv(
    "model_comparison.csv",
    index=False
)


# =========================================================
# 12. PLOT EQUITY CURVES
# =========================================================

plt.figure(
    figsize=(12, 6)
)

for model_name, result in all_results.items():

    equity = result[
        "Backtest"
    ]["Equity"]

    plt.plot(
        equity,
        label=model_name
    )

plt.title(
    "Gold Prediction Strategy - Equity Curve"
)

plt.xlabel(
    "Test Period"
)

plt.ylabel(
    "Portfolio Value"
)

plt.legend()

plt.grid()

plt.tight_layout()

plt.savefig(
    "equity_curve.png"
)

plt.show()


# =========================================================
# 13. PLOT ACTUAL VS PREDICTED
# =========================================================

for model_name, result in all_results.items():

    predictions = result[
        "Predictions"
    ]

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        y_test,
        label="Actual Return"
    )

    plt.plot(
        predictions,
        label="Predicted Return"
    )

    plt.title(
        f"{model_name} - Actual vs Predicted"
    )

    plt.xlabel(
        "Test Period"
    )

    plt.ylabel(
        "Return"
    )

    plt.legend()

    plt.grid()

    plt.tight_layout()

    plt.savefig(
        f"{model_name}_prediction.png"
    )

    plt.show()


print("\nTraining completed.")