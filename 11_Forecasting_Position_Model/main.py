import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_absolute_error,mean_squared_error)

from data_loader import DataLoader
from sequence_builder import SequenceBuilder

from models.lstm_model import LSTMModel
from models.bilstm_model import BiLSTMModel
from models.attention_bilstm import (AttentionBiLSTMModel)

from strategy.prediction_strategy import (PredictionStrategy)
from backtest.engine import (BacktestEngine)

import config


############### Data ###########################
print("Data Loading....")
loader = DataLoader(config.DATA_PATH)
df = loader.load()
df = loader.prepare(df)

############# Features #######################
features = config.FEATURES
target   = config.TARGET
X_raw    = df[features].values
y_raw    = df[target].values

############# train/test ####################
split = int(len(df) *config.TRAIN_RATIO)
print("\nTrain rows:", split)
print("Test rows:", len(df) - split)

########### Scaling data ###################
scaler = StandardScaler()
scaler.fit(X_raw[:split])
X_scaled = scaler.transform(X_raw)

########### Create Sequence ###################
builder = SequenceBuilder(config.WINDOW_SIZE)
X, y    = builder.create_sequences(X_scaled, y_raw)

########## Sequence split ###################
sequence_split = (split - config.WINDOW_SIZE)
X_train = X[:sequence_split]
y_train = y[:sequence_split]
X_test  = X[sequence_split:]
y_test  = y[sequence_split:]


print("\nX_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

test_start_index = split
test_dates = df["Date"].iloc[test_start_index:test_start_index + len(y_test)].values
test_close = df["Close"].iloc[test_start_index:test_start_index + len(y_test)].values

################ Model #######################
models = {
    "LSTM":
        LSTMModel(config.WINDOW_SIZE, len(features)),
    "BiLSTM":
        BiLSTMModel(config.WINDOW_SIZE, len(features)),
    "Attention_BiLSTM":
        AttentionBiLSTMModel(config.WINDOW_SIZE, len(features))
}


############## Strategy #####################
strategy = PredictionStrategy(threshold=config.PREDICTION_THRESHOLD)

############### Backtest ##################
backtester = BacktestEngine(initial_capital  = config.INITIAL_CAPITAL, transaction_cost = config.TRANSACTION_COST)


all_results = {}
os.makedirs("results", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

############# Train each Model ####################
for model_name, model in models.items():

    print("\n")
    print("=" * 60)
    print("TRAINING:", model_name)
    print("=" * 60)


    history     = model.fit(X_train, y_train, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE, validation_data=(X_test, y_test))
    predictions = model.predict(X_test)

    ######### Convert to 1D then print ############
    predicted_return = predictions.flatten()
    actual_return = y_test.flatten()
    print("\nPredicted vs Actual Returns:")
    print("=" * 70)
    print(
        f"{'Date':<15}"
        f"{'Actual':>15}"
        f"{'Predicted':>15}"
        f"{'Error':>15}"
    )
    print("-" * 70)
    for i in range(min(20, len(predicted_return))):
        error = predicted_return[i] - actual_return[i]
        print(
            f"{str(test_dates[i])[:10]:<15}"
            f"{actual_return[i]:>15.6f}"
            f"{predicted_return[i]:>15.6f}"
            f"{error:>15.6f}"
        )

    signals = np.array([strategy.generate_signal(pred) for pred in predictions])
    prediction_df = pd.DataFrame({
        "Date": test_dates,
        "Close": test_close,
        "Actual_Return": actual_return,
        "Predicted_Return": predicted_return,
        "Prediction_Error": predicted_return - actual_return,
        "Signal": signals})

    prediction_file = (f"results/{model_name}_predictions.csv")
    prediction_df.to_csv(prediction_file,index=False)
    print(f"\nPredictions saved to: {prediction_file}")

    mae                 = mean_absolute_error(y_test, predictions)
    rmse                = np.sqrt(mean_squared_error(y_test,predictions))
    actual_direction    = (y_test > 0)  ######## direction accuracy
    predicted_direction = (predictions > 0)
    direction_accuracy  = (actual_direction == predicted_direction).mean()


############### Backtest ######################
    backtest_result = backtester.run(actual_returns=y_test, signals=signals)
    metrics = backtester.metrics(backtest_result)

   ######## Store ###############
    all_results[model_name] = {
        "MAE":    mae,
        "RMSE":   rmse,
        "Direction Accuracy": direction_accuracy,
        "Sharpe": metrics["Sharpe Ratio"],
        "Total Return": metrics["Total Return"],
        "Max Drawdown": metrics["Max Drawdown"],
        "Trades": metrics["Trades"],
        "Actual": actual_return,
        "Predictions": predictions,
        "Backtest": backtest_result,
        "Prediction Date": prediction_df
    }


   ############# Result ##################
    print("\nResults:")
    print("MAE:", mae)
    print("RMSE:",rmse)
    print("Direction Accuracy:",direction_accuracy)
    print("Sharpe:", metrics["Sharpe Ratio"])
    print("Total Return:", metrics["Total Return"])
    print("Max Drawdown:", metrics["Max Drawdown"])
    print("Trades:", metrics["Trades"])

    model.save(f"saved_models/{model_name}.keras")


########## Model Comparsion #################
comparison = []
for model_name, result in all_results.items():
    comparison.append({
        "Model":  model_name,
        "MAE":    result["MAE"],
        "RMSE":   result["RMSE"],
        "Direction Accuracy":  result["Direction Accuracy"],
        "Sharpe":        result["Sharpe"],
        "Total Return":  result["Total Return"],
        "Max Drawdown":  result["Max Drawdown"],
        "Trades":        result["Trades"]
    })

comparison_df = pd.DataFrame(comparison)

print("\n")
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)
print(comparison_df.to_string(index=False))

comparison_df.to_csv("results/model_comparison.csv",index=False)

############ Plot ################
plt.figure(figsize=(12, 6))
for model_name, result in all_results.items():
    equity = result["Backtest"]["Equity"]
    plt.plot(equity, label=model_name)

plt.title("Gold Prediction Strategy - Equity Curve")
plt.xlabel("Test Period")
plt.ylabel("Portfolio Value")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("results/equity_curve.png", dpi=300)
plt.show()


############## Plot Actual, predicted ################
for model_name, result in all_results.items():
    actual = result["Actual"]
    predictions = result["Predictions"]
    plt.figure(figsize=(14, 6))
    plt.plot(actual, label="Actual Return", linewidth=1.5)
    plt.plot(predictions, label="Predicted Return", linewidth=1.2)
    plt.axhline(y=0, linestyle="--", linewidth=1)
    plt.title(f"{model_name} - Actual vs Predicted return")
    plt.xlabel("Test Period")
    plt.ylabel("Return")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"results/{model_name}_actual_vs_prediction.png", dpi=300)
    plt.show()


for model_name, result in all_results.items():
    predicted = result["Predictions"]
    plt.figure(figsize=(14, 5))
    plt.plot(predicted,label="Predicted Return")
    plt.axhline(y=0, linestyle="--", linewidth=1)
    plt.title(f"{model_name} - Predicted Return")
    plt.xlabel("Test Period")
    plt.ylabel("Predicted Return")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"results/{model_name}_predicted_return.png", dpi=300)
    plt.show()


for model_name, result in all_results.items():
    actual = result["Actual"]
    predicted = result["Predictions"]
    plt.figure(figsize=(7, 7))
    plt.scatter(actual, predicted, alpha=0.5)
    minimum = min(actual.min(), predicted.min())
    maximum = max(actual.max(), predicted.max())
    plt.plot([minimum, maximum], [minimum, maximum], linestyle="--")
    plt.title(f"{model_name} - Actual vs Predicted Scatter")
    plt.xlabel("Actual Return")
    plt.ylabel("Predicted Return")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"results/{model_name}_scatter.png", dpi=300)
    plt.show()


print("\nTraining completed.")

print("\nAll files saved inside:")
print("results/")

