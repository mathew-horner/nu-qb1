#!/usr/bin/env python3
import argparse
import pandas as pd
import pickle
import sys
import warnings

from datetime import datetime as dt
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

# File name to pickle the model data to.
MODEL_PICKLE_FILE = "model.pkl"

# File name to read stats from.
STATS_FILE = "data/stats.csv"

warnings.filterwarnings("ignore")

def train(args):
    data = pd.read_csv(STATS_FILE)

    # Parse date strings into date objects in DataFrame.
    data["date"] = data["date"].map(lambda date: dt.strptime(date, "%Y-%m-%d"))

    min_date = min(data["date"])
    max_date_spread = (dt.now() - min_date).days

    def calculate_weight(game_date, max_date_spread):
        """Calculate weight to assign to game sample based on how long ago it was played."""
        date_spread = (dt.now() - game_date).days
        return max(0.1, 1.0 - (date_spread / max_date_spread))

    weights = [calculate_weight(date, max_date_spread) for date in data["date"]]

    X = data[["opp_defense_skill"]]
    y = data[["comp_pct", "yds", "tds", "ints"]]

    model = MultiOutputRegressor(LinearRegression())
    model.fit(X, y, sample_weight=weights)

    with open(MODEL_PICKLE_FILE, "wb+") as f:
        pickle.dump(model, f)

def predict(args):
    opp_defense_skill = float(args.rating)

    model = load_model()
    pred = model.predict([[opp_defense_skill]])

    comp_pct, yds, tds, ints = pred[0]
    comp_pct = round(comp_pct, 2)
    yds = round(yds)
    tds = round(tds)
    ints = round(ints)

    if args.format == "readable":
        print("Completion %:", comp_pct)
        print("Yards:", yds)
        print("TDs:", tds)
        print("INTs:", ints)
    elif args.format == "csv":
        print("Completion %,Yards,TDs,INTs")
        print(f"{comp_pct},{yds},{tds},{ints}")

def evaluate(args):
    # The fields to evaluate MAPE on. We don't do this on TDs and INTs because
    # of their propensity towards values of 0, which MAPE does not work well on.
    FIELDS = ["comp_pct", "yds"]

    data = pd.read_csv(STATS_FILE)
    model = load_model()
    sigmas = [0 for _ in range(len(FIELDS))]

    for row_index, row in data.iterrows():
        rating = row["opp_defense_skill"]
        pred = model.predict([[rating]])[0]
        actual = list(row[FIELDS])
        for idx in range(len(FIELDS)):
            sigmas[idx] += abs((actual[idx] - pred[idx]) / actual[idx])

    n = row_index + 1
    for idx in range(len(FIELDS)):
        sigmas[idx] = 100 * (1 / n) * sigmas[idx]

    print("MAPEs")
    for idx, field in enumerate(FIELDS):
        print(f"{field}: {round(sigmas[idx], 2)}%")

def load_model():
    with open(MODEL_PICKLE_FILE, "rb") as f:
        return pickle.load(f)

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers()

train_parser = subparsers.add_parser("train")
train_parser.set_defaults(func=train)

predict_parser = subparsers.add_parser("predict")
predict_parser.add_argument("rating")
predict_parser.add_argument("-f", "--format", choices=["readable", "csv"], default="readable")
predict_parser.set_defaults(func=predict)

evaluate_parser = subparsers.add_parser("evaluate")
evaluate_parser.set_defaults(func=evaluate)

args = parser.parse_args()
args.func(args)
