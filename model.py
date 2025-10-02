#!/usr/bin/env python3
import argparse
import pandas as pd
import pickle
import sys
import warnings

from datetime import datetime as dt
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

# File name to pickle the model data to
MODEL_PICKLE_FILE = "model.pkl"

# File name to read stats from
STATS_FILE = "data/stats.csv"

warnings.filterwarnings("ignore")

def train(args):
    data = pd.read_csv("data/stats.csv")

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
    with open(MODEL_PICKLE_FILE, "rb") as f:
        model = pickle.load(f)

    opp_defense_skill = float(args.rating)
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

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers()

train_parser = subparsers.add_parser("train")
train_parser.set_defaults(func=train)

predict_parser = subparsers.add_parser("predict")
predict_parser.add_argument("rating")
predict_parser.add_argument("-f", "--format", choices=["readable", "csv"], default="readable")
predict_parser.set_defaults(func=predict)

args = parser.parse_args()
args.func(args)
