import pandas as pd
import pickle
import sys
import warnings

from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

warnings.filterwarnings("ignore")

def usage():
    print("Usage: python3 model.py <train|predict> [args...]")
    exit(1)

if len(sys.argv) < 2:
    usage()

command = sys.argv[1]

if command == "train":
    data = pd.read_csv("data/stats.csv")

    X = data[["opp_defense_skill"]]
    y = data[["comp_pct", "yds", "tds", "ints"]]
    weights = data[["weight"]].values.flatten()

    model = MultiOutputRegressor(LinearRegression())
    model.fit(X, y, sample_weight=weights)

    with open("model.pkl", "wb+") as f:
        pickle.dump(model, f)

elif command == "predict":
    if len(sys.argv) != 3:
        print("Usage: python3 model.py predict <OPPONENT DEFENSE SKILL [0.0, 1.0]>")
        exit(1)

    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    opp_defense_skill = float(sys.argv[2])
    pred = model.predict([[opp_defense_skill]])

    comp_pct, yds, tds, ints = pred[0]
    print("Completion %:", round(comp_pct, 2))
    print("Yards:", round(yds))
    print("TDs:", round(tds))
    print("INTs:", round(ints))

else:
    usage()
