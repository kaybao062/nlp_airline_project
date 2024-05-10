# hack to import modules from parent directories:
import os
import sys

sys.path.insert(0, f"{os.path.dirname(os.path.abspath(__file__))}/../")

from util.dataset import load_policies

policies = load_policies(as_dataframe=True, limit=1000)
airlines = policies["Airline"].unique()
print(airlines)