import pickle

with open("tests/data/h36m/cameras.pkl", "rb") as f:
    data = pickle.load(f)

print(data)
