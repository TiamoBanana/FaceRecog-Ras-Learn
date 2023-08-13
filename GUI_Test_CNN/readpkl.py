import pickle

with open('ExpData/labels.pkl', 'rb') as f:
    labels = pickle.load(f)
print(labels)