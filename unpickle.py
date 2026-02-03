import pickle


with open('artifacts/q_table.pkl', 'rb') as f:
    data = pickle.load(f)

    print(data)