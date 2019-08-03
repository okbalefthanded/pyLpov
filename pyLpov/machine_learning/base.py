import pickle

# Save model after training
def save_model(filename, model):
        pickle.dump(model, open(filename, 'wb'))

