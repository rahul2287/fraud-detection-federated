"""
Trains a model on local client data.
"""
def train_model(model, X, y, epochs=5):
    history = model.fit(X, y, epochs=epochs, verbose=0)
    return model, history
