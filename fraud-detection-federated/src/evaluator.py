"""
Evaluates model performance on test data.
"""
def evaluate_model(model, X_test, y_test):
    return model.evaluate(X_test, y_test, verbose=0)
