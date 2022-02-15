import time
import matplotlib.pyplot as plt
import numpy as np
import visualkeras

def train_model(model, x_train, x_test, y_train, y_test, batch_size, epochs):
    """Trains the given compiled keras model and outputs the history of results and the execution time.

    Args:
        model (keras.Model): Compiled keras model.
        x_train (numpy.Array): Training data.
        x_test (numpy.Array): Test data.
        y_train (numpy.Array): Labels of the training data.
        y_test (numpy.Array): Labels of the test data.
        batch_size (int): Batch size used for the training.
        epochs (int): Number of epochs used for the training.

    Returns:
        The result history and the execution time.
    """
 
    start = time.time()
    # Putting the output of model.fit in a variable "history" gives access to information on accuracy and loss
    history = model.fit(x=x_train, 
            y=y_train, 
            batch_size=batch_size,
            epochs=epochs, 
            validation_data=(x_test, y_test)
            )
    end = time.time()
    execution_time = end - start

    return history, execution_time

def display_model(model):
    """Displays a schematic representation of the given model.

    Args:
        model (keras.Model): The model to display
    """
    return visualkeras.layered_view(model, draw_volume=False, legend=True)


def display_results(history, execution_time, model_name = "model"):
    """Displays accuracy and loss graphs, and prints the execution time.

    Args:
        history (keras.callbacks.History): The history to use for the displays.
        execution_time (Int): Execution time in seconds.
        model_name (str, optional): The name of the model to display. Defaults to "model".
    """
    _, ax = plt.subplots(1,2, figsize=(15,8))

    ax[0].plot(history.history['accuracy'], label='training accuracy')
    ax[0].plot(history.history['val_accuracy'], label='validation accuracy')
    ax[0].grid()
    ax[0].legend()
    ax[0].set_title('Accuracy vs. Epochs')
    ax[0].set_xlabel('# Epochs')

    ax[1].plot(history.history['loss'], label='training loss')
    ax[1].plot(history.history['val_loss'], label='validation loss')
    ax[1].grid()
    ax[1].legend()
    ax[1].set_title('Loss vs. Epochs')
    ax[1].set_xlabel('# Epochs')

    plt.title = f"Results for {model_name}"
    plt.show()
    
    print(f"The model training lasted {execution_time} s.")