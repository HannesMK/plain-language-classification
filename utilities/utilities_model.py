from typing import Any, List

import numpy as np
import pandas as pd
import sklearn.metrics
import tensorflow as tf
import tensorflow_hub as hub
import tf_keras
import transformers
from matplotlib import pyplot as plt


def _plot_history(
    data_history: pd.DataFrame,
    metric_training: str,
    metric_validation: str,
    y_limits: tuple | None,
    x_ticks: list | None,
    y_ticks: list | None,
    location_legend: str,
    font_size: int,
    width_in_inches: float,
    height_in_inches: float,
) -> tuple:
    """
    Helper function that creates a plot showing training and validation metrics over the course of the training epochs

    Parameters
    ----------
    data_history : pd.DataFrame
        Data frame containing the training history of the model
    metric_training : str
        Name of the training metric
    metric_validation : str
        Name of the validation metric
    y_limits : list | None
        Limits of the y axis
    x_ticks : list | None
        Ticks of the x axes
    y_ticks : list | None
        Ticks of the y axes
    location_legend : str
        {"upper left", "upper right", "lower left", "lower right"}, location of the legend
    font_size : int
        Base size of the font, passed to plt.rcParams["font_size"]
    width_in_inches : float
        Width of the figure in inches
    height_in_inches : float
        Height of the figure in inches

    Returns
    -------
    tuple
        (figure, axes)
    """
    plt.rcParams["font.size"] = font_size

    figure, axes = plt.subplots()

    figure.set_size_inches(width_in_inches, height_in_inches)

    epochs = [epoch + 1 for epoch in data_history.index]

    if len(epochs) == 1:
        marker = "o"
    else:
        marker = ""
    
    axes.plot(epochs, data_history[metric_training], label="Training", marker=marker)
    axes.plot(epochs, data_history[metric_validation], label="Validation", marker=marker)

    axes.set_xlabel("Epoch")
    axes.set_ylabel("Loss")

    axes.set_xticks(epochs)

    if y_limits is not None:
        axes.set_ylim(y_limits)

    if x_ticks is not None:
        axes.set_xticks(x_ticks)

    if y_ticks is not None:
        axes.set_yticks(y_ticks)

    axes.legend(loc=location_legend)

    return figure, axes


def build_model_baseline(
    x_training: Any,
    vocabulary_size: int,
    maximum_sequence_length: int | None,
    standardization_method: str | None,
    dimensions_embedding: int,
    units_dense: int,
) -> tf.keras.models.Sequential:
    """
    Builds the baseline model

    Parameters
    ----------
    x_training : Any
        List/array of the training values
    vocabulary_size : int
        Maximum amount of tokens to be known to the model
    maximum_sequence_length : int | None
        Length to which to truncate and pad sequences to
    standardization_method : str | None
        {None, "lower_and_strip_punctuation", "lower", "strip_punctuation"}
    dimensions_embedding : int
        Dimensions of the embedding layer
    units_dense : int
        Number of neurons in the hidden dense layer

    Returns
    -------
    tf.keras.models.Sequential
        Model ready to be compiled
    """
    layer_vectorization = tf.keras.layers.TextVectorization(
        max_tokens=vocabulary_size,
        standardize=standardization_method,
        output_sequence_length=maximum_sequence_length,
    )

    layer_vectorization.adapt(x_training)

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Input(shape=(1,), dtype=tf.string),
            layer_vectorization,
            tf.keras.layers.Embedding(
                input_dim=vocabulary_size,
                output_dim=dimensions_embedding,
            ),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(
                units=units_dense, activation="relu", name="hidden_relu"
            ),
            tf.keras.layers.Dense(units=1, activation="sigmoid", name="output_sigmoid"),
        ],
        name="baseline_model",
    )

    return model


def build_model_lstm(
    x_training: Any,
    vocabulary_size: int,
    maximum_sequence_length: int | None,
    standardization_method: str | None,
    dimensions_embedding: int,
    units_lstm_1: int,
    units_lstm_2: int,
    units_lstm_3: int,
    units_dense: int,
) -> tf.keras.models.Sequential:
    """
    Builds the Long Short-Term Memory model

    Parameters
    ----------
    x_training : Any
        List/array of the training values
    vocabulary_size : int
        Maximum amount of tokens to be known to the model
    maximum_sequence_length : int | None
        Length to which to truncate and pad sequences to
    standardization_method : str | None
        {None, "lower_and_strip_punctuation", "lower", "strip_punctuation"}
    dimensions_embedding : int
        Dimensions of the embedding layer
    units_lstm_1 : int
        Number of neurons in the first LSTM layer
    units_lstm_2 : int
        Number of neurons in the second LSTM layer
    units_lstm_3 : int
        Number of neurons in the third LSTM layer
    units_dense : int
        Number of neurons in the hidden dense layer

    Returns
    -------
    tf.keras.models.Sequential
        Model ready to be compiled
    """
    layer_vectorization = tf.keras.layers.TextVectorization(
        max_tokens=vocabulary_size,
        standardize=standardization_method,
        output_sequence_length=maximum_sequence_length,
    )

    layer_vectorization.adapt(x_training)

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Input(shape=(1,), dtype=tf.string),
            layer_vectorization,
            tf.keras.layers.Embedding(
                input_dim=vocabulary_size,
                output_dim=dimensions_embedding,
                mask_zero=True,
            ),
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(units=units_lstm_1, return_sequences=True),
                name="bidirectional_lstm_1",
            ),
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(units=units_lstm_2, return_sequences=True),
                name="bidirectional_lstm_2",
            ),
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(units=units_lstm_3),
                name="bidirectional_lstm_3",
            ),
            tf.keras.layers.Dense(
                units=units_dense, activation="relu", name="hidden_relu"
            ),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(units=1, activation="sigmoid", name="output_sigmoid"),
        ],
        name="lstm_model",
    )

    return model


def build_model_nnlm(
    embedding_dimensions: int,  # can be 50 or 128
    units_dense: int,
    is_trainable: bool = True,
) -> tf_keras.models.Sequential:
    """
    Builds the model with pre-trained embeddings from a Neural-Net Language Model

    Parameters
    ----------
    embedding_dimensions : int
        {50, 128}, dimensions of the embedding layer
    is_trainable : bool, optional
        Whether or not the pre-trained embedding layer is trainable, by default True

    Returns
    -------
    tf_keras.models.Sequential
        Model ready to be compiled
    """
    hub_layer = hub.KerasLayer(
        f"https://www.kaggle.com/models/google/nnlm/TensorFlow2/de-dim{embedding_dimensions}/1",
        input_shape=[],
        dtype=tf.string,
        trainable=is_trainable,
    )

    # tf_keras is used instead of tf.keras, because tensorflow_hub (Version 0.16.1) is incompatible with Keras 3
    model = tf_keras.models.Sequential(
        [
            hub_layer,
            tf_keras.layers.Dense(
                units=units_dense, activation="relu", name="hidden_relu"
            ),
            tf_keras.layers.Dense(units=1, activation="sigmoid", name="output_sigmoid"),
        ]
    )

    return model


def evaluate_model(
    model: (
        tf.keras.models.Sequential
        | tf_keras.models.Sequential
        | transformers.TFBertForSequenceClassification
    ),
    x_test: Any,
    y_test: Any,
) -> tuple:
    """
    Evaluates how accurately a given model predicts data from the test set

    Parameters
    ----------
    model : tf.keras.models.Sequential | tf_keras.models.Sequential | TFBertForSequenceClassification
        Model to be evaluated
    x_test : Any
        List/array/dataset of the test values
    y_test : Any
        List/array of the test labels

    Returns
    -------
    tuple
        y_predicted, accuracy
    """
    y_predicted = model.predict(x_test)
    
    if type(y_predicted) == transformers.modeling_tf_outputs.TFSequenceClassifierOutput:
        y_predicted = tf.nn.sigmoid(y_predicted["logits"])
        
    y_predicted = np.round(y_predicted)

    accuracy = sklearn.metrics.accuracy_score(y_true=y_test, y_pred=y_predicted)

    return y_predicted, float(accuracy)


def plot_accuracy(
    data_history: pd.DataFrame,
    y_limits: tuple | None = None,
    x_ticks: list | None = None,
    y_ticks: list | None = None,
    location_legend: str = "lower right",
    font_size: int = 18,
    width_in_inches: float = 10,
    height_in_inches: float = 6,
) -> tuple:
    """
    Creates a plot showing training and validation accuracy over the course of the training epochs

    Parameters
    ----------
    data_history : pd.DataFrame
        Data frame containing the training history of the model
    y_limits : list | None, optional
        Limits of the y axis
    x_ticks : list | None, optional
        Ticks of the x axes
    y_ticks : list | None, optional
        Ticks of the y axes
    location_legend : str, optional
        {"upper left", "upper right", "lower left", "lower right"}, location of the legend, by default "lower right"
    font_size : int, optional
        Base size of the font, passed to plt.rcParams["font_size"], by default 18
    width_in_inches : float, optional
        Width of the figure in inches, by default 10
    height_in_inches : float, optional
        Height of the figure in inches, by default 6

    Returns
    -------
    tuple
        (figure, axes)
    """
    
    figure, axes = _plot_history(
        data_history=data_history,
        metric_training="binary_accuracy",
        metric_validation="val_binary_accuracy",
        y_limits=y_limits,
        x_ticks=x_ticks,
        y_ticks=y_ticks,
        location_legend=location_legend,
        font_size=font_size,
        width_in_inches=width_in_inches,
        height_in_inches=height_in_inches,
    )
    
    return figure, axes


def plot_confusion_matrix(
    y_true: Any,
    y_predicted: Any,
    title: str | None = None,
    font_size: int = 18,
    width_in_inches: float = 7,
    height_in_inches: float = 6,
):
    """
    Creates a plot showing the confusion matrix for the given true and predicted labels

    Parameters
    ----------
    y_true : Any
        List/array of the true labels
    y_predicted : Any
        List/array of the predicted labels
    title : str | None, optional
        Title to add to the figure, by default None
    font_size : int, optional
        Base size of the font, passed to plt.rcParams["font_size"], by default 18
    width_in_inches : float, optional
        Width of the figure in inches, by default 7
    height_in_inches : float, optional
        Height of the figure in inches, by default 6
    
    Returns
    -------
    tuple
        (figure, axes)
    """
    plt.rcParams["font.size"] = font_size
    
    confusion_matrix = sklearn.metrics.confusion_matrix(
        y_true=y_true,
        y_pred=y_predicted,
        normalize="all",
    )
    
    display = sklearn.metrics.ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix,
        display_labels=["Regular", "Plain"],
    )
    
    display.plot()
    
    display.figure_.set_size_inches(width_in_inches, height_in_inches)
    
    display.ax_.set_xlabel("Predicted Label")
    display.ax_.set_ylabel("True Label")
    
    if title is not None:
        display.ax_.set_title(title)
    
    return display.figure_, display.ax_


def plot_loss(
    data_history: pd.DataFrame,
    y_limits: tuple | None = None,
    x_ticks: list | None = None,
    y_ticks: list | None = None,
    location_legend: str = "upper right",
    font_size: int = 18,
    width_in_inches: float = 10,
    height_in_inches: float = 6,
) -> tuple:
    """
    Creates a plot showing training and validation loss over the course of the training epochs

    Parameters
    ----------
    data_history : pd.DataFrame
        Data frame containing the training history of the model
    y_limits : list | None, optional
        Limits of the y axis
    x_ticks : list | None, optional
        Ticks of the x axes
    y_ticks : list | None, optional
        Ticks of the y axes
    location_legend : str, optional
        {"upper left", "upper right", "lower left", "lower right"}, location of the legend, by default "upper right"
    font_size : int, optional
        Base size of the font, passed to plt.rcParams["font_size"], by default 18
    width_in_inches : float, optional
        Width of the figure in inches, by default 10
    height_in_inches : float, optional
        Height of the figure in inches, by default 6

    Returns
    -------
    tuple
        (figure, axes)
    """
    
    figure, axes = _plot_history(
        data_history=data_history,
        metric_training="loss",
        metric_validation="val_loss",
        y_limits=y_limits,
        x_ticks=x_ticks,
        y_ticks=y_ticks,
        location_legend=location_legend,
        font_size=font_size,
        width_in_inches=width_in_inches,
        height_in_inches=height_in_inches,
    )
    
    return figure, axes


def tokenize_text_gbert(
    texts: List[str],
    labels: List[int],
    maximum_sequence_length: int,
    model_name: str = "deepset/gbert-base",
) -> tf.data.Dataset:
    """
    Prepares tokenized datasets of texts and labels for the German BERT model

    Parameters
    ----------
    texts : List[str]
        List of the text values
    labels : List[int]
        List of the labels
    maximum_sequence_length : int
        Length to which to truncate and pad sequences to
    model_name : str, optional
        Name of the pre-trained model to load, by default "deepset/gbert-base"

    Returns
    -------
    tf.data.Dataset
        Dataset containing the encoded texts and the labels
    """
    tokenizer = transformers.BertTokenizer.from_pretrained(model_name)

    encoded_texts = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=maximum_sequence_length,
    )

    data = tf.data.Dataset.from_tensor_slices((dict(encoded_texts), labels))

    return data
