from typing import Any, List

import numpy as np
import sklearn.metrics
import tensorflow as tf
import tensorflow_hub as hub
import tf_keras
from matplotlib import pyplot as plt
from transformers import (
    BertTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    TFBertForSequenceClassification,
)


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


def build_model_gbert(
    x_training: List[str],
    x_validation: List[str],
    y_training: List[int],
    y_validation: List[int],
    maximum_sequence_length: int,
    model_name: str = "deepset/gbert-base",
) -> tuple:
    """
    Builds the fine-tuned German BERT model

    Parameters
    ----------
    x_training : List[str]
        List of the training values
    x_validation : List[str]
        List of the validation values
    y_training : List[int]
        List of the training labels
    y_validation : List[int]
        List of the validation labels
    maximum_sequence_length : int
        Length to which to truncate and pad sequences to
    model_name : str, optional
        Name of the pre-trained model to load, by default "deepset/gbert-base"

    Returns
    -------
    tuple
        (data_training, data_validation, model)
    """
    tokenizer = BertTokenizer.from_pretrained(model_name)

    def tokenize_text(
        texts: List[str],
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        maximum_sequence_length: int,
    ):
        encoded_texts = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=maximum_sequence_length,
        )

        return encoded_texts

    encodings_training = tokenize_text(
        texts=x_training,
        tokenizer=tokenizer,
        maximum_sequence_length=maximum_sequence_length,
    )
    encodings_validation = tokenize_text(
        texts=x_validation,
        tokenizer=tokenizer,
        maximum_sequence_length=maximum_sequence_length,
    )

    data_training = tf.data.Dataset.from_tensor_slices(
        (dict(encodings_training), y_training)
    )
    data_validation = tf.data.Dataset.from_tensor_slices(
        (dict(encodings_validation), y_validation)
    )

    model = TFBertForSequenceClassification.from_pretrained(model_name, num_labels=1)

    return data_training, data_validation, model


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
        | TFBertForSequenceClassification
    ),
    x_test: Any,
    y_test: Any,
) -> float:
    """
    Evaluates how accurately a given model predicts data from the test set

    Parameters
    ----------
    model : tf.keras.models.Sequential | tf_keras.models.Sequential | TFBertForSequenceClassification
        Model to be evaluated
    x_test : Any
        List/array of the test values
    y_test : Any
        List/array of the test labels

    Returns
    -------
    float
        Accuracy score
    """
    y_predicted = np.round(model.predict(x_test))

    accuracy = sklearn.metrics.accuracy_score(y_true=y_test, y_pred=y_predicted)

    return float(accuracy)


def plot_accuracy(
    history: tf.keras.callbacks.History,
    y_limits: tuple | None = None,
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
    history : tf.keras.callbacks.History
        Training history of the model
    y_limits : list | None, optional
        Limits of the y axis
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
    plt.rcParams["font.size"] = font_size

    figure, axes = plt.subplots()

    figure.set_size_inches(width_in_inches, height_in_inches)

    epochs = [epoch + 1 for epoch in history.epoch]

    axes.plot(epochs, history.history["binary_accuracy"], label="Training")
    axes.plot(epochs, history.history["val_binary_accuracy"], label="Validation")

    axes.set_xlabel("Epoch")
    axes.set_ylabel("Accuracy")

    axes.set_xticks(epochs)

    if y_limits is not None:
        axes.set_ylim(y_limits)

    if y_ticks is not None:
        axes.set_yticks(y_ticks)

    axes.legend(loc=location_legend)

    return figure, axes


def plot_loss(
    history: tf.keras.callbacks.History,
    y_limits: list | None = None,
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
    history : tf.keras.callbacks.History
        Training history of the model
    y_limits : list | None, optional
        Limits of the y axis
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
    plt.rcParams["font.size"] = font_size

    figure, axes = plt.subplots()

    figure.set_size_inches(width_in_inches, height_in_inches)

    epochs = [epoch + 1 for epoch in history.epoch]

    axes.plot(epochs, history.history["loss"], label="Training")
    axes.plot(epochs, history.history["val_loss"], label="Validation")

    axes.set_xlabel("Epoch")
    axes.set_ylabel("Loss")

    axes.set_xticks(epochs)

    if y_limits is not None:
        axes.set_ylim(y_limits)

    if y_ticks is not None:
        axes.set_yticks(y_ticks)

    axes.legend(loc=location_legend)

    return figure, axes
