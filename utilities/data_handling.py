import json
import os
from typing import Any, List

import nltk
import pandas as pd
from sklearn.model_selection import train_test_split


def _get_data_from_directory(path: str) -> pd.DataFrame:
    """
    Helper function that extracts all data from a given source from the Toborek et al. (2023) dataset.

    Parameters
    ----------
    path : str
        Path to the directory of the data source

    Returns
    -------
    pd.DataFrame
        Data frame containing the data from the specified source/directory
    """
    with open(
        os.path.join(path, "parsed_header.json"), mode="r", encoding="UTF-8"
    ) as file:
        datastore = json.load(file)

    data_total = _initialize_empty_data_frame()

    for filename_html in datastore:
        current_entry = datastore[filename_html]

        filename_txt = filename_html + ".txt"
        is_plain_language = current_entry["easy"]

        data_current_file = _get_data_from_file(
            filepath=os.path.join(path, "parsed", filename_txt),
            is_plain_language=is_plain_language,
        )

        data_total = pd.concat([data_total, data_current_file], ignore_index=True)

    return data_total


def _get_data_from_file(filepath: str, is_plain_language: bool) -> pd.DataFrame:
    """
    Helper function that extracts the data from a file for a given article from the Toborek et al. (2023) dataset.

    Parameters
    ----------
    filepath : str
        Path to the article file
    is_plain_language : bool
        Whether or not the article is in plain language

    Returns
    -------
    pd.DataFrame
        Pandas data frame containing the data from the article file
    """
    data = pd.read_csv(
        filepath_or_buffer=filepath,
        delimiter="<NONE>",
        header=None,
        names=["text"],
        dtype="str",
        engine="python",
    )

    if is_plain_language:
        data["is_plain_language"] = 1
    else:
        data["is_plain_language"] = 0

    data["is_plain_language"] = data["is_plain_language"].astype("int")

    return data


def _initialize_empty_data_frame() -> pd.DataFrame:
    """
    Helper function that initializes an empty data of the required format.

    Returns
    -------
    pd.DataFrame
        Data frame with 0 rows and the required columns and data types
    """
    data = pd.DataFrame(
        {
            "text": pd.Series([], dtype="str"),
            "is_plain_language": pd.Series([], dtype="int"),
        }
    )

    return data


def load_data(has_duplicate_removal: bool = True) -> pd.DataFrame:
    """
    Loads the data of sentences in regular and in plain language with corresponding labels.

    Parameters
    ----------
    has_duplicate_removal : bool, optional
        Whether or not to drop duplicate entries, by default True

    Returns
    -------
    pd.DataFrame
        Data frame with a column for the sentences ("text") and the labels ("is_plain_language")
    """
    data_total = _initialize_empty_data_frame()

    path = os.path.join("..", "data")

    directories_sources = os.listdir(path)

    for directory in directories_sources:
        data_current_directory = _get_data_from_directory(
            path=os.path.join(path, directory)
        )
        data_total = pd.concat([data_total, data_current_directory], ignore_index=True)

    if has_duplicate_removal:
        data_total = data_total.drop_duplicates()

    return data_total


def prepare_and_split_data(
    data: pd.DataFrame | None = None,
    has_duplicate_removal: bool = True,
    has_stop_word_removal: bool = True,
    validation_size: float = 0.1,
    test_size: float = 0.2,
) -> tuple:
    """
    Wrapper function that loads and prepares the data and splits it into training, validation, and test sets.

    Parameters
    ----------
    data : pd.DataFrame | None, optional
        Data frame to prepare, if None, data are loaded automatically, by default None
    has_duplicate_removal : bool, optional
        Whether or not to drop duplicate entries, by default True
    has_stop_word_removal : bool, optional
        Whether or not to remove stop words, by default True
    validation_size : float, optional
        Size of the validation dataset, either as a percentage or as an absolute value. Percentages are interpreted relative to the total size of the dataset, by default 0.1
    test_size : float, optional
        Size of the test dataset, either as a percentage or as an absolute value. Percentages are interpreted relative to the total size of the dataset, by default 0.2

    Returns
    -------
    tuple
        (x_training, x_validation, x_test, y_training, y_validation, y_test)
    """
    if data is None:
        data = load_data(has_duplicate_removal=has_duplicate_removal)

    if has_stop_word_removal:
        data["text"] = remove_stop_words(data["text"])

    x_training, x_validation, x_test, y_training, y_validation, y_test = (
        split_training_validation_test(
            x=data["text"],
            y=data["is_plain_language"],
            validation_size=validation_size,
            test_size=test_size,
            random_state=42,
        )
    )

    return x_training, x_validation, x_test, y_training, y_validation, y_test


def remove_stop_words(x: Any, language: str = "german") -> List[str]:
    """
    Removes stop words from a given list/array of texts using the nltk package.

    Parameters
    ----------
    x : Any
        List/array of string values
    language : str, optional
        Language for which stop words should be removed, by default "german"

    Returns
    -------
    List[str]
        List containing the values of x, without the stop words of the specified language
    """
    nltk.download("stopwords", quiet=True)

    stop_words = set(nltk.corpus.stopwords.words(language))

    return [
        " ".join([word for word in sentence.split() if word.lower() not in stop_words])
        for sentence in x
    ]


def split_training_validation_test(
    x: Any,
    y: Any,
    validation_size: float,
    test_size: float,
    *args,
    **kwargs,
) -> tuple:
    """
    Wrapper function that splits values x and labels y into training, validation, and test sets, relying on functionality from sklearn.model_selection.train_test_split()

    Parameters
    ----------
    x : Any
        List/array of values
    y : Any
        List/array of labels
    validation_size : float
        Size of the validation dataset, either as a percentage or as an absolute value. Percentages are interpreted relative to the total size of the dataset.
    test_size : float
        Size of the test dataset, either as a percentage or as an absolute value. Percentages are interpreted relative to the total size of the dataset.

    Returns
    -------
    tuple
        (x_training, x_validation, x_test, y_training, y_validation, y_test)
    """
    number_of_values = len(x)

    if validation_size < 1:
        validation_size = round(validation_size * number_of_values)

    if test_size < 1:
        test_size = round(test_size * number_of_values)

    x_main, x_test, y_main, y_test = train_test_split(
        x, y, test_size=test_size, *args, **kwargs
    )

    x_training, x_validation, y_training, y_validation = train_test_split(
        x_main, y_main, test_size=validation_size, *args, **kwargs
    )

    return x_training, x_validation, x_test, y_training, y_validation, y_test
