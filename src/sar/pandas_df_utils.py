from functools import lru_cache, wraps
import logging

import pandas as pd
import numpy as np

from constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_LABEL_COL,
)

logger = logging.getLogger(__name__)

def has_columns(df, columns):
    """Check if DataFrame has necessary columns
    Args:
        df (pd.DataFrame): DataFrame
        columns (list(str): columns to check for
    Returns:
        bool: True if DataFrame has specified columns
    """

    result = True
    for column in columns:
        if column not in df.columns:
            logger.error("Missing column: {} in DataFrame".format(column))
            result = False

    return result

def has_same_base_dtype(df_1, df_2, columns=None):
    """Check if specified columns have the same base dtypes across both DataFrames
    Args:
        df_1 (pd.DataFrame): first DataFrame
        df_2 (pd.DataFrame): second DataFrame
        columns (list(str)): columns to check, None checks all columns
    Returns:
        bool: True if DataFrames columns have the same base dtypes
    """

    if columns is None:
        if any(set(df_1.columns).symmetric_difference(set(df_2.columns))):
            logger.error(
                "Cannot test all columns because they are not all shared across DataFrames"
            )
            return False
        columns = df_1.columns

    if not (
        has_columns(df=df_1, columns=columns) and has_columns(df=df_2, columns=columns)
    ):
        return False

    result = True
    for column in columns:
        if df_1[column].dtype.type.__base__ != df_2[column].dtype.type.__base__:
            logger.error("Columns {} do not have the same base datatype".format(column))
            result = False

    return result

def lru_cache_df(maxsize, typed=False):
    """Least-recently-used cache decorator for pandas Dataframes. 
    
    Decorator to wrap a function with a memoizing callable that saves up to the maxsize most recent calls. It can 
    save time when an expensive or I/O bound function is periodically called with the same arguments.
    Inspired in the `lru_cache function <https://docs.python.org/3/library/functools.html#functools.lru_cache>`_.
    Args:
        maxsize (int|None): max size of cache, if set to None cache is boundless
        typed (bool): arguments of different types are cached separately
    """

    def to_pandas_hash(val):
        """Return PandaHash object if input is a DataFrame otherwise return input unchanged"""
        return PandasHash(val) if isinstance(val, pd.DataFrame) else val

    def from_pandas_hash(val):
        """Extract DataFrame if input is PandaHash object otherwise return input unchanged"""
        return val.pandas_object if isinstance(val, PandasHash) else val

    def decorating_function(user_function):
        @wraps(user_function)
        def wrapper(*args, **kwargs):
            # convert DataFrames in args and kwargs to PandaHash objects
            args = tuple([to_pandas_hash(a) for a in args])
            kwargs = {k: to_pandas_hash(v) for k, v in kwargs.items()}
            return cached_wrapper(*args, **kwargs)

        @lru_cache(maxsize=maxsize, typed=typed)
        def cached_wrapper(*args, **kwargs):
            # get DataFrames from PandaHash objects in args and kwargs
            args = tuple([from_pandas_hash(a) for a in args])
            kwargs = {k: from_pandas_hash(v) for k, v in kwargs.items()}
            return user_function(*args, **kwargs)

        # retain lru_cache attributes
        wrapper.cache_info = cached_wrapper.cache_info
        wrapper.cache_clear = cached_wrapper.cache_clear

        return wrapper

    return decorating_function