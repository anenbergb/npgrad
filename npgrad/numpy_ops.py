from scipy.stats import truncnorm
import numpy as np


def trunc_normal(shape, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """
    Returns a NumPy array of shape `shape` sampled from a truncated normal distribution.

    Truncation limits `a` and `b` are expressed in standard deviations from the mean.
    """
    # Convert a and b from "abs stddevs from mean" to truncnorm's required format
    lower = (a - mean) / std
    upper = (b - mean) / std

    return truncnorm.rvs(lower, upper, loc=mean, scale=std, size=shape)
