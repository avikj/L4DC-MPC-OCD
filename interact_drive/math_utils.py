"""Defines useful functions for feature-based cars."""

import tensorflow as tf
import numpy as np


@tf.function
def _f(x, shape=5.):
    """
    A (non-analytic) smooth function that is 0 for x <= 0 and > 0 for x > 0.
    shape is a parameter that affects function shape.
    Useful for tuning the shape of functions that depend on _f.
    Used to define some of the smooth functions below.

    Higher values of l make the slope of _f steeper.

    In fact, we have _f(x) -> 1 as x -> infinity.

    >>> _f(tf.constant(0.)).numpy()
    0.0
    >>> (_f(tf.constant(1.)) > 0).numpy()
    True
    >>> (_f(tf.constant(0.01)) > 0).numpy()
    True
    >>> (_f(tf.constant(1e10)).numpy() - 1) < 1e-5
    True
    """
    x_clipped = tf.where(x > 0, x, tf.zeros_like(x) + 0.01)
    return tf.where(
        x > 0, tf.exp(-1 / (shape * x_clipped)), tf.zeros_like(x)
    )


def gaussian(mean, stddev):
    """
    Returns a Gaussian pdf.

    Args:
        mean: `float`, the mean of the Gaussian.
        stddev: `float`, the standard deviation of the Gaussian

    Returns:
        (function): a function g such that g(x) = e^(-x^2)
    """
    if stddev <= 0:
        raise ValueError("stddev {} must be greater than 0".format(stddev))

    @tf.function
    def g(x):
        return (
            1 / ((2 * np.pi) ** 0.5 * stddev)
            * tf.exp(
                -tf.reduce_sum((x - mean) ** 2, axis=-1) / (2 * stddev ** 2))
        )

    return g


def smooth_threshold(threshold, width, c=5.):
    """
    Returns a function that is 1. for x > threshold, 0. for
    x < threshold - width, and is smooth
    (that is, is infinitely continuously differentiable).

    >>> t = smooth_threshold(0., 1.)
    >>> t(tf.constant(0.)).numpy()
    1.0
    >>> t(tf.constant(-1.)).numpy()
    0.0
    >>> t(tf.constant(-0.5)).numpy()
    0.5

    Args:
        threshold: `float`, the threshold
        width: `float`, the width of the "step". Must be positive if this
                function is to be smooth.
        tf: a mathematical computation tf (likely either np or tf)
        c: shape tuning parameter. Higher c makes the function steeper at the
                edges and less steep
            in in the middles
    Returns:
        (function): a smooth function t such that t(x) = 1 if x > threshold and
                0 for x < threshold - width.
    """

    if width <= 0:
        raise ValueError("Width {} must be greater than 0".format(width))
    shape = tf.constant(c / width)

    @tf.function
    def t(x):
        x_diff = x - (threshold - width)
        return _f(x_diff, shape) / (
            _f(x_diff, shape) + _f(width - x_diff, shape)
        )

    return t


# This is terrible for optimization, so I don't recommend it.
def smooth_plateau(start, end, width):
    """
    Returns a function that is 1 between start and end, 0 outside of
            (start-width, end-width), and smooth.

    Args:
        start: `float`, the start of the plateau
        end: `float`, the end of the plateau
        width: the width of the edges of the plateau
        tf: a mathematical computation tf (likely either np or tf)

    Returns:
        (function): a smooth function pl such that pl(x) = 1 if start < x < end

    """

    if start > end:
        raise ValueError(
            "Start value {} cannot be greater than end value {}".format(start,
                                                                        end)
        )

    if width <= 0:
        raise ValueError("Width {} must be greater than 0".format(width))

    @tf.function
    def pl(x):
        t_start = smooth_threshold(start, width)
        t_end = smooth_threshold(end, width)
        return t_start(x) - t_end(x)

    return pl


def smooth_bump(start, end):
    """
    Returns a smooth function that is 0 on (-infty, start] and [end, infty)
            and is 1 at (start + end)/2.

    >>> bmp = smooth_bump(-1., 1.)
    >>> bmp(tf.constant(0.)).numpy()
    1.0
    >>> bmp(tf.constant(-1.0)).numpy()
    0.0
    >>> bmp(tf.constant(1.0)).numpy()
    0.0
    >>> bmp(tf.constant(0.5)).numpy() > 0
    True

    Args:
        start (float): the point at which the bump starts
        end (float): the point at which the bump ends.

    Returns:
        (function): a smooth bump function bmp
    """
    """
    if start > end:
        raise ValueError(
            "Start value {} cannot be greater than end value {}".format(start,
                                                                        end)
        )
    """
    start = tf.identity(start)
    end = tf.identity(end)

    @tf.function
    def bmp(x):
        width = (end - start) / 2
        center = (start + end) / 2
        x_norm = (x - center) / width
        cond = tf.square(x_norm) < 1
        x_norm_clipped = tf.where(
            cond, x_norm, 0.0
        )  # needed to prevent NaN gradients due to how tf handles control flow
        return tf.where(
            cond, tf.exp(-1 / (1 - x_norm_clipped ** 2) + 1), 0
        )

    return bmp
