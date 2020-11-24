"""(Currently unused) utilities for sharing features between cars."""

import tensorflow as tf


class TFFeature(object):
    """
    This is a wrapper that lets us do math with (Tensorflow) functions.
    For example:

    >>> squared = TFFeature(tf.function(lambda x: x**2))
    >>> squared(3).numpy()
    9
    >>> (squared + 2)(3).numpy()
    11
    >>> (-2*squared)(3).numpy()
    -18

    Basically, we're implementing some Tensorflow 1.x functionality in TF 2.x.
    """

    def __init__(self, f: tf.function):
        self.f = f

    def __call__(self, *args):
        return self.f(*args)

    def __add__(self, r):
        @tf.function
        def summed(*args):
            return self.f(*args) + r

        return TFFeature(summed)

    def __radd__(self, r):
        @tf.function
        def summed(*args):
            return r + self.f(*args)

        return TFFeature(summed)

    def __mul__(self, r):
        @tf.function
        def prod(*args):
            return self.f(*args) * r

        return TFFeature(prod)

    def __rmul__(self, r):
        @tf.function
        def prod(*args):
            return r * self.f(*args)

        return TFFeature(prod)

    def __pos__(self, r):
        return self

    def __neg__(self):
        return TFFeature(tf.function(lambda *args: -self(*args)))

    def __sub__(self, r):
        return TFFeature(tf.function(lambda *args: self(*args) - r))

    def __rsub__(self, r):
        return TFFeature(tf.function(lambda *args: r - self(*args)))

    def __div__(self, r):
        @tf.function
        def ratio(*args):
            return self.f(*args) / r

        return TFFeature(ratio)

    def __rdiv__(self, r):
        @tf.function
        def ratio(*args):
            return r / self.f(*args)

        return TFFeature(ratio)

    def compose_with(self, g):
        @tf.function
        def composed_f(*args):
            return self.f(g(*args))

        return TFFeature(composed_f)
