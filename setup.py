from setuptools import setup

setup(
    name="interact_drive",
    install_requires=[
        "cma",
        # "robo",
        "gym",
        # "flake8",  # linter
        "numpy",
        "matplotlib",
        "moviepy",
        "pyglet>=1.4",  # visualization
        # "pymongo",  # needed for sacred, which refuses to install this
        # "cvxpylayers",  # for IRL
        # "sacred",  # for experiment logging
        "seaborn",
        "tensorflow>=2.0",
    ],
    version='0.1.0',
    author='Lawrence Chan',
    author_email='chanlaw@berkeley.edu',
    description='Implementation of InterACT lab self-driving car algorithms',
    license='MIT'
)
