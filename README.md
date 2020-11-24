# InterACT Lab self-driving car env
![Merging (0.5x speed)](merging.gif)

## Requirements
```
python>=3.6
```

## Installation

```
pip install -e .
```


### Checking your installation
To test that your code works properly, run:
```
python -m unittest interact_drive/reward_inference/tests/test_first_order_ioc.py
```
(This takes about 1-2 minutes locally.)

For a small demo, try running:
```
python experiments/merging.py
```

### Pyglet issues on Mac OS
This package uses `pyglet` to visualize the driving scenarios.

Unfortunately, it turns out that `pyglet` is broken in some of the recent Mac OS
versions. If you get the following message when closing a visualization window:
 
```
ctypes.ArgumentError: argument 1: <class 'RecursionError'>: maximum recursion depth exceeded in __instancecheck__
```
you may wish to comment out 
[`PygletWindow.nextEventMatchingMask_untilDate_inMode_dequeue_`](https://github.com/pyglet/pyglet/blob/ee3a6a739de13e2abe649ff99c8ce4dd59a1f84c/pyglet/window/cocoa/pyglet_window.py)
in your local `pyglet` installation. 

## Getting started
A driving scenario generally consists of a `CarWorld` (located in `world.py`)
and one or more `Car`s (see the `car/` directory). In addition, the more 
interesting cars contain a `Planner` (located in the `planner/` directory). 

Currently, the only nontrivial `Planner` is `NaivePlanner`, which assumes all
the other cars travel forward at their current velocity forever.

### Reward Inference
In addition to experimenting with different driving scenarios, this codebase
supports performing reward inference from trajectory demonstrations 
(located in the `reward_inference/` directory). 

Currently two reward inference procedures have been implemented. 
The first is a gradient-only variant of inverse (locally) optimal control,
located in 
`reward_inference/first_order_ioc.py`.

The second is Levine and Koltun's CIOC algorithm, located in
`reward_inference/second_order_ioc.py`. 
Note that this algorithm assumes that the demonstrator is Boltzmann-rational. 

That being said, by simply increasing the rationality of the Boltzmann-rational
demonstrator, the inferred weights for CIOC approach those of the first-order 
method:
![l2_vs_beta](l2_vs_beta.png)

This graph can be replicated by running the
[Sacred](https://sacred.readthedocs.io/) experiment 
`experiments/cioc_beta_experiment.py`. Note that as a Sacred experiment, this 
experiment has some nice logging support. To see this in action, try running:
```
python cioc_beta_experiment.py -F /path/to/log/dir
```

## Known issues
* `state` should probably be its own class, as should `control`.
That being said, Tensorflow 2.0 doesn't like it when I try. 
* `CarVisualizer` is still missing some functionality compared to the 
legacy `visualizer`. Notably, it doesn't allow you to visualize heatmaps.
* Relatedly,`CarWorld` and `CarVisualizer` only support straight roads. 
* No good support for sharing features.
* `unittest` is currently being used for testing. `pytest` seems just better, 
and we should probably migrate to it. 

## Contributing
### Code style
Please follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) and the 
[Google Python Style Guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md). 
We recommend using an IDE or pylint to ensure that contributions conform to the 
existing code style. 

### Tests
Please write test cases for your code. Place your `unittest` files in a `tests/`
directory at the same level of the module containing the code to be tested. 

### Notable design differences from other similar codebases
* 3+ agent environments are supported natively.
* Both reward design and multi-agent experiments are supported.
* Reward functions can be arbitrary functions of state.
* Features for linear reward functions are now owned by subclasses of `Car`
instead of the `world`/`env`. 
* Everything is done in `tensorflow>=2.0` - no more maintaining a seperate `np` 
and `tf`/`theano` version of each method/value. 
* PEP 8/Google style guide compliant.

### Branches/Pull Requests
Please make pull requests to the `dev` branch when ready, and ask an InterACT 
or CHAI graduate student to review your code. 