from merging import ThreeLaneTestCar
from dont_crash import setup_world, is_in_right_lane, reaches_target_speed, doesnt_collide
import numpy as np
from bayes_opt import BayesianOptimization
import pickle
import contextlib
with contextlib.redirect_stdout(None):  # disable the pygame import print
    from moviepy.editor import ImageSequenceClip
FILENAME = 'bayesopt_results.pkl'

results = pickle.load(open(FILENAME, 'rb'))
our_car, other_car, world = setup_world()

for result in results:
	if result['score'] >= 8:
		print("Writing gif for ", result)
		frames = []
		for init_x in [0.0, 0.1/3, 0.2/3, 0.1]:
			our_car.init_state = np.array([init_x, our_car.init_state[1], our_car.init_state[2], our_car.init_state[3]], dtype=np.float32)
			our_car.weights = np.array(result['weights'], dtype=np.float32)
			world.reset()
			frames.append(world.render("rgb_array"))

			for i in range(15):
				world.step()
				frames.append(world.render("rgb_array"))

		clip = ImageSequenceClip(frames, fps=int(1 / world.dt))
		clip.speedx(0.5).write_gif("dont_crash_s_%d_w_%s.gif"%(int(result['score']), str(list(result['weights']))), program="ffmpeg")