import gym
import matplotlib.pyplot as plt


env = gym.make('Pong-v0')
obs = env.reset()
plt.imshow(env.render(mode='rgb_array'))
plt.show()

env = gym.make('Pong-v0')
obs = env.reset()
video = [] # array to store state space at each step
n_frames = 300
for _ in range(n_frames):
    video.append(env.render(mode='rgb_array'))
    obs, reward, done, _ = env.step(env.action_space.sample())
    if done:
        break
import matplotlib.animation as animation
from JSAnimation.IPython_display import display_animation
from IPython.display import display

fig = plt.figure()
patch = plt.imshow(video[0])
plt.axis('off')


def animate(idx):
    patch.set_data(video[idx])


anim = animation.FuncAnimation(plt.gcf(), animate,
                               frames=len(video), interval=30)

display(display_animation(anim))