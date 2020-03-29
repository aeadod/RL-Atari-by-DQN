import os
from tqdm import tqdm
import numpy as np

import gym
from gym.wrappers import AtariPreprocessing
from gym.wrappers import FrameStack
from expreplay import ReplayMemory, Experience
from DQN_agent import DQNModel

MEMORY_SIZE = 1e6
MEMORY_WARMUP_SIZE = MEMORY_SIZE // 20
IMAGE_SIZE = (84, 84)
CONTEXT_LEN = 4
GAMMA = 0.99

ACTION_REPEAT = 4  # aka FRAME_SKIP
UPDATE_FREQ = 4
SYNC_TARGET_FREQ = 10000 // UPDATE_FREQ

BATCH_SIZE = 64
USE_CUDA = True

def action_random(env):
    action = env.action_space.sample()
    return action

def action_policy(agent, state, exp):
    context = exp.recent_state()
    context.append(state)
    context = np.stack(context, axis=0)
    action = agent.act(context)
    return action
def train_agent(agent, exp):
    batch_all_state, batch_action, batch_reward, batch_isOver = exp.sample_batch(BATCH_SIZE)
    batch_state = batch_all_state[:, :CONTEXT_LEN, :, :]
    batch_next_state = batch_all_state[:, 1:, :, :]
    agent.train(batch_state, batch_action, batch_reward,
                batch_next_state, batch_isOver)

def eval_agent(agent, env, n_episodes=32):
    episode_reward = []
    for _ in range(n_episodes):
        step = 0
        total_reward = 0
        state = env.reset()
        while True:
            step += 1
            action = agent.act(state)
            state, reward, isOver, info = env.step(action)
            total_reward += reward
            if isOver:
                break
        episode_reward.append(total_reward)
    eval_reward = np.mean(episode_reward)
    return eval_reward


def train_episode(agent, env, exp, warmup=False):
    global g_epsilon
    global g_train_batches
    step = 0
    total_reward = 0
    state = env.reset()
    while True:
        step += 1
        # epsilon greedy action
        prob = np.random.random()
        # ======= 将 epsilon 贪婪 代码补充到这里
        if prob < g_epsilon:
            action = action_random(env)
        else:
            action = action_policy(agent, state, exp)
        # ======= 补充代码结束
        next_state, reward, isOver, _ = env.step(action)
        exp.append(Experience(state, action, reward, isOver))
        g_epsilon = max(0.1, g_epsilon - 1e-6)

        # train model
        if not warmup and len(exp) > MEMORY_WARMUP_SIZE:
            # ======= 将 训练智能体 代码补充到这里
            if step % UPDATE_FREQ == 0:
                agent.sync_target_network()
                if g_train_batches % SYNC_TARGET_FREQ == 0:
                    train_agent(agent, exp)
                g_train_batches += 1
            # ======= 补充代码结束

        total_reward += reward
        state = next_state
        if isOver:
            break
    return total_reward, step
env_name = 'PongNoFrameskip-v0'
env = gym.make(env_name)
# env = FireResetEnv(env)
env = AtariPreprocessing(env)
action_dim = env.action_space.n
test_env = gym.make(env_name)
# test_env = FireResetEnv(test_env)
test_env = AtariPreprocessing(test_env)
test_env = FrameStack(test_env, CONTEXT_LEN)
g_epsilon = 1.1
g_train_batches = 0

exp = ReplayMemory(int(MEMORY_SIZE), IMAGE_SIZE, CONTEXT_LEN)
agent = DQNModel(IMAGE_SIZE, action_dim, GAMMA, CONTEXT_LEN, USE_CUDA)