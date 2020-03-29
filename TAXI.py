import gym

# 生成仿真环境
env = gym.make('Taxi-v3')
# 重置仿真环境
obs = env.reset()
# 渲染环境当前状态
#env.render()
m = env.observation_space.n  # size of the state space
n = env.action_space.n  # size of action space
print(m,n)
print("出租车问题状态数量为{:d}，动作数量为{:d}。".format(m, n))
import numpy as np

# Intialize the Q-table and hyperparameters
# Q表，大小为 m*n
Q = np.zeros([m,n])
Q2=np.zeros([m,n])
# 回报的折扣率
gamma = 0.97
# 分幕式训练中最大幕数
max_episode = 1000
# 每一幕最长步数
max_steps = 1000
# 学习率参数
alpha = 0.7
# 随机探索概率
epsilon = 0

for i in range(max_episode):
    # Start with new environment
    s = env.reset()
    done = False
    counter = 0
    for _ in range(max_steps):
        # Choose an action using epsilon greedy policy
        p = np.random.rand()
        if p>epsilon or np.any(Q[s,:])==False:
            a = env.action_space.sample()
        else:
            a = np.argmax(Q[s, :])


        # 请根据 epsilon-贪婪算法 选择动作 a
        # p > epsilon 或尚未学习到某个状态的价值时，随机探索
        # 其它情况，利用已经觉得的价值函数进行贪婪选择 (np.argmax)
        # ======= 将代码补充到这里

        # ======= 补充代码结束

        #env.step(a) //根据所选动作action执行一步
        # 返回新的状态、回报、以及是否完成
        s_new, r, done, _ = env.step(a)
        #r是执行a后得到的奖励，s是执行之前的状态
        # 请根据贝尔曼方程，更新Q表 (np.max)
        # ======= 将代码补充到这里
        Q[s,a] =(1-alpha)*Q[s,a]+alpha*(r+gamma*np.max(Q[s_new,:]))
        # ======= 补充代码结束
        # print(Q[s,a],r)
        s = s_new
        if done:
            break
print(Q)


s = env.reset()
done = False
env.render()
#Test the learned Agent

for i in range(max_steps):
    a = np.argmax(Q[s,:])
    s, _, done, _ = env.step(a)
    #env.render()
    if done:
        break


rewards = []# ======= 将代码补充到这里
rewards2=[]
for _ in range(100):
    s = env.reset()
    done = False
    #env.render()
    rprestep = []
    #Test the learned Agent
    for i in range(max_steps):
        a = np.argmax(Q[s, :])
        s, reward0, done, _ = env.step(a)
        rprestep.append(reward0)
        env.render()
        if done:
            break
    print('----------- ')
    rewards.append(np.sum(rprestep))
r_mean = np.mean(rewards)
r_var = np.var(rewards)
print(rewards)
# # ======= 补充代码结束
print("平均回报为{}，回报的方差为{}。".format(r_mean, r_var))
env.close()

