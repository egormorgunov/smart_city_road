from Env_Single_agent import SmartCityRoad

import random
import numpy as np
from keras import Sequential
from collections import deque
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.optimizers import Adam

env = SmartCityRoad()
np.random.seed(0)


class DQN:

    """ Deep Q-Network (DQN) - алгоритм глубокого Q-обучения, являющийся одним из наиболее распространенных
    методов обучения с подкреплением. Он основан на оценке Q-функции, которая описывает, какая награда ожидается
    при выполнении определенного действия в определенном состоянии. Алгоритм Q-обучения позволяет агенту находить
    оптимальную стратегию действий в среде, максимизирующую получаемую награду."""

    def __init__(self, action_space, state_space):

        self.action_space = action_space
        self.state_space = state_space
        self.epsilon = 1
        self.gamma = .95
        self.batch_size = 64
        self.epsilon_min = .01
        self.epsilon_decay = .5
        self.learning_rate = 0.01
        self.memory = deque(maxlen=100000)
        self.model = self.build_model()


    def build_model(self):
        """ Трехслойная нейронная сеть. Входной и скрытый слой - по 64 нейрона,
        в выходном слое число нейронов соответствует числу действий. """
        model = Sequential()
        model.add(Dense(64, input_shape=(self.state_space,), activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_space, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self):

        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)
        targets = rewards + self.gamma*(np.amax(self.model.predict_on_batch(next_states), axis=1))*(1-dones)
        targets_full = self.model.predict_on_batch(states)

        ind = np.array([i for i in range(self.batch_size)])
        targets_full[[ind], [actions]] = targets

        self.model.fit(states, targets_full, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def train_dqn(episode):

    loss = []

    action_space = 4
    state_space = 9
    max_steps = 100

    agent = DQN(action_space, state_space)
    for e in range(episode):
        state = env.reset()
        state = np.reshape(state, (1, state_space))
        score = 0
        for i in range(max_steps):
            action = agent.act(state)
            reward, next_state, done = env.step(action)
            score += reward
            next_state = np.reshape(next_state, (1, state_space))
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.replay()
            if done:
                print("Эпизод: {}/{}, Награда: {}".format(e, episode, score))
                break
        loss.append(score)
    return loss


if __name__ == '__main__':

    ep = 50
    loss = train_dqn(ep)
    # env.model.save('DQN_results1')
    for i in range(len(loss)):
        loss[i] /= 100
    plt.plot([i for i in range(ep)], loss)
    plt.title('Результаты обучения алгоритмом DQN')
    plt.xlabel('Число эпизодов')
    plt.ylabel('Средняя награда')
    plt.show()
