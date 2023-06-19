from Env_Multi_agent import SmartCityRoad

import random
import numpy as np
from keras import Sequential
from collections import deque
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.optimizers import Adam

env = SmartCityRoad()
np.random.seed(0)


class IQN:

    """ Алгоритм независимого Q-обучения (Independent Q-Network, IQN) представляет собой модификацию алгоритма
    глубокого Q-обучения, которая позволяет каждому агенту обучаться независимо от других агентов. Основная идея IQN
    заключается в том, что каждый агент обучается своей собственной Q-функции, используя опыт, полученный только им. """

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
        reward1, reward2 = [], []
        state1, state2 = [], []
        next_state1, next_state2 = [], []
        action1, action2 = [], []
        for i in range(64):
            reward1.append(rewards[i][0])
            reward2.append(rewards[i][1])
            state1.append(states[i][0].tolist())
            state2.append(states[i][1].tolist())
            next_state1.append(next_states[i][0].tolist())
            next_state2.append(next_states[i][1].tolist())
            action1.append(actions[i][0])
            action2.append(actions[i][1])
        rewards1 = np.array(reward1)
        rewards2 = np.array(reward2)
        states1 = np.array(state1)
        states2 = np.array(state2)
        next_states1 = np.array(next_state1)
        next_states2 = np.array(next_state2)
        actions1 = np.array(action1)
        actions2 = np.array(action2)
        targets1 = rewards1 + self.gamma*(np.amax(self.model.predict_on_batch(next_states1), axis=1))*(1-dones)
        targets2 = rewards2 + self.gamma * (np.amax(self.model.predict_on_batch(next_states2), axis=1)) * (1 - dones)
        targets_full1 = self.model.predict_on_batch(states1)
        targets_full2 = self.model.predict_on_batch(states2)
        ind = np.array([i for i in range(self.batch_size)])
        targets_full1[[ind], [actions1]] = targets1
        targets_full2[[ind], [actions2]] = targets2

        self.model.fit(states1, targets_full1, epochs=1, verbose=0)
        self.model.fit(states2, targets_full2, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def train_dqn(episode):

    loss1, loss2 = [], []

    action_space = 4
    state_space = 9
    max_steps = 200

    agent = IQN(action_space, state_space)
    for e in range(episode):
        state = env.reset()
        state[0] = np.reshape(state[0], (1, state_space))
        state[1] = np.reshape(state[1], (1, state_space))
        score1, score2 = 0, 0
        for i in range(max_steps):
            action = [agent.act(state[0]), agent.act(state[1])]
            reward, next_state, done = env.step(action)
            score1 += reward[0]
            score2 += reward[1]
            next_state[0] = np.reshape(next_state[0], (1, state_space))
            next_state[1] = np.reshape(next_state[1], (1, state_space))
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.replay()
            if done:
                print("episode: {}/{}, score: {}".format(e, episode, score1))
                break
        loss1.append(score1/200)
        loss2.append(score2/200)
        print('Эпизод ', e)
        print('Награда первого агента = ', loss1)
        print('Награда второго агента = ', loss2)
    loss = [loss1, loss2]
    return loss


if __name__ == '__main__':

    ep = 100
    loss = train_dqn(ep)
    for i in range(len(loss[0])):
        loss[0][i] /= 200
        loss[1][i] /= 200
    plt.plot([i for i in range(ep)], loss[0])
    plt.plot([i for i in range(ep)], loss[1])
    plt.title('Результаты обучения алгоритмом IQN')
    plt.xlabel('Число эпизодов')
    plt.ylabel('Средняя награда')
    plt.show()
    plt.plot([i for i in range(len(env.number_of_turns1))], env.number_of_turns1)
    plt.plot([i for i in range(len(env.number_of_turns2))], env.number_of_turns2)
    plt.title('Количество перестроений за эпизод')
    plt.xlabel('Число эпизодов')
    plt.ylabel('Число перестроений')
    plt.show()
