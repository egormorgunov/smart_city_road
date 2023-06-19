""" Тестовый код для запуска среды. Все действия агента выбираются случайно. """
from Env_Multi_agent import SmartCityRoad
import random

env = SmartCityRoad()

if __name__ == '__main__':
    episode = 5
    max_steps = 100
    for e in range(episode):
        for i in range(max_steps):
            action = [random.randint(0, 4), random.randint(0, 4)]
            reward, next_state, done = env.step(action)
