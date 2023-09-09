# smart_city_road
 
![DQN_singleEnvironment](https://github.com/egormorgunov/smart_city_road/assets/108347547/21a9017c-c9ec-4902-bc52-b91abbbf4baf)

 **Smart City Road** is a game environment for multi-agent reinforcement learning. The environment simulates the movement of cars in conditions of dense traffic flow with the separation of agents into cooperators and defectors.

 ## Installation
```
git clone https://github.com/egormorgunov/smart_city_road.git
cd smart_city_road
pip install -e .
```
## Environment Versions

![image](https://github.com/egormorgunov/smart_city_road/assets/108347547/bea98ba0-2645-49ff-af45-f314459ecf4a)

- Single-agent version with 8 cooperative agents and 1 defector agent (see [Single-agent version](single-agent-env/Env_Single_agent.py))
- Single-agent version with 8 cooperative agents and 2 defector agents (see [Multi-agent version](multi-agent-env/Env_Multi_agent.py))

## Environment Documentation
Full environment documentation is given in the following :taxi: [file](Environment_Documentation.pdf) :taxi:

## Testing

To test the **Smart City Road** environment use files "test.py ", which are contained in folders with each of the environment versions (for a single-agent environment, the file is [here](single-agent-env/test.py), for a multi-agent environment - [here](multi-agent-env/test.py)).

```python
from Env_Single_agent import SmartCityRoad
import random

env = SmartCityRoad()

if __name__ == '__main__':
    episode = 5
    max_steps = 100
    for e in range(episode):
        for i in range(max_steps):
            action = random.randint(0, 4)
            reward, next_state, done = env.step(action)
```

