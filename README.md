# smart_city_road
 
![DQN_singleEnvironment](https://github.com/egormorgunov/smart_city_road/assets/108347547/21a9017c-c9ec-4902-bc52-b91abbbf4baf)

 **Smart City Road** - игровая среды для мультиагентного обучения с подкреплением. Среда имитирует движение автомобилей в условиях плотного транспортного потока с разделением агентов на кооператоров и дефекторов.

 ## Установка
```
git clone https://github.com/egormorgunov/smart_city_road.git
cd smart_city_road
pip install -e .
```
## Версии среды

![image](https://github.com/egormorgunov/smart_city_road/assets/108347547/34b0443d-d70a-412b-9acb-ed29482bd351)

- Одноагентная версия с 8 агентами-кооператорами и 1 агентом-дефектором (см. [Одноагентную вариацию среды](single-agent-env/Env_Single_agent.py))
- Мультиагентная версия с 8 агентами-кооператорами и 2 агентами-дефекторами (см. [Мультиагентную версию среды](multi-agent-env/Env_Multi_agent.py))

## Документация среды
Полная документация среды приведена в следующем :taxi: [файле](Environment_Documentation.pdf) :taxi:

## Тестирование

Для тестирования среды **Smart City Road** используйте файлы "test.py", которые содержатся в папках с каждой из версий среды (для одноагентной среды файл находится [здесь](single-agent-env/test.py), для мультиагентной - [здесь](multi-agent-env/test.py)).

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

## Цитирование

