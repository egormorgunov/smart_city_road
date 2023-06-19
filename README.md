# smart_city_road
 
 ![DQN_singleEnvironment](https://github.com/egormorgunov/smart_city_road/assets/108347547/a7ebe5f8-f095-4e47-8f6d-ff070e445d99)

 Smart City Road - игровая среды для мультиагентного обучения с подкреплением. Среда имитирует движение автомобилей в условиях плотного транспортного потока с разделением агентов на кооператоров и дефекторов.

 ## Установка
```
git clone https://github.com/egormorgunov/smart_city_road.git
cd smart_city_road
pip install -e .
```
 
 ## Внешний вид среды
Среда представляет собой графическое двухмерное полотно, на котором расположена двухполосная круговая дорога размером 340х340. По дороге движутся агенты – точки красного и желтого цвета, имитирующие движение автомобилей в транспортном потоке. Агенты движутся против часовой стрелки с ограниченным набором скоростей (2, 4 и 5). Агенты задаются двумя векторами: вектором местоположения (координаты агента) и вектором скорости. Агенты-кооператоры движутся по изначально заданной полосе со скоростью, которая каждый шаг случайно выбирается из заданного диапазона. Они не могут перестраиваться и двигаются «вслепую», не получая никакой информации от среды. Агенты-дефекторы обучаются подключенной нейронной сетью, которая в процессе обучения каждый шаг самостоятельно выбирает скорости для агентов. Дефекторы могут перестраиваться, а также получают информацию от среды в виде матрицы зоны видимости.

![image](https://github.com/egormorgunov/smart_city_road/assets/108347547/a4164437-afc4-4c2e-b820-af62a8f13bd7)



![image](https://github.com/egormorgunov/smart_city_road/assets/108347547/370ff485-8736-4282-bd95-fdcc6dee8d2d)

![image](https://github.com/egormorgunov/smart_city_road/assets/108347547/5f556b23-7094-46df-b128-4c4ae281f881)
