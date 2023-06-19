""" Smart City Road - среда для мультиагентного обучения с подкреплением, созданная с помощью библиотеки turtle.
Среда представляет собой двухполосную круговую дорогу, по которой движется 2 агента-дефектор и 4 агента-кооператора
против часовой стрелки. Кооператоры движутся со случайной скоростью 2, 4 или 5, а скорость дефекторов выбирается
подключенной нейронной сетью. Награды агента будут соответсвовать самой скорости.

Эксперимент разбит на 50 эпизодов. Счетчиком окончания эпизода является переменная number_of_frames,
равная 100 шагам агента. После совершения 100 шагов эпизод завершается функцией reset, и начинается новый.

Каждый агент задан 2 векторами - вектором координат и вектором скорости (point и course), которые изменяются
с течением эпизода. Нейронная сеть получает из среды состояние агента state, состоящее из его координат, скорости
и угла обзор.

Помимо награды за каждый шаг, которую получает нейронная сеть, отдельно рассчитывается награда за эпизод, на основе
которых строится график. Возможные действия агента, управляемые алгоритмом, заданы с помощью параметров action.
Number_of_actions равен 4 и соответствует установлению скорости в 2, 4, 5, а также возможности перестроения."""


import turtle as t
from turtle import *
from freegames import vector, floor
from random import choice


class SmartCityRoad:

    def __init__(self):
        self.done = False
        self.path = Turtle(visible=False)
        self.writer = Turtle(visible=False)
        self.number_of_frames = 200
        self.reward = 0
        self.reward1 = 0
        self.reward2 = 0
        self.rewards = []
        self.info = 0
        self.crash_count = 0
        self.turn_count1 = 0
        self.number_of_turns1 = []
        self.turn_count2 = 0
        self.number_of_turns2 = []
        self.episodes = 1
        self.observation1 = [0, 0, 0, 0, 0]
        self.obs_coord1 = []
        self.observation2 = [0, 0, 0, 0, 0]
        self.obs_coord2 = []

        """Дефекторы, способные перестраиваться и изменять скорость по ходу обучения"""
        self.agents1 = [
            [vector(-140, -160), vector(4, 0)],
            [vector(-60, -160), vector(5, 0)],
        ]

        """Кооператоры внешней полосы. Не могут перестраиваться"""
        self.agents2 = [
            [vector(120, -180), vector(4, 0)],
            [vector(-100, -180), vector(5, 0)],
            [vector(-180, 150), vector(0, -5)],
            [vector(-100, 160), vector(-4, 0)],
        ]

        """Кооператоры внутренней полосы. Не могут перестраиваться"""
        self.agents3 = [
            [vector(140, 50), vector(0, 5)],
            [vector(20, -160), vector(4, 0)],
            [vector(-160, 0), vector(0, -5)],
            [vector(100, 140), vector(-4, 0)],
        ]

        """Среда задается матрицей, состоящей из нулей и единиц, где единицы 
        обозначают дорогу, по которой может двигаться агент"""
        self.tiles = [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
            0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
            0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0,
            0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0,
            0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0,
            0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0,
            0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0,
            0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0,
            0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0,
            0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0,
            0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0,
            0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0,
            0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0,
            0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0,
            0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0,
            0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0,
            0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
            0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]

        """Визуализация среды"""
        self.win = t.Screen()
        self.win.title('Social Dilemma')
        self.win.bgcolor('black')
        self.win.setup(420, 420, 370, 0)
        self.win.tracer(False)
        self.win.listen()
        self.world()

        """Вывод на экран номера эпизода, скорости агента и награды за эпизод"""
        self.score = t.Turtle()
        self.score.speed(10)
        self.score.color('white')
        self.score.penup()
        self.score.hideturtle()
        self.score.goto(-5, 110)
        self.score.write("Episode: {}  Speed: {}   Reward: {}".format(self.episodes, self.reward, self.info),
                         align='center', font=('Arial', 10, 'bold'))

        self.writer = t.Turtle()
        self.writer.speed(10)
        self.writer.color('yellow')
        self.writer.penup()
        self.writer.hideturtle()
        self.writer.goto(-10, 80)
        self.writer.write("Crashes: {}".format(self.crash_count), align='center', font=('Arial', 11, 'bold'))

    def square(self, x, y):
        """Вывод на экран полотна с изображенной на нем дорогой"""
        self.path.up()
        self.path.goto(x, y)
        self.path.down()
        self.path.begin_fill()

        for count in range(4):
            self.path.forward(20)
            self.path.left(90)

        self.path.end_fill()

    def world(self):
        bgcolor('black')
        self.path.color('gray')

        for index in range(len(self.tiles)):
            tile = self.tiles[index]

            if tile > 0:
                x = (index % 20) * 20 - 200
                y = 180 - (index // 20) * 20
                self.square(x, y)
                self.path.up()
                self.path.goto(x + 10, y + 10)
                self.path.dot(2, 'white')

    def offset(self, point):
        """Функция offset необходима для расчета валидности следующего шага агента."""
        x = (floor(point.x, 20) + 200) / 20
        y = (180 - floor(point.y, 20)) / 20
        index = int(x + y * 20)
        return index

    def valid(self, point):
        """Функция возвращает True, если следующий шаг агента возможен и он не упирается в стену."""
        index = self.offset(point)
        if self.tiles[index] == 0:
            return False
        index = self.offset(point + 19)
        if self.tiles[index] == 0:
            return False
        return point.x % 20 == 0 or point.y % 20 == 0

    def speed(self, course, n):
        """Функции изменения скорости агента в значения 2, 4 или 5."""
        g = abs(course)
        v = (course / g) * n
        course.x = v.x
        course.y = v.y

    def speed2(self, point, course):
        self.speed(course, 2)
        point.move(course)

    def speed4(self, point, course):
        self.speed(course, 4)
        point.move(course)

    def speed5(self, point, course):
        self.speed(course, 5)
        point.move(course)

    def edge_check1(self, point, course):
        """Функция поворота для дефектора"""
        self.edge_check2(point, course)
        self.edge_check3(point, course)
        if not self.valid(point+course):
            if point.x + course.x < -180 and point.y != -180:
                point.x = -180
                course.rotate(90)
            if point.x + course.x > 160 and point.y != 160:
                point.x = 160
                course.rotate(90)
            if point.y + course.y < -180 and point.x != -180:
                point.y = -180
                course.rotate(90)
            if point.y + course.y > 160 and point.x != 160:
                point.y = 160
                course.rotate(90)
            else:
                if 0 > point.x + course.x > -160 and -160 < point.y < 140 and abs(course.y) == 0:
                    point.x = -160
                    course.rotate(270)
                if 0 < point.x + course.x < 140 and -160 < point.y < 140 and abs(course.y) == 0:
                    point.x = 140
                    course.rotate(270)
                if 0 > point.y + course.y > -160 and -160 < point.x < 140 and abs(course.x) == 0:
                    point.y = -160
                    course.rotate(270)
                if 0 < point.y + course.y < 140 and -160 < point.x < 140 and abs(course.x) == 0:
                    point.y = 140
                    course.rotate(270)
                else:
                    point.move(course)

    def edge_check2(self, point, course):
        """Функция поворота, если внешний агент упирается в стену"""
        if point.x == 160 and point.y + course.y > 160:
            point.y = 160
            course.rotate(90)
            point.move(course)
        elif point.x == -180 and point.y + course.y < -180:
            point.y = -180
            course.rotate(90)
            point.move(course)
        if point.y == 160 and point.x + course.x < -180:
            point.x = -180
            course.rotate(90)
            point.move(course)
        elif point.y == -180 and point.x + course.x > 160:
            point.x = 160
            course.rotate(90)
            point.move(course)

    def edge_check3(self, point, course):
        """Функция поворота, если внутренний агент упирается в стену"""
        if point.x == 140 and point.y + course.y > 140 and course.y != 0:
            point.y = 140
            course.rotate(90)
            point.move(course)
        if point.x == -160 and point.y + course.y <= -160 and course.y != 0:
            point.y = -160
            course.rotate(90)
            point.move(course)
        if point.y == 140 and point.x + course.x < -160 and course.x != 0:
            point.x = -160
            course.rotate(90)
            point.move(course)
        if point.y == -160 and point.x + course.x > 140 and course.x != 0:
            point.x = 140
            course.rotate(90)
            point.move(course)

    def rotation(self, point, course):
        if (-160 < point.y < 140 and course.y != 0) or (-160 < point.x < 140 and course.x != 0):
            if point.y == -180 or point.y == 160 or point.x == -180 or point.x == 160:
                course.rotate(90)
            elif point.y == -160 or point.y == 140 or point.x == -160 or point.x == 140:
                course.rotate(270)
        point.move(course)

    def go(self, point, course):
        """Функция движения агентов без управления
        нейронной сетью - скорости выбираются случайно"""
        g = abs(course)
        v2 = (course / g) * 2
        v4 = (course / g) * 4
        v5 = (course / g) * 5
        options = [
            v2,
            v4,
            v5
        ]
        plan = choice(options)
        course.x = plan.x
        course.y = plan.y
        point.move(course)

    def observing1(self, agent1, agent2, agent3):
        """Функция отслеживания машин в зоне видимости 25 метров для первого дефектора"""
        self.obs_coord1 = []
        self.observation1 = [0, 0, 0, 0, 0]
        a = agent1[0][1] / abs(agent1[0][1])
        for i in range(1, 26):
            b = agent1[0][0] + a*i
            self.obs_coord1.append(b)
        for i in range(len(self.obs_coord1)):
            if agent1[1][0] == self.obs_coord1[i]:
                print('car is nearby')
                self.observation1[i // 5] = 1
                print(self.observation1)
                return self.observation1
            for j in range(len(agent2)):
                if agent2[j][0] == self.obs_coord1[i]:
                    print('car is nearby')
                    self.observation1[i//5] = 1
                    print(self.observation1)
                    return self.observation1
            for k in range(len(agent3)):
                if agent3[k][0] == self.obs_coord1[i]:
                    print('car is nearby')
                    self.observation1[i//5] = 1
                    print(self.observation1)
                    return self.observation1

    def observing2(self, agent1, agent2, agent3):
        """Функция отслеживания машин в зоне видимости 25 метров для второго дефектора"""
        self.obs_coord2 = []
        self.observation2 = [0, 0, 0, 0, 0]
        a = agent1[1][1] / abs(agent1[1][1])
        for i in range(1, 26):
            b = agent1[1][0] + a*i
            self.obs_coord2.append(b)
        for i in range(len(self.obs_coord2)):
            if agent1[0][0] == self.obs_coord2[i]:
                print('car is nearby')
                self.observation2[i // 5] = 1
                print(self.observation2)
                return self.observation2
            for j in range(len(agent2)):
                if agent2[j][0] == self.obs_coord2[i]:
                    print('car is nearby')
                    self.observation2[i//5] = 1
                    print(self.observation2)
                    return self.observation2
            for k in range(len(agent3)):
                if agent3[k][0] == self.obs_coord2[i]:
                    print('car is nearby')
                    self.observation2[i//5] = 1
                    print(self.observation2)
                    return self.observation2

    def slowdown(self, agent):
        """Счетчик аварий"""
        self.writer.clear()
        self.crash_count += 1
        self.writer.write("Crashes: {}".format(self.crash_count), align='center', font=('Arial', 11, 'bold'))
        self.number_of_frames -= 1
        if len(self.rewards) != 0:
            self.info = format(round((sum(self.rewards)/len(self.rewards)), 2), '.2f')
        self.score.clear()
        self.score.write("Episode: {}  Speed: {}   Reward: {}".format(self.episodes, self.reward, self.info),
                         align='center', font=('Arial', 10, 'bold'))

    def crash(self, agent1, agent2):
        self.a, self.b = 0, 0
        """Функция столкновения. Машина, которая двигалась быстрее (виновник аварии),
         замедляется, а другая машина продолжает ехать"""
        for i in range(0, int(len(self.agents1))):
            for j in range(0, int(len(self.agents2))):
                if agent1[i][0] == agent2[j][0]:
                    v1 = agent1[i][1]
                    v2 = agent2[j][1]
                    if abs(v1.x + v1.y) > abs(v2.x + v2.y):
                        self.slowdown(agent1[i][1])
                        self.a = 1
                        agent2[j][0] += agent2[j][1]
                    else:
                        self.slowdown(agent2[j][1])
                        self.b = 1
                        agent1[i][0] += agent1[i][1]
                    print('CRASH!')
                    return True
                elif abs(agent1[i][1]) > abs(agent2[j][1]):
                    if abs(agent1[i][0]) < abs(agent2[j][0]) and abs(agent1[i][0] - agent2[j][0]) < abs(
                            agent1[i][1]) and \
                            (agent1[i][0][0] == agent2[j][0][0] or agent1[i][0][1] == agent2[j][0][1]):
                        self.slowdown(agent1[i][1])
                        self.a = 1
                        agent2[j][0] += agent2[j][1]
                        print('CRASH!')
                        return True
                elif abs(agent1[i][1]) < abs(agent2[j][1]):
                    if abs(agent1[i][0]) > abs(agent2[j][0]) and abs(agent2[j][0] - agent1[i][0]) < abs(
                            agent2[j][1]) and \
                            (agent1[i][0][0] == agent2[j][0][0] or agent1[i][0][1] == agent2[j][0][1]):
                        self.slowdown(agent2[j][1])
                        self.b = 1
                        agent1[i][0] += agent1[i][1]
                        print('CRASH!')
                        return True

    def move(self):
        """Основная функция движения, запускаемая каждый шаг."""
        hideturtle()
        for point, course in self.agents1:
            """Движения дефектора"""
            self.edge_check1(point, course)
            if len(self.rewards) != 0:
                self.info = format(round((sum(self.rewards) / len(self.rewards)), 2), '.2f')
            self.score.clear()
            self.score.write("Episode: {}  Speed: {}   Reward: {}".format(self.episodes, self.reward, self.info),
                             align='center', font=('Arial', 10, 'bold'))
            self.number_of_frames -= 1
            clear()
            if self.number_of_frames == 0:
                print('Episode ', self.episodes)
                self.episodes += 1
                self.reset()
                """Для изучения поведения агентов в процессе обучения введен счетчик перестроений агента за эпизод."""
                self.number_of_turns1.append(self.turn_count1)
                self.number_of_turns2.append(self.turn_count2)
                print('Turns = ', self.turn_count1, self.turn_count2, self.number_of_turns1, self.number_of_turns2)
                self.turn_count1, self.turn_count2 = 0, 0
            # up()
            goto(point.x + 10, point.y + 10)
            dot(15, 'yellow')
        # clear()
        for point, course in self.agents2:
            """Движения агента внешней полосы"""
            if self.valid(point+course):
                self.go(point, course)
            else:
                self.edge_check2(point, course)
            up()
            goto(point.x + 10, point.y + 10)
            dot(15, 'red')
        for point, course in self.agents3:
            """Движения агента внутренней полосы"""
            if self.valid(point+course):
                self.go(point, course)
            self.edge_check3(point, course)
            up()
            goto(point.x + 10, point.y + 10)
            dot(15, 'red')
        self.observing1(self.agents1, self.agents2, self.agents3)
        self.observing2(self.agents1, self.agents2, self.agents3)
        self.crash(self.agents1, self.agents2)
        self.crash(self.agents1, self.agents3)
        self.win.update()

    def reset(self):
        """Функция reset запускается при завершении эпизода, перезапускает счетчик эпизодов и сообщает
        нейронной сети состояния агента"""
        self.number_of_frames = 200
        self.average_reward = format(round((sum(self.rewards)/self.number_of_frames), 2), '.2f')
        print('Average reward: ', self.average_reward)
        self.rewards = []
        return [[self.agents1[0][0][0], self.agents1[0][0][1], self.agents1[0][1][0], self.agents1[0][1][1],
                self.observation1[0], self.observation1[1], self.observation1[2],
                self.observation1[3], self.observation1[4]], [self.agents2[0][0][0], self.agents2[0][0][1],
                self.agents2[0][1][0], self.agents2[0][1][1], self.observation2[0], self.observation2[1],
                self.observation2[2], self.observation2[3], self.observation2[4]]]

    def step(self, action):
        """Шаг обучения, в который нейронная сеть задает скорость агентов и получает состояния агентов и их награду"""
        self.done = 0

        """Действия первого агента"""
        if action[0] == 0:
            self.speed2(self.agents1[0][0], self.agents1[0][1])
            self.reward1 = abs(self.agents1[0][1])
            self.rewards.append(self.reward1)

        if action[0] == 1:
            self.speed4(self.agents1[0][0], self.agents1[0][1])
            self.reward1 = abs(self.agents1[0][1])
            self.rewards.append(self.reward1)

        if action[0] == 2:
            self.speed5(self.agents1[0][0], self.agents1[0][1])
            self.reward1 = abs(self.agents1[0][1])
            self.rewards.append(self.reward1)

        if action[0] == 3:
            self.rotation(self.agents1[0][0], self.agents1[0][1])
            self.reward1 = abs(self.agents1[0][1])/2
            self.rewards.append(self.reward1)
            self.turn_count1 += 1

        """Действия второго агента"""
        if action[1] == 0:
            self.speed2(self.agents1[1][0], self.agents1[1][1])
            self.reward2 = abs(self.agents1[1][1])
            self.rewards.append(self.reward2)

        if action[1] == 1:
            self.speed4(self.agents1[1][0], self.agents1[1][1])
            self.reward2 = abs(self.agents1[1][1])
            self.rewards.append(self.reward2)

        if action[1] == 2:
            self.speed5(self.agents1[1][0], self.agents1[1][1])
            self.reward2 = abs(self.agents1[1][1])
            self.rewards.append(self.reward2)

        if action[1] == 3:
            self.rotation(self.agents1[1][0], self.agents1[1][1])
            self.reward2 = abs(self.agents1[1][1])/2
            self.rewards.append(self.reward2)
            self.turn_count2 += 1

        """Штрафы за столкновения"""
        if self.crash(self.agents1, self.agents3) is True:
            self.reward1 = -20
            self.rewards.append(self.reward1)
        if self.crash(self.agents2, self.agents3) is True:
            self.reward2 = -20
            self.rewards.append(self.reward2)
        if self.crash(self.agents1, self.agents2) is True:
            if self.a == 1:
                self.reward1 = -20
                self.rewards.append(self.reward1)
            if self.b == 0:
                self.reward2 = -20
                self.rewards.append(self.reward2)
        self.move()
        self.reward = (self.reward1 + self.reward2) // 2
        """Состояния агентов представляют собой массив, состоящий из их координат, скорости и обзора"""
        state1 = [self.agents1[0][0][0], self.agents1[0][0][1], self.agents1[0][1][0], self.agents1[0][1][1],
                 self.observation1[0], self.observation1[1], self.observation1[2], self.observation1[3],
                 self.observation1[4]]
        state2 = [self.agents1[1][0][0], self.agents1[1][0][1], self.agents1[1][1][0], self.agents1[1][1][1],
                 self.observation2[0], self.observation2[1], self.observation2[2], self.observation2[3],
                 self.observation2[4]]
        return [self.reward1, self.reward2], [state1, state2], self.done
