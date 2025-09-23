import numpy as np
import time

class Cell:
    def __init__(self, cell):
        self.cell = cell
        self.organic = 0.
        self.energy = 0.


class Desk:
    def __init__(self, X, Y):
        self.x = X
        self.y = Y
        self.desk = np.full((self.x, self.y), Cell(Zero))
        self.plants = []

    def print_desk(self, mode = 0):
        """печатает всю доску.
        args: mode = ['bio'/'organic'/'energy'][0/1/2] """
        for i in range(self.x):
            row = []
            for j in range(self.y):
                if mode == 0:
                    row.append(self.desk[i, j].cell.num)
                elif mode == 1:
                    row.append(self.desk[i, j].organic)
                else:
                    row.append(self.desk[i, j].energy)
            print(row)


    def create(self, x, y, obj):
        """создаёт новый объект на поле с координатами X и Y."""
        self.desk[x, y].cell = obj
        self.plants.append(Plant(obj))
        log("растение добавлено")

    def make_random_desk(self, count=10):
        """Наполняет поле семенами с равномерным распределением"""
        # Рассчитываем шаг для равномерного распределения
        step_i = max(1, self.y // int(count ** 0.5))
        step_j = max(1, self.x // int(count ** 0.5))
        placed = 0
        for i in range(0, self.y, step_i):
            for j in range(0, self.x, step_j):
                if placed < count:
                    obj = Blue()
                    self.create(i, j, obj)
                    placed += 1
                if placed >= count:
                    log("поле сгенерировано")
                    return

class Plant:
    def __init__(self, seed):
        self.list_of_components = [seed,]
        self.gen = []
        """это геном организма"""

class Component:
    """класс для заимствования повторяющихся методов и параметров."""
    def __init__(self):
        self.num = 0
        """обозначение в матрице"""
        self.coordinates = ()
        """координаты нахождения в поле. (X, Y)"""
        self.energy = 100

class Zero:
    def __init__(self):
        self.num = 0

    def get(self):
        return self.num


class Green(Component):
    """Собирает энергию из воздуха, то есть поглощает свет."""
    def __init__(self):
        super().__init__()
        self.num = 1

class Red(Component):
    """Собирает энергию с ближайших клеток. Поле в котором действует 3x3"""
    def __init__(self):
        super().__init__()
        self.num = 2

class Yellow(Component):
    """Собирает органику из почвы и преобразовывает в энергию/или в органику."""
    def __init__(self):
        super().__init__()
        self.num = 3

class White(Component):
    """стебель. Служит для перемещения энергии от одной клетки к другой"""
    def __init__(self):
        super().__init__()
        self.num = 4

class Blue(Component):
    """Семя. Хранит генетический код и служит центром принятия решений. То есть, там лежит машинное обучение"""
    def __init__(self):
        super().__init__()
        self.num = 5

def log(txt):
    print(f"[{time.localtime()[3], time.localtime()[4], time.localtime()[5]}] {txt}")


desk = Desk(20, 20)
desk.make_random_desk(count=40)
desk.print_desk(0)