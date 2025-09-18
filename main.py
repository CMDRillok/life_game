import numpy as np

X, Y = (20, 20)

class Zero:
    def get(self):
        return 0

class Desk:
    def __init__(self):
        self.desk = np.full((X, Y), Zero)

    def get(self):
        return self.desk

    def print_desk(self):
        desk_array = self.get()
        for i in range(desk_array.shape[0]):
            row = []
            for j in range(desk_array.shape[1]):
                try:
                    row.append(desk_array[i, j].get())
                except TypeError:
                    row.append(0)
            print(row)

    def create(self, x, y, obj):
        self.desk[x, y] = obj

    def make_random_desk(self, count=10):
        """Наполняет поле семенами с равномерным распределением"""
        # Рассчитываем шаг для равномерного распределения
        step_i = max(1, Y // int((count) ** 0.5))
        step_j = max(1, X // int((count) ** 0.5))

        placed = 0
        for i in range(0, Y, step_i):
            for j in range(0, X, step_j):
                if placed < count:
                    self.create(i, j, Blue())
                    placed += 1
                if placed >= count:
                    return

class Plant:
    def __init__(self):
        self.list_of_components = []
        self.gen = []
        """это геном организма"""

class Component:
    """класс для заимствования повторяющихся методов и параметров."""
    def __init__(self):
        self.num = 0
        """обозначение в матрице"""
        self.coordinates = ()
        """координаты нахождения в поле. (X, Y)"""

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


desk = Desk()
desk.make_random_desk(count=100)
desk.print_desk()