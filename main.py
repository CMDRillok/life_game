import numpy as np

desk = np.zeros((10, 10), dtype=int)

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
