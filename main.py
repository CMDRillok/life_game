import numpy as np

desk = np.zeros((10, 10), dtype=int)

class Plant:
    def __init__(self):
        list_of_components = []

class Component:
    def __init__(self):
        self.num = 0

    def get(self):
        return self.num


class Green(Component):
    def __init__(self):
        super().__init__()
        self.num = 1

class Red(Component):
    def __init__(self):
        super().__init__()
        self.num = 2

class Yellow(Component):
    def __init__(self):
        super().__init__()
        self.num = 3

class White(Component):
    def __init__(self):
        super().__init__()
        self.num = 4

class Blue(Component):
    def __init__(self):
        super().__init__()
        self.num = 5
