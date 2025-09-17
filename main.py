import numpy as np

class Zero:
    def get(self):
        return 0

class Desk:
    def __init__(self):
        self.desk = np.full((10, 10), Zero)

    def get(self):
        return self.desk

    def print_desk(self):
        for_print = [[]]
        for i in range(self.get.shape[0]):
            for j in range(self.get.shape[1]):
                for_print[i].append(self.get[i, j])
            for_print.append([])
            print(for_print[i])
desk = Desk
desk.print_desk(desk)

class Plant:
    def __init__(self):
        list_of_components = []

class Component:
    pass


class Green(Component):
    def __init__(self):
        pass

class Red(Component):
    def __init__(self):
        pass

class Yellow(Component):
    def __init__(self):
        pass

class White(Component):
    def __init__(self):
        pass

class Blue(Component):
    def __init__(self):
        pass

