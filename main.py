import numpy as np

class Zero:
    def get():
        return 0

class Desk:
    def __init__(self):
        self.desk = np.full((10, 10), Zero)

    def get(self):
        return self.desk

    def print_desk(self):
        desk_array = self.get()
        for i in range(desk_array.shape[0]):
            row = []
            for j in range(desk_array.shape[1]):
                row.append(desk_array[i, j].get())
            print(row)


desk = Desk()
desk.print_desk()

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

