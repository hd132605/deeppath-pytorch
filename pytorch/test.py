import torch
from torch import nn


variable = None


class Variable(object):
    def __init__(self):
        print("Variable instance created.")


class temp(object):
    def __init__(self):
        global variable
        self.var = variable
        if self.var:
            print("found Variable")
        else:
            variable = Variable()
            self.var = variable


def func_a():
    a = temp()


def func_b():
    a = temp()


def main():
    func_a()
    func_b()
    pass


if __name__ == "__main__":
    main()