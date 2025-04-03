from typing import Callable, List

import numpy as np
from function import Function, Gradient
from scipy.constants import golden_ratio



def gradient_golden_ratio(f: Function, start: List[int], grads: List[Gradient], eps=1e-6):
    iteration = 0
    while True:
        iteration += 1
        print()
        lr = golden_ratio_optimization(f, np.float32(eps), np.float32(5.0), grads[0](np.array(start))[0], grads[0](np.array(start))[1])
        vals_k = []
        for i in range(len(start)):
            vals_k.append(start[i] - lr * grads[0](start)[i])
        if np.abs(f(start) - f(np.array(vals_k))) < eps:
            break
        start = vals_k
    return {'point' : vals_k, 'function value': f(np.array(vals_k))}




def golden_ratio_optimization(f: Function, l: np.float32, r: np.float32, grad_x, grad_y, eps: np.float32 = 1e-5):
    while np.abs(l - r) > eps:
        x_1 = r - (r - l) / golden_ratio
        x_2 = l + (r - l) / golden_ratio
        y_1 = f(np.array([l - x_1 * grad_x, r - x_1 * grad_y]))
        y_2 = f(np.array([l - x_2 * grad_x, r - x_2 * grad_y]))
        if y_1 >= y_2:
            l = x_1
        else:
            r = x_2
    return (l + r) / 2
