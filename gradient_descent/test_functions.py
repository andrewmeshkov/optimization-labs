from function import Function, Gradient
import numpy as np
from golden_ratio import gradient_golden_ratio

# R^2 -> R
class f2(Function):
    def __call__(self, *args, **kwargs):
        coefs = args[0]
        x = coefs[0]
        y = coefs[1]
        return x * x + y * y + 4 * y + x + 2


class f2_grad(Gradient):
    def __call__(self, *args, **kwargs):
        coefs = args[0]
        x = coefs[0]
        y = coefs[1]
        return 2 * x + 1, 2 * y + 4



if __name__ == "__main__":
    print(f2_grad()(np.array([2, 2])))
    opt = gradient_golden_ratio(f2(), [0, 0], [f2_grad()])
    print(opt)