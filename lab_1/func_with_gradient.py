from gradient_descent.function import Function, Gradient


class FuncWithGradient:
    def __init__(self, name: str, func: Function, grad: Gradient):
        self.func = func
        self.grad = grad
        self.name = name

    def get(self) -> (Function, Gradient):
        return self.func, self.grad


    def get_func(self) -> Function:
        return self.func

    def get_grad(self) -> Gradient:
        return self.grad

    def get_name(self) -> str:
        return self.name