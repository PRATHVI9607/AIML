import math


def hill_climb(f, x, step, iters):
    for i in range(iters):
        best = max((x - step, x + step), key=f)
        if f(best) <= f(x):
            return x, f(x), i
        x = best
    return x, f(x), iters


def make_function(expr):
    code = compile(expr.replace('^', '**'), '<f(x)>', 'eval')
    names = {k: v for k, v in vars(math).items() if not k.startswith('_')}
    return lambda x: eval(code, {'__builtins__': {}}, {**names, 'x': x})


print('Hill climbing: finds a maximum of f(x).')
expr = input('f(x) = ')
start = float(input('Start x: '))
step = float(input('Step size: '))
iters = 1000

try:
    x, fx, steps = hill_climb(make_function(expr), start, step, iters)
except Exception as e:
    print(f'Could not evaluate f(x) = {expr}: {e}')
else:
    print(f'Max found at x = {x:.4f}, f(x) = {fx:.4f} after {steps} steps')
    if steps == iters:
        print(f'Warning: stopped after {iters} moves. f may keep rising, so this is not a peak.')
