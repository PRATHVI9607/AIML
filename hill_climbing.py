import math


def hill_climb(f, x, step=0.1, iters=1000):
    """Step to the higher neighbour (x - step or x + step) until neither is higher."""
    for i in range(iters):
        best = max((x - step, x + step), key=f)
        if f(best) <= f(x):
            return x, f(x), i  # peak reached after i moves
        x = best
    return x, f(x), iters  # ran out of moves, so this may not be a peak


def make_function(expr):
    """Turn text like 'sin(x) + x/5' into a Python function of x."""
    code = compile(expr.replace('^', '**'), '<f(x)>', 'eval')  # allow x^2 as well as x**2
    names = {k: v for k, v in vars(math).items() if not k.startswith('_')}  # sin, cos, exp, log, sqrt, pi, e, ...
    # no builtins: only maths names and x are visible. Not a sandbox, so only type expressions you trust.
    return lambda x: eval(code, {'__builtins__': {}}, {**names, 'x': x})


assert abs(hill_climb(make_function('-(x - 3)^2 + 5'), 0.0)[0] - 3) < 0.1  # self-check: known peak at x = 3

print('Hill climbing: finds a maximum of f(x). Press Enter to use the default shown in [ ].')
expr = input('f(x) = [-(x - 3)^2 + 5] ').strip() or '-(x - 3)^2 + 5'
start = float(input('Start x [0]: ') or 0)
step = float(input('Step size [0.1]: ') or 0.1)
iters = 1000

try:
    x, fx, steps = hill_climb(make_function(expr), start, step, iters)
except Exception as e:  # bad expression, unknown name, or f undefined at some x (e.g. log of a negative)
    print(f'Could not evaluate f(x) = {expr}: {e}')
else:
    print(f'Max found at x = {x:.4f}, f(x) = {fx:.4f} after {steps} steps')
    if steps == iters:
        print(f'Warning: stopped after {iters} moves. f may keep rising, so this is not a peak.')
