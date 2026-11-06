def f(x):
    return -(x - 3) ** 2 + 5

def hill_climb(f, x, step=0.1, iters=1000):
    for i in range(iters):
        best = max((x - step, x + step), key=f)
        if f(best) <= f(x):
            break
        x = best
    return x, f(x), i

x, fx, steps = hill_climb(f, x=0.0)
assert abs(x - 3) < 0.1
print(f'Max found at x = {x:.2f}, f(x) = {fx:.2f} after {steps} steps')