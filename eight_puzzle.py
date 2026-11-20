import heapq

GOAL = (1, 2, 3, 4, 5, 6, 7, 8, 0)


def manhattan(s):
    return sum(abs(i // 3 - (t - 1) // 3) + abs(i % 3 - (t - 1) % 3) for i, t in enumerate(s) if t)


def neighbours(s):
    i = s.index(0)
    r, c = divmod(i, 3)
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        if 0 <= r + dr < 3 and 0 <= c + dc < 3:
            j = (r + dr) * 3 + (c + dc)
            b = list(s)
            b[i], b[j] = b[j], b[i]
            yield tuple(b)


def solvable(s):
    tiles = [t for t in s if t]
    return sum(a > b for k, a in enumerate(tiles) for b in tiles[k + 1:]) % 2 == 0


def a_star(start):
    frontier = [(manhattan(start), 0, start)]
    parent = {start: None}
    g_cost = {start: 0}
    expanded = 0
    while frontier:
        f, g, s = heapq.heappop(frontier)
        if g > g_cost[s]:
            continue
        if s == GOAL:
            path = []
            while s:
                path.append(s)
                s = parent[s]
            return path[::-1], expanded
        expanded += 1
        for n in neighbours(s):
            if g + 1 < g_cost.get(n, float('inf')):
                g_cost[n] = g + 1
                parent[n] = s
                heapq.heappush(frontier, (g + 1 + manhattan(n), g + 1, n))


def show(s):
    for r in (0, 3, 6):
        print(' '.join(str(t) if t else '_' for t in s[r:r + 3]))
    print()


assert len(a_star((1, 2, 3, 4, 5, 6, 0, 7, 8))[0]) == 3

text = input('Start board, 9 digits row by row, 0 = blank [1 2 3 0 4 6 7 5 8]: ')
start = tuple(int(ch) for ch in text if ch.isdigit()) or (1, 2, 3, 0, 4, 6, 7, 5, 8)

if sorted(start) != list(range(9)):
    print('Invalid board: type each digit 0-8 exactly once.')
elif not solvable(start):
    print('This board cannot be solved (odd number of inversions).')
else:
    path, expanded = a_star(start)
    for step, s in enumerate(path):
        print('Start:' if step == 0 else f'Move {step}:')
        show(s)
    print(f'Solved in {len(path) - 1} moves, {expanded} states expanded.')
