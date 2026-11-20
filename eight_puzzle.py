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
    g_cost = {start: 0}
    while frontier:
        f, g, s = heapq.heappop(frontier)
        if g > g_cost[s]:
            continue
        if s == GOAL:
            return g
        for n in neighbours(s):
            if g + 1 < g_cost.get(n, float('inf')):
                g_cost[n] = g + 1
                heapq.heappush(frontier, (g + 1 + manhattan(n), g + 1, n))


text = input('Start board, 9 digits row by row, 0 = blank: ')
start = tuple(int(ch) for ch in text if ch.isdigit())

if sorted(start) != list(range(9)):
    print('Invalid board: type each digit 0-8 exactly once.')
elif not solvable(start):
    print('This board cannot be solved (odd number of inversions).')
else:
    print(f'Solved in {a_star(start)} moves.')
