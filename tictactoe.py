WINS = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]

def winner(b):
    for x, y, z in WINS:
        if b[x] != ' ' and b[x] == b[y] == b[z]:
            return b[x]

def minimax(b, depth, computer):
    w = winner(b)
    if w:
        return (10 - depth if w == 'X' else depth - 10), None
    if ' ' not in b:
        return 0, None
    best = (-100 if computer else 100), None
    for i in range(9):
        if b[i] == ' ':
            b[i] = 'X' if computer else 'O'
            score = minimax(b, depth + 1, not computer)[0]
            b[i] = ' '
            if score > best[0] if computer else score < best[0]:
                best = score, i
    return best

def show(b):
    for r in (0, 3, 6):
        print(' | '.join(b[i] if b[i] != ' ' else str(i) for i in range(r, r + 3)))
    print()

def game_over(b):
    msg = {'X': 'Computer wins!', 'O': 'You win!'}.get(winner(b)) or (' ' not in b and 'Draw!')
    if msg:
        print(msg)
    return bool(msg)

b = [' '] * 9
print('Tic Tac Toe - you are O, computer is X\n')
show(b)
while True:
    move = input('Your move (0-8): ').strip()
    if move not in [str(i) for i in range(9) if b[i] == ' ']:
        print('Invalid move, try again.')
        continue
    b[int(move)] = 'O'
    show(b)
    if game_over(b):
        break
    cell = minimax(b, 0, True)[1]
    b[cell] = 'X'
    print(f'Computer plays {cell}')
    show(b)
    if game_over(b):
        break