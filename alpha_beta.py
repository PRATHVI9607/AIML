import math
leaves = [3, 5, 6, 9, 1, 2, 0, -1]

def alpha_beta(node, depth, alpha, beta, maximizing):
    if depth == 0:
        return leaves[node - len(leaves)]
    best = -math.inf if maximizing else math.inf
    for child in (node * 2, node * 2 + 1):
        score = alpha_beta(child, depth - 1, alpha, beta, not maximizing)
        if maximizing:
            best, alpha = max(best, score), max(alpha, score)
        else:
            best, beta = min(best, score), min(beta, score)
        if beta <= alpha and child == node * 2:
            cut = node * 2 + 1 
            first = cut * 2 ** (depth - 1) - len(leaves)
            print(f'At node {node}: alpha={alpha} >= beta={beta}, pruned branch {node}->{cut} '
                  f'(leaves {leaves[first:first + 2 ** (depth - 1)]})')
            break
    return best

print('Best value:', alpha_beta(1, 3, -math.inf, math.inf, True))
