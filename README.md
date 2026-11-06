# AI / ML Algorithms from Scratch

Classic Artificial Intelligence and Machine Learning algorithms written from scratch in plain Python and NumPy, with no ML libraries. Every program is short enough to read in one sitting. This README explains each algorithm from first principles, then walks through every program line by line.

**Author:** PRATHVI9607

---

## Contents

| # | Program | Topic | Type |
|---|---------|-------|------|
| 1 | [tictactoe.py](tictactoe.py) | Minimax game search (play against the computer) | Adversarial search |
| 2 | [alpha_beta.py](alpha_beta.py) | Minimax with Alpha-Beta pruning on a fixed tree | Adversarial search |
| 3 | [hill_climbing.py](hill_climbing.py) | Hill climbing to maximize a single-variable function $f(x)$ | Local search / optimization |
| 4 | [knn.py](knn.py) | k-Nearest Neighbours classifier | Supervised learning |
| 5 | [kmeans.py](kmeans.py) | K-Means clustering with k-means++ initialization | Unsupervised learning |
| 6 | [logistic_regression.py](logistic_regression.py) | Logistic regression trained by gradient descent | Supervised learning |
| 7 | [naive_bayes.py](naive_bayes.py) | Gaussian Naive Bayes classifier | Supervised / probabilistic |
| 8 | [working/](working/) | Programs 4-7 run on the real **Iris** dataset | Real-data versions |

## Requirements and how to run

- Python 3.8 or newer (tested on Python 3.14)
- NumPy (`pip install numpy`). Only programs 4-8 need it. Programs 1-3 use only the standard library.

```bash
python3 tictactoe.py          # interactive game, type a cell number 0-8
python3 alpha_beta.py
python3 hill_climbing.py
python3 knn.py
python3 kmeans.py
python3 logistic_regression.py
python3 naive_bayes.py

cd working
python3 knn_iris.py
python3 kmeans_iris.py
python3 logistic_regression_iris.py
python3 naive_bayes_iris.py
```

---

# 1. Tic Tac Toe with Minimax — [tictactoe.py](tictactoe.py)

## The algorithm from scratch

### Games as trees

Tic Tac Toe is a **two-player, zero-sum, perfect-information** game:

- **Two-player:** you (O) and the computer (X) take turns.
- **Zero-sum:** a win for one player is a loss for the other. One number, the *score*, describes how good a position is for X. Whatever is good for X is equally bad for O.
- **Perfect information:** both players see the whole board. There is no hidden information and no dice.

Every such game can be drawn as a **game tree**:

- The **root** is the current board.
- Each **branch** is one legal move.
- Each **child** is the board after that move.
- The **leaves** are finished games (a win or a draw).

```
                 current board  (X to move)
              /        |         \
         X in 1     X in 2   ...  X in 8        <- X picks one
         /   \       /   \
     O in 2  O in 3 ...               ...       <- O replies
       ...
     leaves: X wins (+), O wins (-), draw (0)
```

### Scoring the leaves

A finished game gets a score from X's point of view:

| Result | Score |
|--------|-------|
| X (computer) wins | positive (`10 - depth`) |
| O (you) win | negative (`depth - 10`) |
| Draw | `0` |

### The Minimax rule

X wants the score as **high** as possible, so X is the **MAX** player. O wants it as **low** as possible, so O is the **MIN** player. Assume both players play perfectly. Then the value of any position is:

- **Leaf:** its score.
- **X to move:** the **maximum** of its children's values. X picks the best move for X.
- **O to move:** the **minimum** of its children's values. O picks the best move for O.

The definition is **recursive**: to know a position's value you need its children's values, and so on down to the leaves. When the recursion returns to the root, the computer plays the move whose child has the best value. That move is optimal: no opponent strategy can do better against it. Tic Tac Toe played perfectly by both sides is a draw, so you can at best draw against this program.

### Why subtract the depth?

`depth` counts how many moves have been made since the search started. Without it, a win in 1 move and a win in 5 moves would both score `+10`, and the computer might wander around before winning.

- **Win:** `10 - depth`. Winning sooner (smaller depth) scores higher, so the computer takes the quickest win.
- **Loss:** `depth - 10`. Losing later (bigger depth) is less negative, so if a loss can't be avoided, the computer delays it as long as possible.

### Backtracking

Instead of copying the board for every child, the program **places a symbol, recurses, then removes the symbol**. One list is reused for the whole search. This technique is called *backtracking*.

### Cost

There are at most 9! = 362,880 move orders, and fewer real games because many end early (255,168 complete games). After your first move, at most 8! = 40,320 orders remain, so a full search is instant. For bigger games like chess, full Minimax is impossible, which is why Alpha-Beta pruning (program 2) and depth limits exist.

## Line by line

The board `b` is a list of 9 strings. Each cell holds `' '` (empty), `'X'` or `'O'`. Indexes map to the grid like this:

```
0 | 1 | 2
3 | 4 | 5
6 | 7 | 8
```

**Winning lines**

- **L1** `WINS = [(0, 1, 2), ...]`: all 8 ways to win: 3 rows `(0,1,2) (3,4,5) (6,7,8)`, 3 columns `(0,3,6) (1,4,7) (2,5,8)` and 2 diagonals `(0,4,8) (2,4,6)`.

**`winner(b)`: who has won?**

- **L3** Defines the function. It takes the board.
- **L4** `for x, y, z in WINS:` loops over every winning line and unpacks the three cell indexes.
- **L5** `if b[x] != ' ' and b[x] == b[y] == b[z]:` The first cell must not be empty, otherwise three empty cells would count as a "win". Python's chained comparison `b[x] == b[y] == b[z]` means all three cells are equal.
- **L6** `return b[x]` returns the winning symbol, `'X'` or `'O'`.
- If the loop finishes without a match, the function ends and Python returns `None` automatically, meaning nobody has won.

**`minimax(b, depth, computer)`: the search**

- **L8** Takes the board, the current depth (moves since the search began) and `computer`, which is `True` when it is X's turn. It returns a pair `(score, best_move)`.
- **L9** `w = winner(b)` checks whether the game has already been won.
- **L10-11** If someone won, return the depth-adjusted score: `10 - depth` for X, `depth - 10` for O. The move is `None` because the game is over.
- **L12-13** `if ' ' not in b:` If no empty cell is left and nobody won, it is a draw: return `0, None`.
- **L14** `best = (-100 if computer else 100), None` starts with an impossible worst score. For MAX (X), any real score beats `-100`. For MIN (O), any real score beats `100`.
- **L15** `for i in range(9):` tries every cell.
- **L16** `if b[i] == ' ':` only empty cells are legal moves.
- **L17** `b[i] = 'X' if computer else 'O'` makes the move for whoever's turn it is.
- **L18** `score = minimax(b, depth + 1, not computer)[0]` recurses. The other player moves next (`not computer`), one move deeper (`depth + 1`). `[0]` keeps only the score, because only the top level needs the move.
- **L19** `b[i] = ' '` **undoes the move** (backtracking), so the board is unchanged for the next trial.
- **L20** `if score > best[0] if computer else score < best[0]:` This is a conditional expression and reads as `(score > best[0]) if computer else (score < best[0])`. MAX keeps a higher score. MIN keeps a lower one.
- **L21** `best = score, i` remembers the new best score and the cell that produced it.
- **L22** `return best` returns `(best_score, best_cell)`.

**`show(b)`: print the board**

- **L25** `for r in (0, 3, 6):` the starting index of each row.
- **L26** For the 3 cells of the row (`range(r, r + 3)`), show the symbol, or the cell's number if it is empty, so you know what to type. `' | '.join(...)` puts separators between them.
- **L27** `print()` adds a blank line after the board.

**`game_over(b)`: has the game ended?**

- **L30** Builds the end message in one expression.
  - `{'X': 'Computer wins!', 'O': 'You win!'}.get(winner(b))` looks up the winner. `.get` returns `None` if there is no winner.
  - `or (' ' not in b and 'Draw!')` runs only when the first part is `None`. If the board is full, `and` gives `'Draw!'`. Otherwise it gives `False`.
  - So `msg` is a message string, or a falsy value if the game continues.
- **L31-32** Print the message if there is one.
- **L33** `return bool(msg)` returns `True` if the game is over.

**Main game loop**

- **L35** `b = [' '] * 9` creates an empty board.
- **L36-37** Prints the title and the empty, numbered board.
- **L38** `while True:` repeats until a `break`.
- **L39** `move = input(...).strip()` reads your move and removes stray spaces.
- **L40** The list `[str(i) for i in range(9) if b[i] == ' ']` contains the numbers of the empty cells, as strings. If your input isn't one of them (a taken cell, a letter, `9`, ...)...
- **L41-42** ...print an error and `continue` back to the top of the loop to ask again.
- **L43** `b[int(move)] = 'O'` places your O.
- **L44** Shows the board.
- **L45-46** Stops if your move ended the game.
- **L47** `cell = minimax(b, 0, True)[1]` searches from the current board, at depth 0, with X to move, and takes the best move (`[1]`).
- **L48** `b[cell] = 'X'` plays the computer's move.
- **L49-50** Announces and shows it.
- **L51-52** Stops if the computer's move ended the game.

Sample session:

```
Your move (0-8): 4          <- you take the centre
Computer plays 0            <- computer takes a corner (the best reply)
X | 1 | 2
3 | O | 5
6 | 7 | 8
```

---

# 2. Alpha-Beta Pruning — [alpha_beta.py](alpha_beta.py)

## The algorithm from scratch

### The problem with plain Minimax

Minimax visits **every** node of the game tree. A tree with branching factor $b$ (moves per position) and depth $d$ has $b^d$ leaves. That number explodes quickly: chess has $b \approx 35$, so looking 10 moves ahead means about $35^{10} \approx 2.7 \times 10^{15}$ leaves.

### The key insight

Often part of the tree **cannot change the final answer**, so there is no need to look at it. Alpha-Beta pruning returns **exactly the same value as Minimax** while skipping those parts.

The algorithm carries two numbers down the tree:

- **α (alpha):** the best score the **MAX** player is already guaranteed somewhere on the path from the root. It starts at $-\infty$ and only goes up.
- **β (beta):** the best score the **MIN** player is already guaranteed somewhere on the path from the root. It starts at $+\infty$ and only goes down.

The real value of the game is always somewhere in the window $[\alpha, \beta]$.

**Pruning rule:** as soon as $\beta \le \alpha$ at a node, stop exploring its remaining children.

**Why that is safe, with an example:** suppose MAX already has a move worth 5 (α = 5). MAX now looks at a second move, where MIN replies. MIN's first reply there is worth 2, so MIN can force a value of **at most 2** in this branch (β = 2). MAX would never choose this branch, because 5 beats anything ≤ 2. Whatever the other replies are worth, they cannot matter, so they are skipped.

### How much does it save?

- **Worst case:** if the best moves happen to be checked last, nothing is pruned and the cost stays $O(b^d)$.
- **Best case:** if the best moves are checked first, the cost drops to about $O(b^{d/2})$. That means you can search **twice as deep** in the same time.

## The tree used in the program

The tree is a full binary tree of depth 3 with **heap numbering**: the root is node 1, and the children of node `n` are `2n` and `2n+1`. With this numbering, no tree structure has to be stored.

```
Level 0 (MAX)                    1
                          /             \
Level 1 (MIN)            2               3
                      /     \         /     \
Level 2 (MAX)        4       5       6       7
                    / \     / \     / \     / \
Level 3 (leaves)   8   9  10  11  12  13  14  15
values:            3   5   6   9   1   2   0  -1
```

### Step-by-step trace

1. Node 4 (MAX) = max(3, 5) = **5**. Node 2 (MIN) now knows it can force ≤ 5, so β = 5.
2. Node 5 (MAX, with α = −∞ and β = 5) sees leaf 10 = **6**, so α = 6. Now α = 6 ≥ β = 5: **prune leaf 11 (value 9)**. MIN at node 2 already has 5, so it would never allow a node worth ≥ 6.
3. Node 2 = min(5, 6) = **5**. The root (MAX) now has α = 5.
4. Node 6 (MAX, with α = 5 and β = +∞) = max(1, 2) = **2**. Node 3 (MIN) sets β = 2. Now β = 2 ≤ α = 5: **prune node 7 (leaves 0 and −1)**. MAX at the root already has 5, so a branch worth ≤ 2 is useless.
5. Root = max(5, 2) = **5**.

Three of the 8 leaves were never looked at, and the answer is still exactly what full Minimax gives.

Output:

```
At node 5: alpha=6 >= beta=5, pruned branch 5->11 (leaves [9])
At node 3: alpha=5 >= beta=2, pruned branch 3->7 (leaves [0, -1])
Best value: 5
```

## Line by line

- **L1** `import math` is needed for `math.inf` (infinity).
- **L2** `leaves = [...]` holds the values of the 8 leaves, nodes 8 to 15, from left to right.
- **L4** `alpha_beta(node, depth, alpha, beta, maximizing)` takes:
  - `node`: the heap number.
  - `depth`: how many levels remain below this node.
  - `alpha` and `beta`: the current window.
  - `maximizing`: `True` for a MAX node.
- **L5-6** `if depth == 0:` means this is a leaf. Leaves are nodes 8 to 15, so `node - len(leaves)` (`node - 8`) turns the node number into a list index from 0 to 7.
- **L7** `best` starts at the worst possible value: $-\infty$ for MAX and $+\infty$ for MIN.
- **L8** `for child in (node * 2, node * 2 + 1):` visits the left child, then the right child.
- **L9** Recurses into the child with one fewer level, the same window, and the other player.
- **L10-11** **MAX node:** `best` becomes the larger value, and `alpha` is raised, because MAX is now guaranteed at least this score.
- **L12-13** **MIN node:** `best` becomes the smaller value, and `beta` is lowered, because MIN can now force at most this score.
- **L14** `if beta <= alpha and child == node * 2:` is the pruning condition.
  - `beta <= alpha` means the rest of this node cannot matter.
  - `child == node * 2` checks that we have only seen the **left** child, so there is still a right child to skip. After the right child the loop ends anyway, so there would be nothing to prune.
- **L15** `cut = node * 2 + 1` is the right child, the subtree being skipped.
- **L16** `first = cut * 2 ** (depth - 1) - len(leaves)` finds the index of the first leaf under `cut`. Going down one level doubles the node number, and `cut` sits `depth - 1` levels above the leaves. Its leftmost leaf is therefore `cut * 2^(depth-1)`, and subtracting 8 gives the list index.
- **L17-18** Prints which node pruned which branch, and the leaf values that were skipped. A subtree of height `depth - 1` has `2 ** (depth - 1)` leaves, so that slice is printed.
- **L19** `break` skips the remaining child. **This is the actual pruning.**
- **L20** Returns the node's value.
- **L22** Starts at the root: node 1, 3 levels to the leaves, window $(-\infty, +\infty)$, MAX to move. Prints the result.

`alpha` and `beta` are plain numbers passed **by value**. Each call has its own copy, so a tightened window only affects that node's own children, never its siblings. That is exactly how the algorithm is defined.

---

# 3. Hill Climbing — [hill_climbing.py](hill_climbing.py)

## The algorithm from scratch

### Optimization as a landscape

We want the $x$ that makes $f(x)$ as large as possible. Picture the graph of $f$ as a landscape: $x$ is your position, and $f(x)$ is your height. Hill climbing is what a hiker does in thick fog: **you can't see the peak, only the ground right next to you, so you keep stepping uphill.**

### The procedure

1. Start at some point $x$.
2. Look at the **neighbours** of $x$. Here they are $x - \text{step}$ and $x + \text{step}$.
3. If the better neighbour is **higher** than where you stand, move there.
4. If neither neighbour is higher, you are at the top of a hill. **Stop.**

This is a **greedy local search**:

- *Greedy:* it always takes the best immediate improvement.
- *Local:* it only remembers the current point, not the path, so it uses almost no memory.

### The function in the program

$$f(x) = -(x - 3)^2 + 5$$

This is an upside-down parabola with a single peak at $x = 3$, where $f(3) = 5$. The search starts at $x = 0$, takes steps of 0.1, and needs $3 / 0.1 = 30$ steps to reach the top.

```
f(x)
 5 |            *  <- peak at x = 3
   |        *       *
   |     *             *
   |  *                   *
   +--+----+----+----+----+---- x
      0    1    2    3    4
      ^ start     30 steps of +0.1
```

### Limitations

- **Local maxima:** if $f$ has several hills, the search stops at the top of whichever hill it starts on. That might not be the highest one. Common fixes are *random-restart* hill climbing (start from many points and keep the best) and *simulated annealing* (sometimes accept a downhill move).
- **Plateaus:** on flat ground, no neighbour is higher, so the search stops even though it is not at a peak.
- **Step size:** the answer can only be accurate to within one step. A smaller step is more precise but needs more iterations.
- **Unbounded functions:** if $f$ keeps rising forever (for example $f(x) = x$), the search never stops by itself. It ends only when `iters` runs out.

## Line by line

- **L1-2** `f(x)` is the function to maximize. You can replace it with any single-variable function.
- **L4** `hill_climb(f, x, step=0.1, iters=1000)` takes the function, the start point, the distance to a neighbour, and a safety limit on the number of moves.
- **L5** `for i in range(iters):` runs at most `iters` iterations. `i` counts them.
- **L6** `best = max((x - step, x + step), key=f)` picks whichever of the two neighbours has the larger `f` value. `key=f` tells `max` to compare the points by `f(point)` rather than by the numbers themselves.
- **L7-8** `if f(best) <= f(x): break` If even the better neighbour is not higher than the current point, we are on a peak (or a plateau), so stop. The `<=` matters: with `<`, the search could step back and forth forever on flat ground.
- **L9** `x = best` moves uphill.
- **L10** Returns the final point, its height, and `i`. When the loop stops at a peak, `i` equals the number of moves made.
- **L12** Runs the search from $x = 0$ and unpacks the three results.
- **L13** `assert abs(x - 3) < 0.1` is a self-check that the known peak at 3 was found. **If you change `f`, change or delete this line.**
- **L14** Prints the result. `:.2f` rounds to 2 decimals, because adding 0.1 thirty times gives `3.0000000000000013` in floating-point arithmetic.

Output:

```
Max found at x = 3.00, f(x) = 5.00 after 30 steps
```

---

# 4. k-Nearest Neighbours (kNN) — [knn.py](knn.py)

## The algorithm from scratch

### Supervised classification

In **classification**, we have training examples whose correct **labels** are known, for example points labelled `A` or `B`. The goal is to predict the label of a new, unlabelled point.

### The idea: similar things are near each other

kNN is one of the simplest learning algorithms. It rests on one assumption: **points that are close together usually share a label.** To classify a new point:

1. Measure its **distance** to every training point.
2. Take the **k closest** ones, its "neighbours".
3. Let them **vote**. The most common label among them wins.

### Distance

The program uses **Euclidean distance**, the straight-line distance you would measure with a ruler. For points $p$ and $q$ with $d$ features:

$$\text{dist}(p, q) = \sqrt{(p_1 - q_1)^2 + (p_2 - q_2)^2 + \dots + (p_d - q_d)^2}$$

### Properties

- **No training step.** kNN just memorizes the data, which is why it is called a *lazy learner*. All the work happens at prediction time: about $n \times d$ operations per query, for $n$ training points with $d$ features.
- **Choosing k.**
  - With `k = 1`, the prediction follows the single nearest point, so one mislabelled or noisy point can flip it.
  - A larger `k` averages out noise but blurs the boundary between classes.
  - An odd `k` avoids tied votes when there are 2 classes.
- **Scale matters.** A feature measured in thousands dominates one measured in single digits. Features are usually rescaled to similar ranges first.

### Worked example: classifying `[4, 4]`

| Training point | Label | Distance to (4, 4) |
|---|---|---|
| (1, 2) | A | √(9+4) = 3.61 |
| (2, 3) | A | √(4+1) = **2.24** |
| (3, 1) | A | √(1+9) = **3.16** |
| (6, 5) | B | √(4+1) = **2.24** |
| (7, 7) | B | √(9+9) = 4.24 |
| (8, 6) | B | √(16+4) = 4.47 |

The 3 nearest are (2,3) A, (6,5) B and (3,1) A. The vote is 2 A to 1 B, so the prediction is **A**.

## Line by line

- **L1** `import numpy as np` for fast array maths.
- **L2** `from collections import Counter`. A `Counter` counts how often each item appears, which is exactly what the vote needs.
- **L4** `knn_predict(X_train, y_train, X_test, k=3)` takes the training points, their labels, the points to classify, and the number of neighbours.
- **L5** `preds = []` collects one prediction per test point.
- **L6** `for x in X_test:` handles one test point at a time.
- **L7** `nearest = np.argsort(np.linalg.norm(X_train - x, axis=1))[:k]`. Read it from the inside out:
  - `X_train - x`: NumPy **broadcasting** subtracts `x` from *every* row at once, giving the difference vector to each training point. Shape: `(6, 2)`.
  - `np.linalg.norm(..., axis=1)`: the length of each row, which is the Euclidean distance. Shape: `(6,)`.
  - `np.argsort(...)`: the *indexes* that would sort the distances from smallest to largest.
  - `[:k]`: keeps the first `k` indexes, the k nearest points.
- **L8**
  - `y_train[nearest]` gets the labels of those neighbours, for example `['A', 'B', 'A']`.
  - `Counter(...)` counts them: `{'A': 2, 'B': 1}`.
  - `.most_common(1)` returns the top entry as a list: `[('A', 2)]`.
  - `[0][0]` takes the label `'A'` out of that list.
  - If counts tie, `Counter` keeps the label it saw first, which is the label of the nearest neighbour.
- **L9** Returns all predictions as a NumPy array.
- **L11** Training points. Group A sits around (1-3, 1-3), and group B around (6-8, 5-7).
- **L12** Their labels, in the same order.
- **L13** New points to classify.
- **L15-16** `zip` pairs each test point with its prediction, and the loop prints them.

Output:

```
[2 2] -> A
[7 6] -> B
[4 4] -> A
```

---

# 5. K-Means Clustering — [kmeans.py](kmeans.py)

## The algorithm from scratch

### Unsupervised learning

This time **there are no labels**. We only have points, and we want to discover groups ("clusters") of points that sit close together. K-Means finds `k` clusters, where you choose `k`.

### What makes a good clustering?

Each cluster has a **centroid**, its centre point: the mean of all its points. A clustering is good when every point is close to its own centroid. K-Means minimizes the **inertia** (also called the within-cluster sum of squares):

$$\text{inertia} = \sum_{\text{each point } x} \lVert x - \text{centroid of } x\text{'s cluster} \rVert^2$$

### Lloyd's algorithm: repeat two steps

1. **Assign:** put every point in the cluster of its **nearest** centroid.
2. **Update:** move every centroid to the **mean** of the points assigned to it.

Repeat until the centroids stop moving.

**Why it always stops:**

- The assign step can only lower the inertia, because each point moves to a closer centroid.
- The update step can only lower it too, because the mean is the point with the smallest total squared distance to a group.
- The inertia therefore never goes up. There are also only finitely many ways to split the points into groups.

So the algorithm must settle. But it settles in a **local** minimum, which is not always the best possible clustering. That depends on where the centroids start.

### k-means++ initialization

Starting with random centroids can put two of them inside the same real cluster. **k-means++** spreads the starting centroids out:

1. Pick the first centroid uniformly at random from the data points.
2. For every point, compute $D(x)$, its distance to the **nearest** centroid chosen so far.
3. Pick the next centroid at random, with probability proportional to $D(x)^2$. Far-away points are much more likely to be picked.
4. Repeat until there are `k` centroids.

### Choosing k

K-Means cannot tell you the right `k`. Common methods are domain knowledge, the "elbow" in a plot of inertia against `k`, and silhouette scores. In this program the data clearly has 3 groups, so `k = 3`.

## Line by line

- **L1** `import numpy as np`.
- **L3** `kmeans(X, k, iters=100, seed=0)` takes the points, the number of clusters, the maximum number of iterations, and a random seed so that every run gives the same result.
- **L4** `rng = np.random.default_rng(seed)` creates a seeded random number generator.

**k-means++ initialization (L5-8)**

- **L5** `centroids = X[[rng.integers(len(X))]]` picks one random index and takes that point. The **double brackets** index with a list, so the result keeps 2 dimensions, shape `(1, 2)`: a list of centroids holding one centroid, not a bare point.
- **L6** `for _ in range(k - 1):` adds the remaining `k - 1` centroids.
- **L7** `d = np.linalg.norm(X[:, None] - centroids, axis=2).min(axis=1) ** 2` computes $D(x)^2$ for every point, using broadcasting:
  - `X[:, None]` has shape `(n, 1, 2)`. `centroids` has shape `(c, 2)`.
  - Subtracting them broadcasts to `(n, c, 2)`: the difference from every point to every centroid.
  - `norm(axis=2)` gives shape `(n, c)`: a table of distances, one row per point and one column per centroid.
  - `.min(axis=1)` gives shape `(n,)`: each point's distance to its **nearest** centroid.
  - `** 2` squares it.
- **L8** `rng.choice(len(X), p=d / d.sum())` picks an index with probability proportional to `d`. Dividing by the sum turns the distances into probabilities that add up to 1. Points that are already centroids have `d = 0`, so they can't be picked again. `np.vstack` appends the chosen point as a new row.

**Lloyd's loop (L9-14)**

- **L9** `for _ in range(iters):` repeats at most `iters` times.
- **L10** **Assign step.** The same distance table as L7, but `.argmin(axis=1)` returns the *index* of the nearest centroid instead of the distance. `labels[j]` is the cluster number of point `j`.
- **L11** **Update step.** For each cluster `i`:
  - `X[labels == i]` selects the points in cluster `i` (a boolean mask).
  - `.mean(axis=0)` averages them feature by feature. That average is the new centroid.
  - `if np.any(labels == i) else centroids[i]`: if a cluster ended up empty, its old centroid is kept. The mean of zero points would be `NaN`.
- **L12-13** `if np.allclose(new, centroids): break` If the centroids barely moved (within a tiny tolerance), the algorithm has converged, so stop early.
- **L14** `centroids = new` accepts the new centroids and loops again.
- **L15** Returns the final centroids and each point's cluster.

**Main program (L17-22)**

- **L17** 9 points forming 3 visible groups: bottom-left near (1, 1.5), top near (7, 9), and bottom-right near (9, 2).
- **L18** Clusters them into 3 groups.
- **L20** Prints the centroids, rounded to 2 decimals.
- **L21-22** Prints each point with its cluster number.

Output:

```
Centroids:
 [[9.   2.33]
 [1.17 1.47]
 [7.33 9.  ]]
[1. 2.] -> cluster 1
[1.5 1.8] -> cluster 1
[1.  0.6] -> cluster 1
[5. 8.] -> cluster 2
[8. 8.] -> cluster 2
[ 9. 11.] -> cluster 2
[8. 2.] -> cluster 0
[9. 3.] -> cluster 0
[10.  2.] -> cluster 0
```

The cluster *numbers* are arbitrary: "cluster 1" only means "the group whose centroid ended up in row 1". What matters is which points are grouped together.

---

# 6. Logistic Regression — [logistic_regression.py](logistic_regression.py)

## The algorithm from scratch

### From a line to a probability

We want to predict a **binary** label, class 0 or class 1, from features $x = (x_1, x_2, \dots)$.

**Step 1: a linear score.** Give each feature a **weight** $w_i$ and add a **bias** $b$:

$$z = w_1 x_1 + w_2 x_2 + \dots + b = w \cdot x + b$$

A large positive $z$ should mean "probably class 1". A large negative $z$ should mean "probably class 0".

**Step 2: squash the score into a probability.** The **sigmoid** function maps any number into the range $(0, 1)$:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

| z | σ(z) |
|---|---|
| −∞ | 0 |
| −2 | 0.12 |
| 0 | 0.5 |
| +2 | 0.88 |
| +∞ | 1 |

So $p = \sigma(w \cdot x + b)$ is the model's probability that $x$ belongs to class 1.

**Step 3: decide.** Predict class 1 if $p \ge 0.5$. That is the same as $z \ge 0$. The set of points where $w \cdot x + b = 0$ is the **decision boundary**: a straight line in 2D, or a flat plane in more dimensions. Logistic regression is therefore a **linear classifier**.

### Measuring error: log loss

For one example with true label $y$ (0 or 1) and predicted probability $p$:

$$L = -\big[\, y \log p + (1 - y) \log(1 - p) \,\big]$$

- If $y = 1$, the loss is $-\log p$. It is near 0 when $p \approx 1$ and grows huge when $p \approx 0$.
- If $y = 0$, the loss is $-\log(1 - p)$, which works the same way in reverse.

Confident wrong answers are punished heavily. The training loss is the average of $L$ over all examples.

### Learning: gradient descent

We need the $w$ and $b$ that minimize the average loss. **Gradient descent** does this by starting at $w = 0, b = 0$ and repeatedly taking a small step **downhill**, opposite to the gradient (the slope) of the loss:

$$w \leftarrow w - \text{lr} \cdot \frac{\partial L}{\partial w}, \qquad b \leftarrow b - \text{lr} \cdot \frac{\partial L}{\partial b}$$

Here **lr** is the *learning rate*, the size of each step.

**Deriving the gradient.** Using the chain rule, and the fact that the sigmoid's derivative is $\sigma'(z) = p(1-p)$:

$$\frac{\partial L}{\partial z} = -\frac{y}{p}\,p(1-p) + \frac{1-y}{1-p}\,p(1-p) = -y(1-p) + (1-y)p = p - y$$

Everything cancels down to the **error** $p - y$. Since $z = w \cdot x + b$, we have $\partial z / \partial w = x$ and $\partial z / \partial b = 1$. Averaged over $n$ examples, and written with matrices (where $X$ holds one example per row):

$$\nabla_w = \frac{1}{n} X^\top (p - y), \qquad \nabla_b = \frac{1}{n}\sum (p - y)$$

Each weight is nudged in proportion to how much its feature contributed to the errors.

This is the same idea as hill climbing (program 3), but going **downhill**. Instead of testing neighbours, it uses the slope to know which way is down.

### Worked example: classifying `[4, 4]`

The trained model is $w = [0.99, 0.49]$, $b = -5.74$:

$$z = 0.99 \cdot 4 + 0.49 \cdot 4 - 5.74 = 0.18 \;\Rightarrow\; p = \sigma(0.18) \approx 0.54 \ge 0.5 \;\Rightarrow\; \text{class } 1$$

The decision boundary is the line $0.99\,x_1 + 0.49\,x_2 = 5.74$, and `[4, 4]` lies just on the class-1 side of it.

## Line by line

- **L1** `import numpy as np`.
- **L3** `train(X, y, lr=0.1, iters=1000)` takes the features, labels (0/1), learning rate and number of gradient steps.
- **L4** Docstring.
- **L5** `w, b = np.zeros(X.shape[1]), 0.0` starts with one zero weight per feature (`X.shape[1]` is the number of columns) and a zero bias. At this start, every prediction is $\sigma(0) = 0.5$.
- **L6** `for _ in range(iters):` takes 1000 gradient steps.
- **L7** `p = 1 / (1 + np.exp(-(X @ w + b)))` is the **forward pass**. `X @ w` is a matrix-vector product, giving every example's score at once. Adding `b` and applying the sigmoid gives a probability for every example.
- **L8** `w -= lr * X.T @ (p - y) / len(y)` is the weight update. `p - y` is the error per example. `X.T @ (p - y) / n` is the gradient $\nabla_w$ derived above. Step against it, scaled by `lr`.
- **L9** `b -= lr * np.mean(p - y)` is the bias update. Its gradient is the mean error.
- **L10** Returns the learned parameters.
- **L12-13** `predict` computes the probabilities, compares them with `>= 0.5` (giving `True`/`False`), and `.astype(int)` turns those into `1`/`0`.
- **L15-17** The same toy data as kNN, with labels 0 and 1.
- **L19** Trains the model.
- **L20** Prints the learned weights and bias.
- **L21-22** Prints each test point's predicted class.

Output:

```
Weights: [0.99 0.49] Bias: -5.74
[2 2] -> class 0
[7 6] -> class 1
[4 4] -> class 1
```

kNN called `[4, 4]` class A (0), but logistic regression says class 1. The point sits right between the two groups. kNN decides by its local neighbours, while logistic regression draws one straight line through the whole space. Different models can reasonably disagree near a boundary.

---

# 7. Gaussian Naive Bayes — [naive_bayes.py](naive_bayes.py)

## The algorithm from scratch

### Bayes' theorem

We want the probability of each class $c$ given an observed point $x$. Bayes' theorem turns that around into quantities we can estimate from data:

$$P(c \mid x) = \frac{P(x \mid c)\, P(c)}{P(x)}$$

- $P(c)$, the **prior:** how common class $c$ is overall.
- $P(x \mid c)$, the **likelihood:** how typical $x$ is for points of class $c$.
- $P(c \mid x)$, the **posterior:** the probability we want.
- $P(x)$, the **evidence:** the same for every class, so it does not affect which class wins and can be ignored.

The prediction is the class with the highest posterior: the largest $P(x \mid c) \cdot P(c)$.

### The "naive" assumption

Estimating $P(x \mid c)$ for all features jointly is hard. Naive Bayes **assumes the features are independent of each other within a class**, so the joint likelihood becomes a simple product:

$$P(x \mid c) = P(x_1 \mid c) \cdot P(x_2 \mid c) \cdots P(x_d \mid c)$$

This assumption is rarely exactly true, which is why the method is called naive. It still works surprisingly well in practice, and it makes training trivial.

### "Gaussian": modelling each feature as a bell curve

For numeric features, each $P(x_i \mid c)$ is modelled as a **normal (Gaussian) distribution**, with a mean $\mu$ and variance $\sigma^2$ estimated from the training points of class $c$:

$$P(x_i \mid c) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{(x_i - \mu)^2}{2\sigma^2}\right)$$

So **training** means computing, for each class, its prior and each feature's mean and variance. Nothing else.

### The log trick

Multiplying many probabilities smaller than 1 can **underflow** to 0 in floating point. Logarithms turn products into sums, and because log only ever increases, the class with the largest product is also the class with the largest log. Taking the log of the Gaussian:

$$\log P(x_i \mid c) = -\tfrac12\left[\log(2\pi\sigma^2) + \frac{(x_i-\mu)^2}{\sigma^2}\right]$$

The score of class $c$ is:

$$\text{score}(c) = \log P(c) - \tfrac12 \sum_i \left[\log(2\pi\sigma_i^2) + \frac{(x_i-\mu_i)^2}{\sigma_i^2}\right]$$

The predicted class is the one with the highest score. This formula is exactly line 7 of the program.

## Line by line

- **L1** `import numpy as np`.
- **L3** `train(X, y)` learns the model.
- **L4** One dictionary comprehension builds the whole model:
  - `for c in np.unique(y)` loops over each distinct class (`'A'`, `'B'`).
  - `y == c` is a boolean mask that is `True` for rows of class `c`.
  - `np.mean(y == c)`: the mean of a True/False array is the fraction of `True` values. That fraction is the **prior** $P(c)$.
  - `X[y == c].mean(axis=0)`: the **mean** of each feature over the rows of class `c`.
  - `X[y == c].var(axis=0) + 1e-9`: the **variance** of each feature. The tiny `1e-9` keeps the variance above 0, because a feature with identical values in a class would otherwise cause a division by zero.
  - The result is `{class: (prior, means, variances)}`.
- **L6** `predict(X, model)` classifies many points at once.
- **L7-8** For each class, compute the log score of **every** test point at once:
  - `(X - mean) ** 2 / var`: the squared distance from the class mean, measured in units of variance, for every point and feature.
  - `np.log(2 * np.pi * var)`: the normalising term of the Gaussian.
  - `np.sum(..., axis=1)`: sums over features. This is the naive product, done in log space.
  - `np.log(prior) - 0.5 * ...`: the full score formula above.
  - `scores` ends up as a list with one array per class. Each array has one score per test point.
- **L9**
  - `np.argmax(scores, axis=0)`: NumPy stacks the list into a `(classes, points)` table and, for each point (column), takes the row of the highest score.
  - `np.array(list(model))[...]`: `list(model)` lists the class names in the same order as `model.values()`, so the winning index is turned back into a class name.
- **L11-13** The same toy data as kNN.
- **L15** Trains the model.
- **L16** Predicts the test points.
- **L17** Self-check: the model must classify its own training points correctly. If it can't, something is broken.
- **L18-19** Prints the predictions.

Output:

```
[2 2] -> A
[7 6] -> B
[4 4] -> A
```

---

# 8. Real-data versions — [working/](working/)

Toy data with 6 points is good for understanding, but real data is messier. The [working/](working/) folder runs programs 4-7 on a real, standard dataset, and measures how well each one performs.

## The Iris dataset — [working/iris.csv](working/iris.csv)

The **Iris** flower dataset (R. A. Fisher, 1936) is one of the most famous datasets in machine learning. It is downloaded from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/53/iris) and licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

- **150 flowers**, with **50 of each species**: *Iris setosa*, *Iris versicolor* and *Iris virginica*.
- **4 measurements** per flower, in cm: sepal length, sepal width, petal length and petal width.
- Each CSV row looks like `5.1,3.5,1.4,0.2,Iris-setosa`: the four measurements, then the species.

Setosa is easy to tell apart, because its petals are much smaller. Versicolor and virginica overlap, so they are where models make mistakes.

## Key ideas used in this folder

- **Train/test split.** Testing a model on the same data it learned from overstates how good it is, because the model may simply have memorized that data. So 20% of the flowers (30) are hidden during training and used only for testing.
- **Accuracy.** The fraction of test flowers whose species was predicted correctly.
- **Feature standardization.** Subtract each feature's mean and divide by its standard deviation, so every feature has mean 0 and spread 1. Gradient descent (logistic regression) then treats all features equally. The mean and standard deviation are computed from the **training data only**, so no information leaks from the test set.
- **Multiple restarts.** K-Means can settle in a bad local minimum (see program 5). Running it from several random starts and keeping the result with the lowest inertia fixes that.

## Results

| Program | Result |
|---|---|
| `knn_iris.py` | Test accuracy: k=1 90%, k=3 90%, k=5 93.33%, k=7 96.67% |
| `naive_bayes_iris.py` | Test accuracy 90%. Setosa 5/5, versicolor 13/13, virginica 9/12 |
| `kmeans_iris.py` | Setosa: all 50 in one cluster. Versicolor/virginica: 84 of 100 grouped correctly (without ever seeing labels) |
| `logistic_regression_iris.py` | Versicolor vs virginica: train accuracy 98.75%, test 85% |

The accuracies come from one particular 30-flower test set (`seed=0`). A different shuffle would move them by a few percent.

## [working/iris.py](working/iris.py): shared data helper

- **L1** `import csv`: Python's built-in CSV reader.
- **L2** `from pathlib import Path`: file path handling.
- **L4** `import numpy as np`.
- **L6** `FEATURES` holds the names of the 4 columns, used for printing.
- **L9** `load_iris()` reads the dataset.
- **L10** Docstring: returns a `150 x 4` array of measurements and an array of species names.
- **L11** `Path(__file__).with_name('iris.csv')` builds the path of `iris.csv` **in the same folder as this file**, so the scripts work whichever folder you run them from. `with open(...) as f` opens the file and closes it automatically afterwards.
- **L12** `[r for r in csv.reader(f) if r]` reads every row as a list of strings. `if r` skips empty rows, because the file ends with a blank line.
- **L13** `r[:4]` takes the first 4 values of each row, and `dtype=float` converts them from text to numbers. That array is `X`. `r[4]`, the species name, goes into `y`.
- **L16** `split(X, y, test_frac=0.2, seed=0)` divides the data into training and test sets.
- **L17** Docstring.
- **L18** `permutation(len(X))` returns the numbers `0..149` in a random (but seeded, so repeatable) order. The file is sorted by species, so shuffling is essential. Without it, the test set would contain only virginica.
- **L19** `n = int(len(X) * test_frac)` is the test set size: 30.
- **L20** The first `n` shuffled indexes become the test set, and the rest (120) the training set. Indexing `X` and `y` with the **same** indexes keeps every flower with its own label.

## [working/knn_iris.py](working/knn_iris.py)

- **L1-2** Imports, as in `knn.py`.
- **L4** Imports the loader and splitter from `iris.py`.
- **L7-13** `knn_predict`: identical to `knn.py`, plus a docstring (L8). Now each point has 4 features instead of 2. Euclidean distance works the same way in any number of dimensions.
- **L16** Loads the 150 flowers.
- **L17** Splits them into 120 for training and 30 for testing.
- **L19** Tries several values of `k` to show how the choice affects accuracy.
- **L20** `knn_predict(...) == y_test` gives a True/False array with one entry per test flower (`True` = correct). Its mean is the fraction correct, which is the accuracy.
- **L21** `:.2%` formats the number as a percentage with 2 decimals.

## [working/naive_bayes_iris.py](working/naive_bayes_iris.py)

- **L1-3** Imports.
- **L6-15** `train` and `predict`: identical to `naive_bayes.py`, plus docstrings. Each class now stores 4 means and 4 variances.
- **L18-19** Load and split.
- **L21** Train on the 120 training flowers.
- **L22** Predict the 30 test flowers.
- **L23** Overall accuracy.
- **L24-25** Per-species results.
  - `(preds == c) & (y_test == c)` is `True` where the flower **is** species `c` **and** was predicted as `c`. Summing that counts the correct ones.
  - `np.sum(y_test == c)` counts how many test flowers are species `c`.
  - The output shows that all of naive Bayes' mistakes are virginica flowers mistaken for versicolor, the overlapping pair.

## [working/kmeans_iris.py](working/kmeans_iris.py)

- **L1-3** Imports.
- **L6-20** `kmeans`: identical to `kmeans.py`, plus a docstring and a comment.
- **L23-25** `inertia(X, centroids, labels)` measures how tight a clustering is.
  - `centroids[labels]` builds, for every point, its own cluster's centroid (one row per point).
  - `(X - centroids[labels]) ** 2` gives each point's squared difference from that centroid.
  - `np.sum` adds it all up. Lower is better.
- **L28** Loads the data. **The species labels `y` are never given to K-Means.** They are only used afterwards, to check whether the clusters match real species.
- **L30** Runs K-Means 10 times (`seed=s` for `s = 0..9`) and keeps the result with the lowest inertia.
  - `kmeans(...) for s in range(10)` is a generator that produces 10 `(centroids, labels)` results.
  - `key=lambda r: inertia(X, *r)` scores each one. `*r` unpacks the pair into the `centroids` and `labels` arguments.
  - `min(...)` returns the best result, which is unpacked into `centroids, labels`.
  - Without the restarts, seed 0 alone got stuck in a poor solution: it split setosa into two clusters and merged versicolor with virginica.
- **L32** Prints the centroids with the feature names. Each centroid describes a "typical flower" for its cluster.
- **L33-36** For each cluster, counts which real species ended up in it. `np.unique(..., return_counts=True)` returns the distinct species and how often each appears. `.tolist()` converts both to plain Python values so they print cleanly.

Output:

```
Centroids (sepal length, sepal width, petal length, petal width):
 [[5.9  2.75 4.39 1.43]
 [6.85 3.07 5.74 2.07]
 [5.01 3.42 1.46 0.24]]

How many of each species landed in each cluster:
cluster 0: {'Iris-versicolor': 48, 'Iris-virginica': 14}
cluster 1: {'Iris-versicolor': 2, 'Iris-virginica': 36}
cluster 2: {'Iris-setosa': 50}
```

## [working/logistic_regression_iris.py](working/logistic_regression_iris.py)

- **L1-3** Imports.
- **L6-17** `train` and `predict`: identical to `logistic_regression.py`.
- **L20** Comment. Logistic regression as written here separates **two** classes. Setosa is dropped because it is trivially separable. The interesting test is the overlapping pair.
- **L21** Loads the data.
- **L22** `keep = y != 'Iris-setosa'` is a mask that is `True` for the 100 non-setosa flowers.
- **L23** Keeps only those rows. `(y[keep] == 'Iris-virginica').astype(int)` turns the species names into numbers: virginica becomes 1 and versicolor becomes 0.
- **L24** Splits the 100 flowers into 80 for training and 20 for testing.
- **L27** Computes the mean and standard deviation of each feature **from the training set only**.
- **L28** Standardizes both sets with those training statistics.
- **L30** Trains.
- **L31** Prints the weights. Petal length (2.86) and petal width (3.66) get the largest weights, so they are the features that best separate the two species. Sepal size matters much less.
- **L32-33** Train and test accuracy. Test accuracy (85%) is lower than train accuracy (98.75%). That gap is normal: the model is scored on flowers it has never seen.

---

## License and credits

- Code: written by PRATHVI9607 for learning purposes.
- Iris dataset: Fisher, R. A. (1936). UCI Machine Learning Repository, licensed under CC BY 4.0.
