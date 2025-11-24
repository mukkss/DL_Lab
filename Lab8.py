from collections import deque

# Grid
grid = [
    ["X",".",".",".","."],
    [".","X","X","G","."],
    [".",".",".","X","."],
    ["S",".",".",".","."]]

start = (3, 0)
goal  = (1, 3)

ROWS, COLS = len(grid), len(grid[0])

# Movements
moves = {
    "up":    (-1, 0),
    "down":  (1, 0),
    "left":  (0, -1),
    "right": (0, 1)
}

# BFS setup
queue = deque([start])
visited = set([start])
parent = {}      # child -> (parent, action)

# BFS Search
while queue:
    r, c = queue.popleft()

    if (r, c) == goal:
        break

    for action, (dr, dc) in moves.items():
        nr, nc = r + dr, c + dc

        # Skip invalid positions
        if not (0 <= nr < ROWS and 0 <= nc < COLS):
            continue
        if grid[nr][nc] == "X":
            continue
        if (nr, nc) in visited:
            continue

        visited.add((nr, nc))
        parent[(nr, nc)] = ((r, c), action)
        queue.append((nr, nc))

# Path reconstruction
if goal not in parent:
    print("No path!")
else:
    actions = []
    cur = goal

    # Walk backwards from goal → start
    while cur != start:
        cur, action = parent[cur]
        actions.append(action)

    actions.reverse()
    print("Path:", actions)

    # Show positions taken
    pos = start
    for act in actions:
        dr, dc = moves[act]
        pos = (pos[0] + dr, pos[1] + dc)
        print(pos)
