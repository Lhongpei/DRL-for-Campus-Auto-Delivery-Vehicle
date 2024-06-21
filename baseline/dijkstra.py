import heapq
def dijkstra_grid_with_weights(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    distances = [[float('inf')] * cols for _ in range(rows)]
    distances[start[0]][start[1]] = 0
    priority_queue = [(0, start)]
    predecessors = {start: None}

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] 

    while priority_queue:
        current_distance, (current_row, current_col) = heapq.heappop(priority_queue)

        if (current_row, current_col) == goal:
            break

        for dr, dc in directions:
            neighbor_row, neighbor_col = current_row + dr, current_col + dc

            if 0 <= neighbor_row < rows and 0 <= neighbor_col < cols:
                if grid[neighbor_row][neighbor_col] == float('inf'):  
                    continue

                distance = current_distance + grid[neighbor_row][neighbor_col]
                if distance < distances[neighbor_row][neighbor_col]:
                    distances[neighbor_row][neighbor_col] = distance
                    heapq.heappush(priority_queue, (distance, (neighbor_row, neighbor_col)))
                    predecessors[(neighbor_row, neighbor_col)] = (current_row, current_col)

    return distances, predecessors

def reconstruct_path(predecessors, goal):
    path = []
    current = goal
    while current:
        path.append(current)
        current = predecessors.get(current)
    path.reverse()
    return path