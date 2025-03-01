import math
from collections import deque
import numpy as np


def generate_grid(edge_servers):
    adjacency_list = {}
    rows = math.isqrt(edge_servers)  # Approximate square root for rows
    cols = math.ceil(edge_servers / rows)

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]
    positions = {}  # Store positions for visualization

    index = 0
    for r in range(rows):
        for c in range(cols):
            if index >= edge_servers:
                continue
            adjacency_list[index] = set()
            positions[index] = (c + 0.5 * (r % 2), -r)  # Offset odd rows

            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                neighbor_index = nr * cols + nc
                if 0 <= nr < rows and 0 <= nc < cols and neighbor_index < edge_servers:
                    adjacency_list[index].add(neighbor_index)
            index += 1

    return adjacency_list, positions


def get_shortest_path(adjacency_list, start, end):
    queue = deque([(start, [start])])
    visited = set()

    while queue:
        node, path = queue.popleft()
        if node in visited:
            continue
        visited.add(node)

        if node == end:
            return path

        for neighbor in adjacency_list[node]:
            queue.append((neighbor, path + [neighbor]))

    return None


def remove_node(adjacency_list, index):
    if index in adjacency_list:
        adjacency_list[index] = {}
    for neighbors in adjacency_list.values():
        if index in neighbors:
            neighbors.remove(index)

    return adjacency_list


def shortest_hop_distance(adjacency_list):
    shortest_distances = np.ones((len(adjacency_list), len(adjacency_list))) * 0
    for i in range(len(adjacency_list)):
        for j in range(len(adjacency_list)):
            if i == j:
                shortest_distances[i, j] = 0
            else:
                path = get_shortest_path(adjacency_list, i, j)
                if path is None:
                    shortest_distances[i, j] = np.inf
                else:
                    shortest_distances[i, j] = len(path) - 1

    return shortest_distances


if __name__ == "__main__":
    edge_servers = 3

    adjacency_list, positions = generate_grid(edge_servers)

    st = shortest_hop_distance(adjacency_list)
    print(st)

    adjacency_list = remove_node(adjacency_list, 1)
    st = shortest_hop_distance(adjacency_list)
    print(st)
