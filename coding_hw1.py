from typing import List, Tuple, Dict, Optional, cast
from environments.environment_abstract import Environment, State
from environments.farm_grid_world import FarmState
from environments.n_puzzle import NPuzzleState
from heapq import heappush, heappop
import time
import numpy as np


class Node:
    def __init__(self, state: State, path_cost: float, parent_action: Optional[int], parent):
        self.state: State = state
        self.parent: Optional[Node] = parent
        self.path_cost: float = path_cost
        self.parent_action: Optional[int] = parent_action

    def __hash__(self):
        return self.state.__hash__()

    def __gt__(self, other):
        return self.path_cost < other.path_cost

    def __eq__(self, other):
        return self.state == other.state


def get_next_state_and_transition_cost(env: Environment, state: State, action: int) -> Tuple[State, float]:
    """

    :param env: Environment
    :param state: State
    :param action: Action
    :return: the next state and the transition cost
    """
    rw, states_a, _ = env.state_action_dynamics(state, action)
    state: State = states_a[0]
    transition_cost: float = -rw

    return state, transition_cost


def visualize_bfs(viz, closed_states: List[State], queue: List[Node], wait: float):
    """

    :param viz: visualizer
    :param closed_states: states in CLOSED
    :param queue: states in priority queue
    :param wait: number of seconds to wait after displaying
    :return: None
    """

    if viz is None:
        return

    grid_dim_x, grid_dim_y = viz.env.grid_shape
    for pos_i in range(grid_dim_x):
        for pos_j in range(grid_dim_y):
            viz.board.itemconfigure(viz.grid_squares[pos_i][pos_j], fill="white")

    for state_u in closed_states:
        pos_i_up, pos_j_up = state_u.agent_idx
        viz.board.itemconfigure(viz.grid_squares[pos_i_up][pos_j_up], fill="red")

    for node in queue:
        state_u: FarmState = cast(FarmState, node.state)
        pos_i_up, pos_j_up = state_u.agent_idx
        viz.board.itemconfigure(viz.grid_squares[pos_i_up][pos_j_up], fill="grey")

    viz.window.update()
    time.sleep(wait)


def manhattan_distance(state: State, env: Environment) -> int:
    #Written By Noah Robertson
    'return sum(abs(state.tiles - env.goal_tiles))'

    #Grabs the dimension of the board which is the sqrt of the amount of items in the tile array
    dim = np.sqrt(len(state.tiles))

    # Calculate row and column indices for state and goal
    state_stack = np.column_stack(divmod(state.tiles, dim))
    goal_stack = np.column_stack(divmod(env.goal_tiles, dim))

    # Calculate the Manhattan distance
    total_distance = np.sum(np.abs(state_stack - goal_stack))

    return total_distance

def path (node: Node) -> List[int]:
    #Written By Noah Robertson
    #Tracks the path from the final node through its parents
    path = []
    while (node.parent is not None):
        path.append(node.parent_action)
        node = node.parent
    #Reverses the path to get the actual path taken
    path.reverse()
    return path

def search_optimal(state_start: State, env: Environment, viz) -> Optional[List[int]]:
    #Written By Noah Robertson
    """ Return an optimal path

    :param state_start: starting state
    :param env: environment
    :param viz: visualization object

    :return: a list of integers representing the actions that should be taken to reach the goal or None if no solution
    """
    # Based off of pseudocode in the slides

    'Uniform Cost Search:'
    """
    node = Node(state_start, 0, None, None)
    frontier = []
    reached = {}
    heappush(frontier, (node.path_cost, node))

    while frontier:
        _, current_node = heappop(frontier)
        if env.is_terminal(current_node.state):
            return path(current_node)
        for action in env.get_actions(current_node.state):
            child_state, child_cost = get_next_state_and_transition_cost(env, current_node.state, action)
            child = Node(child_state, child_cost + current_node.path_cost, action, current_node)
            if (child.state not in reached) or (child.path_cost < reached[child.state].path_cost):
                viz_queue = (x[1] for x in frontier)
                visualize_bfs(viz,list(reached.keys()), viz_queue, 0.05)
                reached[child.state] = child
                heappush(frontier, (child.path_cost, child))
    """

    #Added heuristics to Uniform Cost Search to implement A*
    #Uses an optimized manhattan distance function for the heurtistics function

    'A* Search:'
    node = Node(state_start, 0, None, None)
    frontier = []
    reached = {}
    heappush(frontier, (node.path_cost, node))
    while frontier:
        _, current_node = heappop(frontier)
        if env.is_terminal(current_node.state):
            return path(current_node)
        for action in env.get_actions(current_node.state):
            child_state, child_cost = get_next_state_and_transition_cost(env, current_node.state, action)
            child_heuristic = manhattan_distance(child_state, env)
            child = Node(child_state, child_cost + current_node.path_cost, action, current_node)
            if (child.state not in reached) or (child.path_cost < reached[child.state].path_cost):
                reached[child.state] = child
                heappush(frontier, (child.path_cost + child_heuristic, child))
                

        


def search_speed(state_start: State, env: Environment, viz) -> Optional[List[int]]:
    #Written By Noah Robertson
    """ Return a path as quickly as possible

    :param state_start: starting state
    :param env: environment
    :param viz: visualization object

    :return: a list of integers representing the actions that should be taken to reach the goal or None if no solution
    """

    #Adapted Uniform Cost Search based of heuristics

    'Greedy BFS'
    node = Node(state_start, 0, None, None)
    frontier = []
    reached = {}
    heappush(frontier, (node.path_cost, node))

    while frontier:
        _, current_node = heappop(frontier)
        if env.is_terminal(current_node.state):
            return path(current_node)
        for action in env.get_actions(current_node.state):
            child_state, child_cost = get_next_state_and_transition_cost(env, current_node.state, action)
            child_heuristic = manhattan_distance(child_state, env)
            child = Node(child_state, child_cost + current_node.path_cost, action, current_node)
            if child.state not in reached:
                reached[child.state] = child
                heappush(frontier, (child_heuristic, child))