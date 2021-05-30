
from __future__ import division
from __future__ import print_function

import sys, os
import psutil
import math
import time
import collections
import heapq
import queue as Q
import numpy as np


#### SKELETON CODE ####
## The Class that Represents the Puzzle
class PuzzleState(object):
    """
        The PuzzleState stores a board configuration and implements
        movement instructions to generate valid children.
    """
    def __init__(self, config, n, parent=None, action="Initial", cost=0, path=collections.deque()):
        """
        :param config->np array : Represents the n*n board, 
        	for e.g. [[0,1,2],[3,4,5],[6,7,8]] represents the goal state.
        :param n->int : Size of the board
        :param parent->PuzzleState
        :param action->string
        :param cost->int
        """
        if n*n != len(config) or n < 2:
            raise Exception("The length of config is not correct!")
        if set(config) != set(range(n*n)):
            raise Exception("Config contains invalid/duplicate entries : ", config)

        self.n        = n
        self.cost     = cost
        self.parent   = parent
        self.action   = action
        self.config   = np.reshape(np.array(config),(n, n)) # config from list n*n np arr
        self.children = collections.deque()
        self.f = 0	# f = cost + heuristic <- used in calculate_total_cost()

        # Get the index and (row, col) of empty block
        self.blank_index = np.where(self.config == 0)


    def display(self):
        """ Display this Puzzle state as a n*n board """
        for i in range(self.n):
            print(self.config[3*i : 3*(i+1)])

    def move_up(self):
        """ 
        Moves the blank tile one row up.
        :return a PuzzleState with the new configuration
        """
        # Check to see if move is possible. If not, no up child
        if self.blank_index[0] == 0:
        	return
        # Save the value to be moved to tmp var
        row = self.blank_index[0]
        col = self.blank_index[1]
        tmp = self.config[row - 1, col]

        # Copy the np array & swap the 0 and value it displaces
        child_config = np.copy(self.config)
        child_config[row - 1, col] = 0
        child_config[self.blank_index] = tmp
        # Reshape np arr to list for parameter of PuzzleState
        sz_reshape = self.n ** 2
        child_config = list(np.reshape(child_config, (sz_reshape,)))

        return PuzzleState(child_config, self.n, parent=self, action='Up', cost=self.cost+1)
      
    def move_down(self):
        """
        Moves the blank tile one row down.
        :return a PuzzleState with the new configuration
        """
        # Check to see if move is possible. If not, no up child
        if self.blank_index[0] == self.n - 1:
        	return
        # Save the value to be moved to tmp var
        row = self.blank_index[0]
        col = self.blank_index[1]
        tmp = self.config[row + 1, col]

        # Copy the np array & swap the 0 and value it displaces
        child_config = np.copy(self.config)
        child_config[row + 1, col] = 0
        child_config[self.blank_index] = tmp

        # Reshape np arr to list for parameter of PuzzleState
        sz_reshape = self.n ** 2
        child_config = list(np.reshape(child_config, (sz_reshape,)))

        return PuzzleState(child_config, self.n, parent=self, action='Down', cost=self.cost+1)
      
    def move_left(self):
        """
        Moves the blank tile one column to the left.
        :return a PuzzleState with the new configuration
        """
        # Check to see if move is possible. If not, no up child
        if self.blank_index[1] == 0:
        	return
        # Save the value to be moved to tmp var
        row = self.blank_index[0]
        col = self.blank_index[1]
        tmp = self.config[row, col - 1]

        # Copy the np array & swap the 0 and value it displaces
        child_config = np.copy(self.config)
        child_config[row, col - 1] = 0
        child_config[self.blank_index] = tmp

        # Reshape np arr to list for parameter of PuzzleState
        sz_reshape = self.n ** 2
        child_config = list(np.reshape(child_config, (sz_reshape,)))

        return PuzzleState(child_config, self.n, parent=self, action='Left', cost=self.cost+1)

    def move_right(self):
        """
        Moves the blank tile one column to the right.
        :return a PuzzleState with the new configuration
        """
        # Check to see if move is possible. If not, no up child
        if self.blank_index[1] == self.n - 1:
        	return
        # Save the value to be moved to tmp var
        row = self.blank_index[0]
        col = self.blank_index[1]
        tmp = self.config[row, col + 1]

        # Copy the np array & swap the 0 and value it displaces
        child_config = np.copy(self.config)
        child_config[row, col + 1] = 0
        child_config[self.blank_index] = tmp

        # Reshape np arr to list for parameter of PuzzleState
        sz_reshape = self.n ** 2
        child_config = list(np.reshape(child_config, (sz_reshape,)))

        return PuzzleState(child_config, self.n, parent=self, action='Right', cost=self.cost+1)
      
    def expand(self):
        """ Generate the child nodes of this node """
        
        # Node has already been expanded
        if len(self.children) != 0:
            return self.children
        
        # Add child nodes in order of UDLR
        children = collections.deque([
            self.move_up(),
            self.move_down(),
            self.move_left(),
            self.move_right()])

        # Compose self.children of all non-None children states
        self.children = collections.deque(state for state in children if state is not None)
        return self.children

    # Make states hashable for comparison in sets (np arrays are otherwise unhashable)
    def __hash__(self):
    	return hash(self.config.tobytes())

    """ Custom comparators for use in set and heap"""
    def __eq__(state1, state2):
    	return (state1.config == state2.config).all()

    def __gt__(state1, state2):
    	return state1.f > state2.f

    def __lt__(state1, state2):
    	return state1.f < state2.f

    # Override str to print array when PuzzleState obj to be printed
    def __str__(self):
    	return str(self.config)


# Allows for states to be compared based on cost
def state_key(state):
	return state.cost

# Function that Writes to output.txt
### Students need to change the method to have the corresponding parameters
def writeOutput(path, cost, nodesExpanded, searchDepth, maxDepth, runTime, ram):
    f = open("output.txt", "w")
    f.write("path_to_goal: " + str(list(path)) + "\n" + 
    	"cost_of_path: " + str(cost) + "\n" +
    	"nodes_expanded: " + str(nodesExpanded) + "\n" +
    	"search_depth: " + str(searchDepth) + "\n" +
    	"max_search_depth: " + str(maxDepth) + "\n" +
    	"running_time: " + "%.8f"%runTime + "\n" +
    	"max_ram_usage: " + "%.8f"%ram + "\n")
    f.close

def printOutput(path, cost, nodesExpanded, searchDepth, maxDepth, runTime, ram):
	print("path_to_goal: " + str(list(path)) + "\n" + 
    	"cost_of_path: " + str(cost) + "\n" +
    	"nodes_expanded: " + str(nodesExpanded) + "\n" +
    	"search_depth: " + str(searchDepth) + "\n" +
    	"max_search_depth: " + str(maxDepth) + "\n" +
    	"running_time: " + "%.8f"%runTime + "\n" +
    	"max_ram_usage: " + "%.8f"%ram)

def bfs_search(initial_state):
    """BFS search"""
    # Init frontier queue and explored set
    frontier = Q.Queue()
    frontierSet = set() # Set greatly reduces runtime vs. checking if state in deque (~20% faster)
    frontier.put(initial_state)
    frontierSet.add(initial_state)
    explored = set()

    # Diagnostics
    startTime = time.time()
    path = collections.deque()
    cost = len(path)
    nodesExpanded = 0
    searchDepth = 0
    maxDepth = 0

    # Loop until no more nodes in state space or goal found
    while not frontier.empty():
    	# 1) Remove node from frontier & add to explored
    	state = frontier.get()
    	explored.add(state)

    	# 2) Test state against goal - if so, get path & break
    	if test_goal(state):
    		get_path(state, path)
    		cost = state.cost
    		break

    	# 3) Not goal - Expand the node
    	nodesExpanded += 1
    	for child in state.expand():
    		if child not in explored and child not in frontierSet:
    			frontier.put(child)
    			frontierSet.add(child)

    # Collect system diagnostics
    stopTime = time.time()
    pid = os.getpid()
    py = psutil.Process(pid)
    maxRamUsage = py.memory_info()[0]/2.**30

    # Get search depths
    #     searchDepth == length of path by definition
    #     maxSearchDepth == max depth reached out of nodes that reached frontier
    searchDepth = len(path)
    maxDepth = max(frontierSet, key=state_key).cost

    # Write output to file
    writeOutput(path, cost, nodesExpanded, searchDepth, maxDepth, (stopTime-startTime), maxRamUsage)  

def dfs_search(initial_state):
    """DFS search"""
    # Init frontier stack and explored set
    frontier = collections.deque()
    frontierSet = set() # Set greatly reduces runtime vs. checking if state in deque
    frontier.append(initial_state)
    frontierSet.add(initial_state)
    explored = set()

    # Diagnostics
    startTime = time.time()
    progressTime = 0
    path = collections.deque()
    cost = len(path)
    nodesExpanded = 0
    searchDepth = 0
    maxDepth = 0

    # Loop until no more nodes in state space or goal found
    while len(frontier) > 0:
    	# 1) Remove node from frontier & add to explored
    	state = frontier.pop()
    	explored.add(state)

    	# 2) Test state against goal - if so, get path & break
    	if test_goal(state):
    		get_path(state, path)
    		cost = state.cost
    		break

    	# 3) Not goal - Expand the node
    	nodesExpanded += 1
    	for child in reversed(state.expand()):
    		if child not in explored and child not in frontierSet:
    			frontier.append(child)
    			frontierSet.add(child)

    # Collect system diagnostics
    stopTime = time.time()
    pid = os.getpid()
    py = psutil.Process(pid)
    maxRamUsage = py.memory_info()[0]/2.**30

    # Get search depths
    searchDepth = len(path)
    if frontier:
    	maxDepth = max(frontier, key=state_key).cost
    else:
    	maxDepth = max(explored, key=state_key).cost

    # Write output to file
    writeOutput(path, cost, nodesExpanded, searchDepth, maxDepth, (stopTime-startTime), maxRamUsage)

def A_star_search(initial_state):
    """A * search"""
    # Init frontier stack and explored set
    frontier = []
    frontierSet = set()
    explored = set()
    heapq.heappush(frontier, initial_state)

    # Diagnostics
    startTime = time.time()
    progressTime = 0
    path = collections.deque()
    cost = len(path)
    nodesExpanded = 0
    searchDepth = 0
    maxDepth = 0
    order = 0

    # Loop until no more nodes in state space or goal found
    while frontier:
    	# 1) Remove node from frontier & add to explored
    	state = heapq.heappop(frontier)
    	explored.add(state)

    	# 2) Test state against goal - if so, get path & break
    	if test_goal(state):
    		get_path(state, path)
    		break

    	# 3) Not goal - Expand the node
    	nodesExpanded += 1
    	for child in state.expand():
    		calculate_total_cost(child)
    		if child not in explored and child not in frontierSet:
    			heapq.heappush(frontier, child)
    		elif child in frontierSet:
    			# Try removing from frontier
    			frontier.remove(child)
    			heapq.heappush(frontier, child)

    # Collect system diagnostics
    stopTime = time.time()
    pid = os.getpid()
    py = psutil.Process(pid)
    maxRamUsage = py.memory_info()[0]/2.**30

    # Get search depths
    searchDepth = len(path)
    cost = len(path)
    if frontierSet:
    	maxDepth = max(frontierSet, key=state_key).cost
    else:
    	maxDepth = max(explored, key=state_key).cost

    # Write output to file
    writeOutput(path, cost, nodesExpanded, searchDepth, maxDepth, (stopTime-startTime), maxRamUsage)

def calculate_total_cost(state):
    """calculate the total estimated cost of a state"""
    # h = sum of manhattan distance for each tile from curr position to goal position
    h = 0
    for tile in np.nditer(state.config):
    	h += calculate_manhattan_dist(list(np.where(state.config == tile)), tile, state.n)
    # f(n) = g(n) + h(n)
    state.f = state.cost + h

def calculate_manhattan_dist(idx, value, n):
    """calculate the manhattan distance of a tile"""
    # Dict of coordinates for goal state
    goal = {0: [0, 0],
    	1: [0, 1],
    	2: [0, 2],
    	3: [1, 0],
    	4: [1, 1],
    	5: [1, 2],
    	6: [2, 0],
    	7: [2, 1],
    	8: [2, 2]}
    xDif = abs(idx[0][0] - goal[int(value)][0])
    yDif = abs(idx[1][0] - goal[int(value)][1])
    return xDif + yDif

def test_goal(puzzle_state):
    """test the state is the goal state or not"""
    goal = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    return (puzzle_state.config == goal).all()


def get_path(puzzle_state, path):
	"""return the state from root to parameter"""
	# Traverse from state to root to create a path in-place
	node = puzzle_state
	while node is not None:
		# Disregard initial state
		if node.action != 'Initial':
			path.appendleft(node.action)
		node = node.parent


# Main Function that reads in Input and Runs corresponding Algorithm
def main():
    search_mode = sys.argv[1].lower()
    begin_state = sys.argv[2].split(",")
    begin_state = list(map(int, begin_state))
    board_size  = int(math.sqrt(len(begin_state)))
    hard_state  = PuzzleState(begin_state, board_size)
    start_time  = time.time()
    
    if   search_mode == "bfs": bfs_search(hard_state)
    elif search_mode == "dfs": dfs_search(hard_state)
    elif search_mode == "ast": A_star_search(hard_state)
    else: 
        print("Enter valid command arguments !")
        
    end_time = time.time()
    print("Program completed in %.3f second(s)"%(end_time-start_time))

if __name__ == '__main__':
    main()