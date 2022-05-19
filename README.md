## N-Puzzle Solver

Welcome to n-puzzle solver. Feed in a comma separated n-puzzle and get back the moves required to solve it.

This repo holds an implementation of three seach algorithms to search all possible states of an n-puzzle, finding a path from the given state, to the puzzle's goal state.

To run this, ensure you `pip install requirements.txt`.

Then choose your search algorithm:
**bfs**: Breadth First Search
**dfs**: Depth First Search
**ast**: A-Star Search

And, of course, the n-puzzle you want to solve as a comma separated arg: `1,2,5,3,4,0,6,7,8`

To run the solver, run the following from the directory containing puzzle.py:

`python puzzle.py <search_algorithm> <comma_separated_npuzzle>`

Running the command will generate run statistics and the solving moves given your search algorithm in `output.txt`

So running `python puzzle.py bfs 1,2,5,3,4,0,6,7,8` generates:

```
path_to_goal: ['Up', 'Left', 'Left']
cost_of_path: 3
nodes_expanded: 10
search_depth: 3
max_search_depth: 4
running_time: 0.00140691
max_ram_usage: 0.02195740
```

While running `python puzzle.py bfs 1,2,5,3,4,0,6,7,8` generates:

```
path_to_goal: ['Up', 'Left', 'Left']
cost_of_path: 3
nodes_expanded: 181437
search_depth: 3
max_search_depth: 1
running_time: 18.12146282
max_ram_usage: 0.78310776
```

And lastly, A-Star proves to be much quickest with `python puzzle.py bfs 1,2,5,3,4,0,6,7,8` generating:

```
path_to_goal: ['Up', 'Left', 'Left']
cost_of_path: 3
nodes_expanded: 3
search_depth: 3
max_search_depth: 3
running_time: 0.00072789
max_ram_usage: 0.02154541
```



