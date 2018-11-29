# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """

        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    #print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    fringe = util.Stack()
    pacmanCurrentPosition = (problem.getStartState(),'',0)
    pacmanRoute = []
    visitedPosition = set()

    while (problem.isGoalState(pacmanCurrentPosition[0]) == False):
        pacmanSuccessor = problem.getSuccessors(pacmanCurrentPosition[0])
        visitedPosition.add(pacmanCurrentPosition[0])
        isSuccessor = 0
        for item in pacmanSuccessor:
            if item[0] not in visitedPosition:
                fringe.push(item)
                isSuccessor = 1
        if isSuccessor == 0:
            fringe.pop()
            pacmanRoute.pop()
            if fringe.isEmpty():
                print('No solution!')
                return []
        pacmanCurrentPosition = fringe.pop()
        fringe.push((pacmanCurrentPosition))
        if pacmanCurrentPosition[0] not in visitedPosition:
            pacmanRoute.append(pacmanCurrentPosition[1])

    print(len(pacmanRoute))
    return pacmanRoute

    # while (problem.isGoalState(pacmanCurrentPosition[0]) == False):
    #     fringe.push((pacmanCurrentPosition))
    #     pacmanSuccessor = problem.getSuccessors(pacmanCurrentPosition[0])
    #     visitedPosition.add(pacmanCurrentPosition[0])
    #     isSuccessor = 0
    #     for item in pacmanSuccessor:
    #         if item[0] not in visitedPosition:
    #             fringe.push(item)
    #             isSuccessor = 1
    #     if isSuccessor == 0:
    #         fringe.pop()
    #         pacmanRoute.pop()
    #         if fringe.isEmpty():
    #             print('No solution!')
    #             return []
    #     else:
    #         pacmanRoute.append(pacmanCurrentPosition[1])
    #     pacmanCurrentPosition = fringe.pop()
    #     pacmanRoute.pop()
    # print(len(pacmanRoute))
    # return pacmanRoute


    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    fringe = util.Queue()
    visitedPosition = set()
    visitedPosition.add(problem.getStartState())
    pacmanCurrent = [problem.getStartState(), []]
    fringe.push(pacmanCurrent)

    while(not fringe.isEmpty()):
        pacmanCurrent = fringe.pop()
        if problem.isGoalState(pacmanCurrent[0]):
            return pacmanCurrent[1]
        #Strategy
        pacmanSuccessors = problem.getSuccessors(pacmanCurrent[0])
        validSuccessors = []
        for item in pacmanSuccessors:
            if item[0] not in visitedPosition:
                pacmanRoute = pacmanCurrent[1].copy()
                pacmanRoute.append(item[1])
                validSuccessors.append([item[0], pacmanRoute])
        for item in validSuccessors:
            fringe.push(item)
            visitedPosition.add(item[0])

        if problem.isGoalState(pacmanCurrent[0]): return pacmanRoute

    print(problem._expanded)

    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    print('UCS')
    pacmanCurrent = [problem.getStartState(), [], 0]
    visitedPosition = set()
    #visitedPosition.add(problem.getStartState())
    fringe = util.PriorityQueue()
    fringe.push(pacmanCurrent, pacmanCurrent[2])
    while not fringe.isEmpty():
        pacmanCurrent = fringe.pop()
        if pacmanCurrent[0] in visitedPosition:
            continue
        else:
            visitedPosition.add(pacmanCurrent[0])
        if problem.isGoalState(pacmanCurrent[0]): return pacmanCurrent[1]
        else:
            pacmanSuccessors = problem.getSuccessors(pacmanCurrent[0])
        Successor = []
        for item in pacmanSuccessors:  # item: [(x,y), 'direction', cost]
            if item[0] not in visitedPosition:
                pacmanRoute = pacmanCurrent[1].copy()
                pacmanRoute.append(item[1])
                sumCost = pacmanCurrent[2]
                Successor.append([item[0], pacmanRoute, sumCost+item[2]])
        for item in Successor:
            fringe.push(item, item[2])
    return pacmanCurrent[1]


    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    pacmanCurrent = [problem.getStartState(), [], 0]
    fringe = util.PriorityQueue()
    fringe.push(pacmanCurrent, 0)
    closed = set()
    while not fringe.isEmpty():
        pacmanCurrent = fringe.pop()
        if problem.isGoalState(pacmanCurrent[0]): return pacmanCurrent[1]
        if not pacmanCurrent[0] in closed:
            closed.add(pacmanCurrent[0])
            for item in problem.getSuccessors(pacmanCurrent[0]):
                pacmanRoute = pacmanCurrent[1].copy()
                pacmanRoute.append(item[1])
                sumCost = pacmanCurrent[2]
                fringe.push([item[0], pacmanRoute, sumCost+item[2]], sumCost+item[2]+heuristic(item[0], problem))
    return []

    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
