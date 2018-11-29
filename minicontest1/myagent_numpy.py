# myAgents.py
# ---------------
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
import numpy
from game import Agent
from searchProblems import PositionSearchProblem
"""
IMPORTANT
`agent` defines which agent you will use. By default, it is set to ClosestDotAgent,
but when you're ready to test your own agent, replace it with MyAgent
"""
def createAgents(num_pacmen, agent='MyAgent'):
    return [eval(agent)(index=i) for i in range(num_pacmen)]

def initializeK(X, K, birthPositions):
    X = numpy.array(X)
    C = birthPositions
    for k in range(len(birthPositions), K):
        D2 = numpy.array([min([abs(c[0]-x[0])+abs(c[1]-x[1]) for c in C]) for x in X])
        probs = D2/D2.sum()
        cumprobs = probs.cumsum()
        r = numpy.random.rand()
        for j,p in enumerate(cumprobs):
            if r < p:
                i = j
                break
        C.append(X[i])
    return C


class MyAgent(Agent):
    """
    Implementation of your agent.
    """

    def getAction(self, state):
        """
        Returns the next action the agent will take
        """

        "*** YOUR CODE HERE ***"
        if self.goal is not None:
            x, y = self.goal
            if not state.hasFood(x, y) and len(self.actions) > 0:
                if self.init > 0:
                    self.actions = []
                    self.goal = None
                else:
                    m = state.getNumFood()
                    skip = m // 2 // state.getNumAgents()
                    if skip == 0:
                        skip = m - 1
                    problem = SkipFoodSearchProblem(state, self.index, skip)
                    self.actions, self.goal = breadthFirstSearchWithGoalStateReturnNoaddFilter(problem, self.filterSet)
        if len(self.actions) > 0:
            action = self.actions[0]
            del self.actions[0]
            return action
        else:
            problem = AnyFoodSearchProblem(state, self.index)
            self.actions, self.goal, self.filterSet = breadthFirstSearchWithGoalStateReturn(problem, self.filterSet)
            self.init -= 1
            return self.getAction(state)

    def initialize(self):
        """
        Intialize anything you want to here. This function is called
        when the agent is first created. If you don't need to use it, then
        leave it blank
        """

        "*** YOUR CODE HERE"
        self.init = 4
        self.actions = []
        self.goal = None
        self.filterSet = set([])

    def registerInitialState(self, state):

        if self.index==0:

            n = state.getNumAgents()
            self.pacmanNumber = n
            if n == 1:
                return
            clusteredPacmans = []
            remainPacmans = state.getPacmanPositions()
            while remainPacmans :
                problem = PacmanSearchProblem_Position(state, remainPacmans[0])
                nearbyPacmen = breadthFirstSearchCountLimited(problem, 7)
                clusteredPacmans.append(nearbyPacmen)
                for position in nearbyPacmen:
                    remainPacmans.remove(position)

            X = state.getFood().asList()
            #print(X)
            birthPosition = []
            for pacmanCluster in clusteredPacmans:
                birthPosition.append(numpy.array(pacmanCluster[0]))

            pacmanDestinations = initializeK(X, n, birthPosition)
            MyAgent.pacmanGo2Destination = [0]*n
            pacmanPositions = state.getPacmanPositions()
            pacmanDestinationslist = []
            for pacmanDestination in pacmanDestinations:
                pacmanDestinationslist.append(tuple(pacmanDestination))

            pacmanPositionsCopy = pacmanPositions.copy()

            for destination in pacmanDestinationslist:
                minP2D = 99999
                for pacmanIndex, pacmanPosition in enumerate(pacmanPositionsCopy):
                    tempDistance = manhattanDistance(pacmanPosition, destination)
                    if tempDistance < minP2D:
                        minP2D = tempDistance
                        minpacmanPosition = pacmanPosition
                MyAgent.pacmanGo2Destination[pacmanPositions.index(minpacmanPosition)] = destination
                pacmanPositionsCopy.remove(minpacmanPosition)
            #print(MyAgent.pacmanGo2Destination)
        else:
            problem = PositionSearchProblem(state, agentIndex=self.index, goal=tuple(MyAgent.pacmanGo2Destination[self.index]), warn=False)
            self.actions = aStarSearch(problem)

            self.mode = "GoToSearchRegion"





"""
Put any other SearchProblems or search methods below. You may also import classes/methods in
search.py and searchProblems.py. (ClosestDotAgent as an example below)
"""
def manhattanHeuristic(position, problem, info={}):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

def aStarSearch(problem, heuristic=manhattanHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    pacmanCurrent = [problem.getStartState(), [], 0]
    import util
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

def manhattanDistance(xy1, xy2):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])


def breadthFirstSearchWithGoalStateReturnNoaddFilter(problem, filterSet):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    closed_set = filterSet.copy()
    from util import Queue
    fringes = Queue()
    state = problem.getStartState()
    node = (state, None)
    temp_fringe = [node]
    fringes.push(temp_fringe)
    while not fringes.isEmpty():
        fringe = fringes.pop()
        state = fringe[-1][0]
        if problem.isGoalState(state):
            actions = []
            for node in fringe[1:]:
                actions.append(node[1])
            return actions, state
        if state not in closed_set:
            closed_set.add(state)
            successors = problem.getSuccessors(state)
            for successor in successors:
                if successor[0] in closed_set:
                    continue
                node = (successor[0], successor[1])
                temp_fringe = fringe.copy()
                temp_fringe.append(node)
                fringes.push(temp_fringe)
    return [], None


def breadthFirstSearchLimited(problem, limit=3):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    closed_set = set([])
    from util import Queue
    fringes = Queue()
    state = problem.getStartState()
    node = (state, None)
    temp_fringe = [node]
    fringes.push(temp_fringe)
    while not fringes.isEmpty():
        fringe = fringes.pop()
        state = fringe[-1][0]
        if problem.isGoalState(state):
            actions = []
            for node in fringe[1:]:
                actions.append(node[1])
            return actions
        if state not in closed_set:
            closed_set.add(state)
            successors = problem.getSuccessors(state)
            successors.reverse()
            for successor in successors:
                if successor[0] in closed_set:
                    continue
                node = (successor[0], successor[1])
                temp_fringe = fringe.copy()
                temp_fringe.append(node)
                if len(temp_fringe) > limit:
                    continue
                fringes.push(temp_fringe)
    # print('Not found!')
    return []


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    closed_set = set([])
    from util import Queue
    fringes = Queue()
    state = problem.getStartState()
    node = (state, None)
    temp_fringe = [node]
    fringes.push(temp_fringe)
    while not fringes.isEmpty():
        fringe = fringes.pop()
        state = fringe[-1][0]
        if problem.isGoalState(state):
            actions = []
            for node in fringe[1:]:
                actions.append(node[1])
            return actions
        if state not in closed_set:
            closed_set.add(state)
            successors = problem.getSuccessors(state)
            for successor in successors:
                if successor[0] in closed_set:
                    continue
                node = (successor[0], successor[1])
                temp_fringe = fringe.copy()
                temp_fringe.append(node)
                fringes.push(temp_fringe)
    # print('Not found!')
    return []

def breadthFirstSearchWithGoalStateReturn(problem, filterSet):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    childNum = {}
    closed_set = filterSet.copy()
    dead_fringe = []
    tentativeDict = {}
    from util import Queue
    fringes = Queue()
    state = problem.getStartState()
    node = (state, None)
    temp_fringe = [node]
    fringes.push(temp_fringe)
    while not fringes.isEmpty():
        fringe = fringes.pop()
        state = fringe[-1][0]
        if problem.isGoalState(state):
            actions = []
            for node in fringe[1:]:
                actions.append(node[1])
            for f in dead_fringe:
                i = -1
                father = f[i][0]
                childNumber = childNum[father]
                while childNumber == 0:
                    tmp = father
                    i -= 1
                    father = f[i][0]
                    childNumber = childNum[father] - 1
                    childNum[father] = childNumber
                    if tentativeDict.get(father) is not None:
                        tentativeDict[father].append(tmp)
                    else:
                        tentativeDict[father] = [tmp]
                    if childNumber == 0:
                        filterSet.add(father)
                        for child in tentativeDict[father]:
                            filterSet.remove(child)
            return actions, state, filterSet
        if state not in closed_set:
            closed_set.add(state)
            successors = problem.getSuccessors(state)
            cnt = len(successors)
            for successor in successors:
                if successor[0] in closed_set:
                    cnt -= 1
                    continue
                node = (successor[0], successor[1])
                temp_fringe = fringe.copy()
                temp_fringe.append(node)
                fringes.push(temp_fringe)
            if cnt == 0:
                dead_fringe.append(fringe)
                filterSet.add(fringe[-1][0])
            childNum[state] = cnt
    return [], None, filterSet

# """寰呬紭鍖�"""
def breadthFirstSearchCountLimited(problem, limit=3):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # count = 0
    goals = []
    closed_set = set([])
    from util import Queue
    fringes = Queue()
    state = problem.getStartState()
    node = state
    temp_fringe = [node]
    fringes.push(temp_fringe)
    while not fringes.isEmpty():
        fringe = fringes.pop()
        state = fringe[-1]
        if state not in closed_set:
            closed_set.add(state)
            if problem.isGoalState(state):
                goals.append(state)
            successors = problem.getSuccessors(state)
            for successor in successors:
                if successor[0] in closed_set:
                    continue
                node = successor[0]
                temp_fringe = fringe.copy()
                temp_fringe.append(node)
                if len(temp_fringe) > limit:
                    continue
                fringes.push(temp_fringe)
    return goals


class SkipFoodSearchProblem(PositionSearchProblem):
    def __init__(self, gameState, agentIndex, skip):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = gameState.getFood()
        self.skip = skip
        self.skipedFood = set([])

        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition(agentIndex)
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def isGoalState(self, state):
        x,y = state
        if self.food[x][y] == True:
            if state in self.skipedFood:
                return False
            if self.skip > 0:
                self.skip -= 1
                self.skipedFood.add(state)
                return False
            # print(self.skipedFood)
            return True
            # print("Skip food:", state)
        else:
            return False


class AnyFoodSearchProblem(PositionSearchProblem):
    def __init__(self, gameState, agentIndex):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = gameState.getFood()
        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition(agentIndex)
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def isGoalState(self, state):
        x,y = state
        return self.food[x][y] == True


class PacmanSearchProblem(PositionSearchProblem):
    def __init__(self, gameState, agentIndex):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.pacmanPositions = set(gameState.getPacmanPositions())

        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition(agentIndex)
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def isGoalState(self, state):
        return state in self.pacmanPositions

class PacmanSearchProblem_Position(PositionSearchProblem):
    def __init__(self, gameState, agentPosition):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.pacmanPositions = set(gameState.getPacmanPositions())

        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.getWalls()
        self.costFn = lambda x: 1
        self.startState = agentPosition
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def isGoalState(self, state):
        return state in self.pacmanPositions