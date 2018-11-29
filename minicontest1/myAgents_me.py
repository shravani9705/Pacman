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

from game import Agent
from searchProblems import PositionSearchProblem

"""
Version2.1:全新策略

调参，目前最佳
普通人的init             self.init = 3
skip = 0的人的init       self.init = SKIPCOEF // 2
开始转战的系数            SKIPCOEF = (m // 2 // (maxIndex - minIndex))+1
开始后转战步数            skip = m // 2 // self.pacmanNumber

最高得分：1190.6690497116244
"""

"""
IMPORTANT
`agent` defines which agent you will use. By default, it is set to ClosestDotAgent,
but when you're ready to test your own agent, replace it with MyAgent
"""
def createAgents(num_pacmen, agent='MyAgent'):
    return [eval(agent)(index=i) for i in range(num_pacmen)]


import scipy
def initializeK(X, K, birthPositions):
    X = scipy.array(X)
    C = birthPositions
    for k in range(len(birthPositions), K):
        D2 = scipy.array([min([abs(c[0]-x[0])+abs(c[1]-x[1]) for c in C]) for x in X])
        probs = D2/D2.sum()
        cumprobs = probs.cumsum()
        r = scipy.rand()
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
    pacmanGo2Destination = []
    def getAction(self, state):
        """
        Returns the next action the agent will take
        """

        "*** YOUR CODE HERE ***"
        if state.getNumFood <= 5:
            outliers = getOutliers(state, 5, 5)


            problem = PositionSearchProblem(state, agentIndex=self.index,
                                                    goal=outliers[self.index], warn=False)
                    #print("gannar go to outlier", targetOutlier, "haha", "I am", self.index, "I am at", state.getPacmanPosition(self.index))
            self.actions = aStarSearch(problem)
            return self.getAction(state)
        if self.goal is not None:
            x, y = self.goal
            if not state.hasFood(x, y) and len(self.actions) > 0:
                self.mode = "SkipSearch"
                self.goal = None

                m = state.getNumFood()
                skip = m // 2 // state.getNumAgents()
                if skip == 0:
                    skip = m - 1
                problem = SkipFoodSearchProblem(state, self.index, skip)
                self.actions, self.goal = breadthFirstSearchWithGoalStateReturn(problem)

            # if not state.hasFood(x, y):
            #     self.mode = "LimitedDepthSearch"
            #     #print("first time", self.index)
            #     self.goal = None
            #     return self.getAction(state)
        if len(self.actions) > 0:
            action = self.actions[0]
            del self.actions[0]
            return action
        else:
            problem = AnyFoodSearchProblem(state, self.index)
            # self.actions = breadthFirstSearch(problem)
            self.actions, self.goal = breadthFirstSearchWithGoalStateReturn(problem)
            if n
            return self.getAction(state)
        # else:
        #     if self.mode == "GoToSearchRegion" or self.mode == "LimitedDepthSearch":
        #         self.mode = "LimitedDepthSearch"
        #         n = state.getNumAgents()
        #         m = state.getNumFood()
        #         width = state.getWidth()
        #         height = state.getHeight()
        #         depth = width*height // n // 10
        #         #print("depth: ", depth)
        #
        #         problem = LimitedDepthSearchProblem(state, self.index)
        #         self.actions, self.goal = breadthFirstSearchLimited(problem, depth)
        #         #print("limited depth agent:", self.index, "route", self.actions)
        #         if not self.actions:
        #             self.mode = "LimitedDepth2GoToOutlier"
        #             #print("I am ", self.index, "I will go to outliers, haha")
        #
        #         return self.getAction(state)
        #     elif self.mode == "LimitedDepth2GoToOutlier" or self.mode == "SkipSearch":
        #         self.mode = "LimitedDepthSearch"
        #         if not self.outliers:
        #             self.outliers = (getOutliers(state, max(state.getWidth()//3, 2), max(state.getHeight()//3, 2)))
        #             #print(self.outliers)
        #
        #         if not self.outliers:
        #             # self.coef *= 1.5
        #             #print("become a dumbass, haha", self.index, state.getPacmanPosition(self.index))
        #             return self.getAction(state)
        #         for i in range(len(self.outliers)):
        #             if i == 0:
        #                 #print(self.outliers)
        #                 minDistance = manhattanDistance(self.outliers[i], state.getPacmanPosition(self.index))
        #                 targetOutlier = self.outliers[i]
        #
        #             else:
        #                 if manhattanDistance(self.outliers[i], state.getPacmanPosition(self.index)) < minDistance:
        #                     targetOutlier = self.outliers[i]
        #         self.outliers.remove(targetOutlier)
        #         problem = PositionSearchProblem(state, agentIndex=self.index,
        #                                         goal=targetOutlier, warn=False)
        #         #print("gannar go to outlier", targetOutlier, "haha", "I am", self.index, "I am at", state.getPacmanPosition(self.index))
        #         self.actions = aStarSearch(problem)
        #         return self.getAction(state)
            # elif self.mode == "GoToOutlier":
            #     self.mode = "LimitedDepthSearch"
            #     return self.getAction(state)
            # elif self.mode == "GoToOutlier" or self.mode == "AnyFoodSearch":
            #     self.mode == "AnyFoodSearch"
            #     #print("become a dumbass, haha", self.index, state.getPacmanPosition(self.index))
            #     problem = AnyFoodSearchProblem(state, self.index)
            #     # self.actions = breadthFirstSearch(problem)
            #     self.actions = breadthFirstSearch(problem)
            #     return self.getAction(state)






        # if self.goal is not None:
        #     x, y = self.goal
        #     if not state.hasFood(x, y) and len(self.actions) > 0:
        #         if self.init > 0:
        #             self.actions = []
        #             self.goal = None
        #             print("Agent:", self.index, self.init)
        #         else:
        #             m = state.getNumFood()
        #             skip = m // 2 // self.pacmanNumber
        #             if skip == 0:
        #                 skip = m - 1
        #             problem = SkipFoodSearchProblem(state, self.index, skip)
        #             self.actions, self.goal = breadthFirstSearchWithGoalStateReturn(problem)
        #             # self.goal = None
        # if len(self.actions) > 0:
        #     action = self.actions[0]
        #     del self.actions[0]
        #     return action
        # else:
        #     problem = AnyFoodSearchProblem(state, self.index)
        #     # self.actions = breadthFirstSearch(problem)
        #     self.actions, self.goal = breadthFirstSearchWithGoalStateReturn(problem)
        #     self.init -= 1
        #     return self.getAction(state)

    def initialize(self):
        """
        Intialize anything you want to here. This function is called
        when the agent is first created. If you don't need to use it, then
        leave it blank
        """

        "*** YOUR CODE HERE"
        self.mode = "GoToSearchRegion"
        self.actions = []
        self.goal = None
        self.coef = 1
        self.outliers = []

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
                birthPosition.append(scipy.array(pacmanCluster[0]))

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



        # pacmanPositions = state.getPacmanPositions()
        # m = state.getNumFood()
        # pacmanIndexes = {}
        # for i in range(n):
        #     pacmanIndexes[pacmanPositions[i]] = i
        # # myPosition = pacmanPositions[self.index]
        # # print("Agent:", self.index, pacmanPositions)
        # # print(nearbyPacmenNumber)
        # minIndex = self.index
        # maxIndex = self.index
        # for pacman in nearbyPacmen:
        #     tmp = pacmanIndexes[pacman]
        #     if tmp < minIndex:
        #         minIndex = tmp
        #     elif tmp > maxIndex:
        #         maxIndex = tmp
        # rank = self.index - minIndex
        # X = state.getFood().asList()
        # print(X)
        # if self.index == 0:
        #     problem = AnyFoodSearchProblem(state, self.index)
        #     self.actions = breadthFirstSearch(problem)
        # else:
        #     pacmanDestinations = initializeK(X, nearbyPacmenNumber-1)
        #     problem = PositionSearchProblem(state, agentIndex=self.index, goal=tuple(pacmanDestinations[self.index-minIndex-1]))
        #     self.actions = aStarSearch(problem)



        # # for i in range(n):
        # #     # if pacmanPositions[i] == myPosition:
        # #     if pacmanPositions[i][0] and pacmanPositions[i][1]
        # #         nearbyPacmen.append(i)
        # # print("Agent:", self.index, 'NearbyPacman:', len(nearbyPacmen))
        # SKIPCOEF = (m // 2 // (maxIndex - minIndex))+1
        # # print("SKIPCOEF", SKIPCOEF)
        # rank = self.index - minIndex
        # skip = rank * SKIPCOEF
        # # if skip == 0:
        # #     self.init = SKIPCOEF // 2 - 1
        # if skip == 0:
        #     self.init = SKIPCOEF // 2 - 1
        # else:
        #     self.init = SKIPCOEF // 2**(rank+2)
        # # print("Agent:", self.index, 'Skip:', skip)
        # problem = SkipFoodSearchProblem(state, self.index, skip)
        # self.actions = breadthFirstSearch(problem)




"""
Put any other SearchProblems or search methods below. You may also import classes/methods in
search.py and searchProblems.py. (ClosestDotAgent as an example below)
"""

def getOutliers(state, width, height):
    foodGrid = state.getFood()
    foodList = foodGrid.asList()
    pacmanList = state.getPacmanPositions()
    outliers = []
    outliersUp = 5
    for food in foodList:

        foodRegion = getRegion(food, state, width, height)
        foodSum = 0

        for i in range(foodRegion[0],foodRegion[1]):
            breakflag = 0
            for j in range(foodRegion[2], foodRegion[3]):
                if foodGrid[i][j] == True:
                    foodSum += 1

        if foodSum < 3: outliers.append(food)
        if len(outliers) >= outliersUp: return outliers
    return outliers
        # if breakflag == 0 and foodSum <= 3:
        #     return food
        # if breakflag == 0 and food
        #

def getRegion(position, state, width, height):
    mapWidth = state.getWidth()
    mapHeight = state.getHeight()
    leftborder = max(1, position[0]-width//2)
    rightborder = min(mapWidth-1, position[0]+width//2)
    topborder = min(mapHeight-1, position[1]+height//2)
    bottomborder = max(1, position[1]-height//2)

    return [leftborder, rightborder, topborder, bottomborder]







def manhattanDistance(xy1, xy2):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

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
            return actions, state
        if state not in closed_set:
            closed_set.add(state)
            successors = problem.getSuccessors(state)
            successors.reverse()
            for successor in successors:
                if successor[0] == state:
                    continue
                node = (successor[0], successor[1])
                temp_fringe = fringe.copy()
                temp_fringe.append(node)
                if len(temp_fringe) > limit:
                    continue
                fringes.push(temp_fringe)
    # print('Not found!')
    return [], None


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
                if successor[0] == state:
                    continue
                node = (successor[0], successor[1])
                temp_fringe = fringe.copy()
                temp_fringe.append(node)
                fringes.push(temp_fringe)
    # print('Not found!')
    return []

def breadthFirstSearchWithGoalStateReturn(problem):
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
            return actions, state
        if state not in closed_set:
            closed_set.add(state)
            successors = problem.getSuccessors(state)
            for successor in successors:
                if successor[0] == state:
                    continue
                node = (successor[0], successor[1])
                temp_fringe = fringe.copy()
                temp_fringe.append(node)
                fringes.push(temp_fringe)
    # print('Not found!')
    return [], None

# """待优化"""
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
                if successor[0] == state:
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

class LimitedDepthSearchProblem(PositionSearchProblem):
    def __init__(self, gameState, agentIndex):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.pacmanPositions = set(gameState.getPacmanPositions())

        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.getWalls()
        self.food = gameState.getFood()
        self.startState = gameState.getPacmanPosition(agentIndex)
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE

    def isGoalState(self, state):
        x,y = state
        return self.food[x][y] == True
    ## improvement 7 food optimal path




















































