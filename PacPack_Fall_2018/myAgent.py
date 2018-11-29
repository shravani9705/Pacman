# myAgentP3.py
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
# This file was based on the starter code for student bots, and refined 
# by Mesut (Xiaocheng) Yang


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
from util import nearestPoint
# import math
#########
# Agent #
#########
class MyAgent(CaptureAgent):

  def registerInitialState(self, gameState):
    self.deathmap = getDeathMap(gameState)
    print("Death Map:")
    print(self.deathmap)
    CaptureAgent.registerInitialState(self, gameState)
    self.depth = 4
    self.start = gameState.getAgentPosition(self.index)
    self.weights = [5, 3, 0, 0]   
    self.punishDeath = 300

  def chooseAction(self, gameState):

    maxDepth = self.depth
    n = gameState.getNumAgents()
    if gameState.getAgentPosition(self.index) in [(1,1), (1,2)]:
      self.punishDeath -= 100
    def minimaxTree_node(punishDeath, gameState, k, maxDepth, parrentNode):
      depth = k//n+1
      ghosts = gameState.getGhostTeamIndices()
      if gameState.isOver() or depth > self.depth:
        return self.evaluationFunction(gameState, punishDeath)

      agentIndex = k%n
      actionList = gameState.getLegalActions(agentIndex)

      if agentIndex == self.index: #pacman
        actionList = actionsWithoutStop(actionList)
        actionList = actionsWithoutReverse(actionList, gameState, self.index)
        if gameState.getAgentPosition(self.index) in [(1,1), (1,2)]:
          # print('Die!!!!!!!!!!!!!!!!!AgentPosition:')
          punishDeath -= 100
        if gameState.getAgentPosition(self.index) not in self.deathmap:
          actionList = actionsWithoutDeath(self.distancer, self.deathmap, actionList, gameState, self.index)

        maxscore = -99999
        for action in actionList:
          nextState = gameState.generateSuccessor(agentIndex, action)
          #print("pacman action: ", action, "pacman position: ", gameState.getPacmanPosition())
          thisActionTreeNode = [[action], []]
          score = minimaxTree_node(punishDeath, nextState, k+1, maxDepth, thisActionTreeNode)
          maxscore = max(score, maxscore)
          thisActionTreeNode[0].append(score)   #[[action, score], []]
          parrentNode[1].append(thisActionTreeNode)
        return maxscore
      elif agentIndex in ghosts: #ghost
        sumScore = 0
        for action in actionList:
          nextState = gameState.generateSuccessor(agentIndex, action)
          #print("ghost index: ", agentIndex, "ghost position: ", gameState.getGhostPosition(agentIndex))
          thisActionTreeNode = [[action], []]
          score = minimaxTree_node(punishDeath, nextState, k+1, maxDepth, thisActionTreeNode)
          sumScore += score
          thisActionTreeNode[0].append(score)   #[[action, score], []]
          parrentNode[1].append(thisActionTreeNode)
        chanceScore = sumScore / len(actionList)
        return chanceScore
      else:
        # broadcast = self.receivedBroadcast
        return minimaxTree_node(punishDeath, gameState, k+1, maxDepth, parrentNode)

    def findPacmanPath(treeNode, maxDepth, k, actions):
      ghosts = gameState.getGhostTeamIndices()
      goDeep = k//n
      if goDeep > maxDepth:
        return
      if not treeNode[1]: return
      if k%n == self.index:
        maxScore = -99999
        for i in range(len(treeNode[1])):
          if treeNode[1][i][0][1] > maxScore:
              maxScore = treeNode[1][i][0][1]   #[1]: child node list, [i]: ith child node, [0]: child node action and score, [1]: child node score
              action = treeNode[1][i][0][0]
              index = i
        actions.append(action)
        findPacmanPath(treeNode[1][index], maxDepth, k+1, actions)
      elif k%n in ghosts:
        minScore = 99999
        for i in range(len(treeNode[1])):
          if treeNode[1][i][0][1] < minScore:
            minScore = treeNode[1][i][0][1]
            index = i
        findPacmanPath(treeNode[1][index], maxDepth, k+1, actions)
      else:
        findPacmanPath(treeNode, maxDepth, k+1, actions)

    tree = [["first"], []]
    finalscore = minimaxTree_node(self.punishDeath, gameState, 0, maxDepth, tree)

    # print(maxDepth)
    actions = []
    findPacmanPath(tree, maxDepth, 0, actions)
    # actions = gameState.getLegalActions(self.index)
    print('###############STATE################')
    print('current position: {0}'.format(gameState.getAgentPosition(self.index)))
    if gameState.getAgentPosition(self.index) in self.deathmap:
      print('Now in the deathMap')
    print("finalscore: ", finalscore)
    print('actionList: {0}'.format(actions))
    print('####################################\n')

    return actions[0]

  def evaluationFunction(self, state, punishDeath):
    foods = state.getFood().asList()
    ghosts = [state.getAgentPosition(ghost) for ghost in state.getGhostTeamIndices()]
    friends = [state.getAgentPosition(pacman) for pacman in state.getPacmanTeamIndices() if pacman != self.index]

    pacman = state.getAgentPosition(self.index)
    # if len(foods)>0 
    #   minDis = 99999
    #   for food in foods:
    #     if self.distancer.getDistance(pacman, food) < minDis:
          
    # else:
    #   closestFood = 1.0
    closestFood = min(self.distancer.getDistance(pacman, food) for food in foods) + 2.0 \
        if len(foods) > 0 else 1.0
    closestGhost = min(self.distancer.getDistance(pacman, ghost) for ghost in ghosts) + 1.0 \
        if len(ghosts) > 0 else 1.0
    closestFriend = min(self.distancer.getDistance(pacman, friend) for friend in friends) + 1.0 \
        if len(friends) > 0 else 1.0

    closestFoodReward = 1.0 / closestFood
    closestGhostPenalty = 1.0 / (closestGhost ** 2) if closestGhost < 20 else 0
    closestFriendPenalty = 1.0 / (closestFriend ** 2) if closestFriend < 5 else 0

    numFood = len(foods)

    features = [-numFood, closestFoodReward, closestGhostPenalty, closestFriendPenalty]

    value = sum(feature * weight for feature, weight in zip(features, self.weights)) + punishDeath
    return value

def getDeathMap(gameState):
    Death = 1
    DeathMap = {}
    iter = 0
    while Death:
        iter += 1
        # print(iter)
        # if (1,1) in DeathMap.keys():
        #     print(DeathMap[(1,1)])
        Death = 0
        walls = gameState.getWalls()
        width = walls.width
        height = walls.height
        for i in range(1, width-1):
            for j in range(1, height-1):
                if (not gameState.hasWall(i,j)) and (not ((i,j) in DeathMap.keys())):
                    numWalls = 0
                    dectList = [(i-1,j), (i,j-1), (i,j+1), (i+1,j)]
                    for pos in dectList:
                        if gameState.hasWall(pos[0], pos[1]) or (pos in DeathMap.keys()):
                            numWalls += 1
                    if numWalls > 2:
                        Death = 1
                        deathNum = 1
                        for pos in dectList:
                            if pos in DeathMap.keys():
                                deathNum += DeathMap[pos]
                        DeathMap[(i,j)] = deathNum
    
    return DeathMap

def actionsWithoutDeath(distancer, deathMap, actionList, gameState, agentIndex):
    actionSet = set(actionList)
    removeSet = set()
    # print('*************************************')
    # print('actionlist = {0}'.format(actionList))
    for action in actionList:
        nextState = gameState.generateSuccessor(agentIndex, action)
        nextPosition = nextState.getAgentPosition(agentIndex) 
        # print('for action {0}, next position is {1}'.format(action, nextPosition))
        ghosts = nextState.getGhostTeamIndices()
        ## Assume only one ghost
        ghostPos = nextState.getAgentPosition(ghosts[0])
        if nextPosition in deathMap.keys():
            deathDis = deathMap[nextPosition]
            ghostDis = distancer.getDistance(nextPosition, ghostPos)
            # print('Action{0} would move to {1} deathMap, Ghost Dis={2}, deathDis*2 = {3}'.format(action, nextPosition, ghostDis, deathDis*2))
            if ghostDis < deathDis*2+1:
                removeSet.add(action)

    actionList = list(actionSet - removeSet)
    # if len(actionSet)>len(actionList):
    #     print('removeSet: {}'.format(removeSet))
    #     print('actionList: {}'.format(actionList))
    # print('*************************************')
    return actionList

def actionsWithoutStop(legalActions):
  """
  Filters actions by removing the STOP action
  """
  legalActions = list(legalActions)
  if Directions.STOP in legalActions:
    legalActions.remove(Directions.STOP)
  return legalActions

def actionsWithoutReverse(legalActions, gameState, agentIndex):
  """
  Filters actions by removing REVERSE, i.e. the opposite action to the previous one
  """
  legalActions = list(legalActions)
  reverse = Directions.REVERSE[gameState.getAgentState(agentIndex).configuration.direction]
  if len (legalActions) > 1 and reverse in legalActions:
    legalActions.remove(reverse)
  return legalActions


        




