# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util, sys

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        return legalMoves[chosenIndex]


    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newGhostStates = successorGameState.getGhostStates()

        minInt = -sys.maxint - 1
        maxInt = sys.maxint

        # Don't want pacman to stop at all
        if action == Directions.STOP:
            return minInt

        for ghostState in newGhostStates:
            gpos = ghostState.getPosition()           
            if gpos == newPos:
                # If the ghost is scared eat it
                if ghostState.scaredTimer > 0:
                    return maxInt
                 # Otherwise avoid at all cost!
                else:
                    return minInt

        dist = []            
        for pos in currentGameState.getFood().asList():
            # The closer the food the smaller the manhattan distance is, 
            # so to make it larger than farther food just make it negative
            md = util.manhattanDistance(newPos, pos) * -1            
            dist.append(md)

        return max(dist)


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """
    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        # use minimax to find the immediate action that can result in the
        # best reward. begin by maximizing pacman (agent index = 0)
        bestValue, bestAction = self.minimax(gameState, self.depth, 0)
        return bestAction

    # gets the optimized leaf value and the immediate move that can be made
    # to achieve that reward
    def minimax(self, gameState, depth, maximizingAgentIndex):
        # gets the list of moves that can be made from this state
        actions = gameState.getLegalActions(maximizingAgentIndex)

        # base case - we've recursed as far as we want, or can
        if depth == 0 or len(actions) == 0:
            return self.evaluationFunction(gameState), None

        # depth counts as a pacman action and a response from all ghosts
        # only decrease the depth if we're working with the last ghost
        if maximizingAgentIndex == gameState.getNumAgents() - 1:
            depth = depth - 1

        # get the index for the next agent to recurse on
        nextAgentIndex = (maximizingAgentIndex + 1) % (gameState.getNumAgents())

        # if the current agent is 0 (pacman) we want to maximze the value
        # but if the current agent is a ghost, we want to minimize it
        if maximizingAgentIndex == 0:
            bestValue = -sys.maxint - 1       # smallest integer value
            minormax = lambda x, y: max(x, y) # want to take the max
        else:
            bestValue = sys.maxint            # biggest integer value
            minormax = lambda x, y: min(x, y) # want to take the min

        # we will recurse through all possible actions to find the action
        # that minimizes or maximizes the value
        bestAction = actions[0]               # initialize action variable
        for action in actions:
            # get the child game state for the current agent in a direction
            childState = gameState.generateSuccessor(maximizingAgentIndex, action)

            # we only care about the immediate action to get toward the
            # leaf node. so we ditch the action from the state below
            value, _ = self.minimax(childState, depth, nextAgentIndex)

            # depending on the agent, maximize or minimize the value
            bestValue = minormax(bestValue, value)
            if value == bestValue:
                bestAction = action

        return bestValue, bestAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        # finds the immediate action that can result in the best reward using
        # alpha beta pruning.
        bestValue, bestAction = self.alphabeta(gameState, self.depth, 0)
        return bestAction

    # an alpha beta optimized version of minimax with variable depth and multiple agents
    def alphabeta(self, gameState, depth, maximizingAgentIndex, alpha=(-sys.maxint - 1), beta=(sys.maxint)):
        # gets the list of possible actions from the current state
        actions = gameState.getLegalActions(maximizingAgentIndex)

        # the base case - if we have recursed as far as we can go or as far as
        # we want to go
        if depth == 0 or len(actions) == 0:
            return self.evaluationFunction(gameState), None

        # a depth of 1 means that pacman has moved as well as every other ghost
        # so we only want to update the depth after all agents have moved (when
        # we're currently looking at the last agent)
        if maximizingAgentIndex == gameState.getNumAgents() - 1:
            depth = depth - 1

        # update the index of the next agent. if we're at the last agent roll over
        # to agent with index of 0 (pacman)
        nextAgentIndex = (maximizingAgentIndex + 1) % (gameState.getNumAgents())

        # if we're currently looking at agent 0 (pacman) we want to maximize the
        # reward. so we initialize all of the parameters for maximization. otherwise
        # we're looking at a ghost so we initialize all paramerters for minimization
        if maximizingAgentIndex == 0:
            bestValue = -sys.maxint - 1         # smallest integer value
            minormax = lambda x, y: max(x, y)   # want to take the max
            minOrMaxBest = lambda v: v > beta   # if value is as good as it can get
            getAlphaAndBeta = lambda v: (max(alpha, v), beta)  # update a and b
        else:
            bestValue = sys.maxint              # biggest integer value
            minormax = lambda x, y: min(x, y)   # want to take the min
            minOrMaxBest = lambda v: v < alpha  # if value is as good as it can get
            getAlphaAndBeta = lambda v: (alpha, min(beta, v))  # update a and b

        # recurse through all actions
        bestAction = actions[0]                 # initialize action variable
        for action in actions:
            # get the child game state for the current agent in each possible direction
            childState = gameState.generateSuccessor(maximizingAgentIndex, action)

            # get the alpha beta minimaxed value from the next state for the next
            # agent and minimize or maximize it accordingly
            value, _ = self.alphabeta(childState, depth, nextAgentIndex, alpha, beta)
            bestValue = minormax(bestValue, value)

            # if we've found the best value so far, make sure bestAction matches
            if value == bestValue:
                bestAction = action

            # if we've found the best value we could possibly find, just return
            # these values because there's no point in looking any further
            if minOrMaxBest(bestValue):
                return bestValue, bestAction

            # update alpha and beta, depending on the agent, with the best value so far
            alpha, beta = getAlphaAndBeta(bestValue)

        return bestValue, bestAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        # finds the immediate action that can result in the best reward using
        # the expectimax algorithm
        bestValue, bestAction = self.expectimax(gameState, self.depth, 0)
        return bestAction

    # get the maximized value of the averaged values of each player resulting
    # in the best immediate move for pacman
    def expectimax(self, gameState, depth, maximizingAgentIndex):
        # gets the list of possible actions from the current state
        actions = gameState.getLegalActions(maximizingAgentIndex)

        # the base case - if we have recursed as far as we can go or as far as
        # we want to go
        if depth == 0 or len(actions) == 0:
            return self.evaluationFunction(gameState), None

        # a depth of 1 means that pacman has moved as well as every other ghost
        # so we only want to update the depth after all agents have moved (when
        # we're currently looking at the last agent)
        if maximizingAgentIndex == gameState.getNumAgents() - 1:
            depth = depth - 1

        # update the index of the next agent. if we're at the last agent roll over
        # to agent with index of 0 (pacman)
        nextAgentIndex = (maximizingAgentIndex + 1) % (gameState.getNumAgents())

        # if we're currently looking at agent 0 (pacman) we want to maximize the
        # reward. so we initialize a lambda function to maximize the values. when
        # we're looking at a ghost we want to average all of the values
        if maximizingAgentIndex == 0:
            # want to take the max from a list
            maxoravg = lambda vals: max(vals)
        else:
            # want to take the avg from a list
            maxoravg = lambda vals: (sum(vals) * 1.0) / (len(vals) * 1.0)

        # we recursively find the values of all of the child game states and add the
        # values to a list
        values = []
        for action in actions:
            # get the child game state for the current agent in each possible direction
            childState = gameState.generateSuccessor(maximizingAgentIndex, action)

            # get the value of the child game state (averaged or max depending on agent)
            value, _ = self.expectimax(childState, depth, nextAgentIndex)
            values.append(value)

        # get either the maximum value from all of the child game states (when pacman)
        # or the average value of all of the child game state (when any of the ghosts)
        bestValue = maxoravg(values)
        bestAction = None

        # can only return a definitive action to take on pacman's turn. we can't really
        # return an action for pacman to take when we've average the ghosts moves...
        if maximizingAgentIndex == 0:
            # get the action that corresponds with the maximized value
            i = values.index(bestValue)
            bestAction = actions[i]

        return bestValue, bestAction


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """

    minInt = -sys.maxint - 1
    maxInt = sys.maxint
    max_dist_g = 5

    pacmanPosition = currentGameState.getPacmanPosition()
    foodPositions = currentGameState.getFood().asList()

    newGhostStates = currentGameState.getGhostStates()
    newPos = currentGameState.getPacmanPosition()


    fdist = -float("inf")
    for fpos in foodPositions:
        md = util.manhattanDistance(newPos, fpos)
        fdist = max(fdist, md)

    for ghostState in newGhostStates:
        gpos = ghostState.getPosition()
        gd = util.manhattanDistance(newPos, gpos)
        if gd < max_dist_g:
            return minInt
    
    caplen = len(currentGameState.getCapsules())
    foodlen =  currentGameState.getNumFood()
    score = currentGameState.getScore()
    return -1 * (fdist +  (100*caplen) + (10000 *foodlen) - (100*score))



# Abbreviation
better = betterEvaluationFunction
