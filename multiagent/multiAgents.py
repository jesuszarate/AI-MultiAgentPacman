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

        
        "Add more of your code here if you want to"

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
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        ma = 1000000
        min = -1000000
        
        # Don't want pacman to stop at all
        if action == Directions.STOP:
            return min

        for ghost in newGhostStates:
            gpos = ghostState.getPosition()           
            
            if gpos == newPos:
                # If the ghost is scared eat it
                if ghostState.scaredTimer > 0:
                    return ma
                 # Otherwise avoid at all cost!
                else:
                    return min
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
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

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
        "*** YOUR CODE HERE ***"

        '''
        gameState.getLegalActions(agentIndex):                                                                                                                                   
            Returns a list of legal actions for an agent                                                                                                                           
            agentIndex=0 means Pacman, ghosts are >= 1                                                                                                                             
                                                                                                                                                                                   
          gameState.generateSuccessor(agentIndex, action):                                                                                                                         
            Returns the successor game state after an agent takes an action                                                                                                        
                                                                                                                                                                                   
          gameState.getNumAgents():                                                                                                                                                
            Returns the total number of agents in the game
        '''
        pacActions = gameState.getLegalActions(0) # Pacman's legal actions
        numAgents = gameState.getNumAgents()
        successor = gameState.generateSuccessor(0, pacActions[0]) #Pacman's first legal action
        depth = self.depth

        import pdb; pdb.set_trace()
        util.raiseNotDefined()
        '''
    def value(self, s):
        if s is a max node:
            return maxValue(s)  
        if s is an exp node:
            return expValue(s)
        if s is a terminal node:
            return evaluation(s)

    def maxValue(self, successor):
        values = []
        for s in successors(successor):
            values.append(value(s))
        return max(values)

    def expValue(self, s):
        values = 0
        sccr = successors(s)
        for s in sccr:
            p = probability(len(sccr))
            v += p * value(s)

        return v
        '''
    def probability(length):
        return 1/length

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

