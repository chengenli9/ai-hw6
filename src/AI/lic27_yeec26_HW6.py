# HW6 TD Learning
# Authors: Chengen, Chris

import random
import sys
sys.path.append("..")  #so other modules can be found in parent dir
from Player import *
from Constants import *
from Construction import CONSTR_STATS
from Ant import UNIT_STATS
from Move import Move
from GameState import *
from AIPlayerUtils import *
import os
import pickle


DISCOUNT_FACTOR = 0.9
LEARNING_RATE = 0.1
EXPLORATION_RATE = 0.1
AI_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_FILE = os.path.join(AI_DIR, "./lic27_yeec26_weights.txt")

#rewards
REWARD_WIN = 1.0
REWARD_LOSS = -1.0
REWARD_STEP = -0.01

## converts a game state into a category
# returns a tuple
def categorizeState(state, myId, enemyId):
    myId = state.whoseTurn
    enemyId = 1 - myId
    myInv = state.inventories[myId]
    enemyInv = state.inventories[enemyId]

    myAnthill = myInv.getAnthill()
    myTunnel = myInv.getTunnels()[0] if myInv.getTunnels() else None
    enemyAnthill = enemyInv.getAnthill()

    # count my ants
    myWorkers = sum(1 for ant in myInv.ants if ant.type == WORKER)
    mySoldiers = sum(1 for ant in myInv.ants if ant.type == SOLDIER)
    myRangeSoldiers = sum(1 for ant in myInv.ants if ant.type == R_SOLDIER)
    myDrones = sum(1 for ant in myInv.ants if ant.type == DRONE)

    # count enemy ants
    enemyWorkers = sum(1 for ant in enemyInv.ants if ant.type == WORKER)
    enemySoldiers = sum(1 for ant in enemyInv.ants if ant.type == SOLDIER or ant.type == R_SOLDIER or ant.type == DRONE)

    # Food amounts
    myFood = min(myInv.foodCount, 11)  # Cap at win condition
    enemyFood = min(enemyInv.foodCount, 11)

    # Queen Health
    myQueen = myInv.getQueen()
    enemyQueen = enemyInv.getQueen()
    myQueenHealth = myQueen.health // 2 if myQueen else 0  # Bucket by 2s
    enemyQueenHealth = enemyQueen.health // 2 if enemyQueen else 0

    # Worker with food status
    workersWithFood = sum(1 for ant in myInv.ants 
                           if ant.type == WORKER and ant.carrying)
    
    # Distance features 
    # Find closest worker to food
    workers = [ant for ant in myInv.ants if ant.type == WORKER]
    foodSources = getConstrList(state, None, (FOOD,))
    
    minWorkerFoodDist = 10
    if workers and foodSources:
        for worker in workers:
            for food in foodSources:
                dist = abs(worker.coords[0] - food.coords[0]) + abs(worker.coords[1] - food.coords[1])
                minWorkerFoodDist = min(minWorkerFoodDist, dist)
    minWorkerFoodDist = min(minWorkerFoodDist // 3, 3)  # Bucket distances
    
    # Distance from worker with food to tunnel/anthill (for depositing)
    minWorkerDepositDist = 10
    if workersWithFood > 0 and (myTunnel or myAnthill):
        for worker in workers:
            if worker.carrying:
                # Check distance to tunnel
                if myTunnel:
                    dist = abs(worker.coords[0] - myTunnel.coords[0]) + abs(worker.coords[1] - myTunnel.coords[1])
                    minWorkerDepositDist = min(minWorkerDepositDist, dist)
                # Check distance to anthill
                if myAnthill:
                    dist = abs(worker.coords[0] - myAnthill.coords[0]) + abs(worker.coords[1] - myAnthill.coords[1])
                    minWorkerDepositDist = min(minWorkerDepositDist, dist)
    minWorkerDepositDist = min(minWorkerDepositDist // 3, 3)  # Bucket distances
    
    # Threat level - are enemy ants near our key structures?
    threatLevel = 0
    if myAnthill:
        for enemyAnt in enemyInv.ants:
            dist = abs(enemyAnt.coords[0] - myAnthill.coords[0]) + abs(enemyAnt.coords[1] - myAnthill.coords[1])
            if dist < 5:
                threatLevel += 1
    
    # Enemy anthill threat - how close are our attacking ants?
    enemyAnthillThreat = 0
    if enemyAnthill:
        for myAnt in myInv.ants:
            if myAnt.type in [SOLDIER, R_SOLDIER, DRONE]:  # Only count attacking ants
                dist = abs(myAnt.coords[0] - enemyAnthill.coords[0]) + abs(myAnt.coords[1] - enemyAnthill.coords[1])
                if dist < 5:
                    enemyAnthillThreat += 1

    features = (
        myFood,
        enemyFood,
        myWorkers,
        mySoldiers,
        myRangeSoldiers,
        myDrones,
        enemyWorkers,
        enemySoldiers,
        myQueenHealth,
        enemyQueenHealth,
        workersWithFood,
        minWorkerFoodDist,
        minWorkerDepositDist,
        min(threatLevel, 3),  # Cap threat level
        min(enemyAnthillThreat, 3)  # Offensive pressure
    )
    
    return features


##
#AIPlayer
#Description: The responsbility of this class is to interact with the game by
#deciding a valid move based on a given game state. This class has methods that
#will be implemented by students in Dr. Nuxoll's AI course.
#
#Variables:
#   playerId - The id of the player.
##
class AIPlayer(Player):
  

    #__init__
    #Description: Creates a new Player
    #
    #Parameters:
    #   inputPlayerId - The id to give the new player (int)
    #   cpy           - whether the player is a copy (when playing itself)
    ##
    def __init__(self, inputPlayerId):
        super(AIPlayer,self).__init__(inputPlayerId, "TD bot")
        self.stateUtilities = {}
        self.stateHistory = []
        self.loadWeights()
        self.gamesPlayed = 0


    # # load weights function
    def loadWeights(self):
        print(f"Looking for weights file at: {os.path.abspath(WEIGHTS_FILE)}")

        if os.path.exists(WEIGHTS_FILE):
            try:
                self.stateUtilities = {}
                with open(WEIGHTS_FILE, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if "|" not in line:
                            continue
                        state, utility = line.split("|", 1)
                        self.stateUtilities[state] = float(utility)

                print(f"Loaded {len(self.stateUtilities)} state utilities from {WEIGHTS_FILE}")

            except Exception as e:
                print(f"Error loading weights: {e}")
                self.stateUtilities = {}
        else:
            print("No saved weights found, starting fresh")


    # save weights for training
    def saveWeights(self):
        try:
            with open(WEIGHTS_FILE, 'w') as f:
                for state, utility in self.stateUtilities.items():
                    f.write(f"{state}|{utility}\n")

            print(f"Saved {len(self.stateUtilities)} state utilities to {WEIGHTS_FILE}")

        except Exception as e:
            print(f"Error saving weights: {e}")




    # get utility for a state category
    def getUtility(self, stateCategory):
        if stateCategory not in self.stateUtilities:
            self.stateUtilities[stateCategory] = 0.0
        return self.stateUtilities[stateCategory]
    
    # utility function using TD learning
    def updateUtility(self, stateCategory, reward, nextStateCategory):
        # TD-Learning Equation: 
        #   U(s) = U(s) + LEARNING_RATE * (Reward(s) + DISCOUNT_FACTOR * U(s') - U(s))
        currentUtility = self.getUtility(stateCategory)
        nextUtility = self.getUtility(nextStateCategory)

        # update TD
        newUtility = currentUtility + LEARNING_RATE * (reward + DISCOUNT_FACTOR * nextUtility - currentUtility)

        self.stateUtilities[stateCategory] = newUtility



    ##
    #getPlacement
    #
    #Description: called during setup phase for each Construction that
    #   must be placed by the player.  These items are: 1 Anthill on
    #   the player's side; 1 tunnel on player's side; 9 grass on the
    #   player's side; and 2 food on the enemy's side.
    #
    #Parameters:
    #   construction - the Construction to be placed.
    #   currentState - the state of the game at this point in time.
    #
    #Return: The coordinates of where the construction is to be placed
    ##
    def getPlacement(self, currentState):
        numToPlace = 0
        #implemented by students to return their next move
        if currentState.phase == SETUP_PHASE_1:    #stuff on my side
            numToPlace = 11
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    #Choose any x location
                    x = random.randint(0, 9)
                    #Choose any y location on your side of the board
                    y = random.randint(0, 3)
                    #Set the move if this space is empty
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        #Just need to make the space non-empty. So I threw whatever I felt like in there.
                        currentState.board[x][y].constr == True
                moves.append(move)
            return moves
        elif currentState.phase == SETUP_PHASE_2:   #stuff on foe's side
            numToPlace = 2
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    #Choose any x location
                    x = random.randint(0, 9)
                    #Choose any y location on enemy side of the board
                    y = random.randint(6, 9)
                    #Set the move if this space is empty
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        #Just need to make the space non-empty. So I threw whatever I felt like in there.
                        currentState.board[x][y].constr == True
                moves.append(move)
            return moves
        else:
            return [(0, 0)]
    
    ##
    #getMove
    #Description: Gets the next move from the Player.
    #
    #Parameters:
    #   currentState - The state of the current game waiting for the player's move (GameState)
    #
    #Return: The Move to be made
    ##
    def getMove(self, currentState):
        myId = currentState.whoseTurn
        enemyId = 1 - myId
        currentCategory = categorizeState(currentState, myId, enemyId)

        # DEBUG - Print first 10 moves only
        if len(self.stateHistory) < 10:
            print(f"Move {len(self.stateHistory)}: {currentCategory}")
            print(f"  Unique states so far: {len(self.stateUtilities)}")

        # update previous state if exits
        if len(self.stateHistory) > 0:
            prevState, prevReward = self.stateHistory[-1]
            self.updateUtility(prevState, prevReward, currentCategory)

        # get all legal moves
        moves = listAllLegalMoves(currentState)

        # Filter out building too many ants
        numAnts = len(currentState.inventories[currentState.whoseTurn].ants)
        if numAnts >= 3:
            moves = [m for m in moves if m.moveType != BUILD]

        if random.random() < EXPLORATION_RATE:
            # Explore: random move
            selectedMove = random.choice(moves)
        else:
            # Exploit: choose move leading to highest utility state
            bestMove = None
            bestUtility = float('-inf')
            
            for move in moves:
                # Predict next state after this move
                nextState = getNextState(currentState, move)
                nextCategory = categorizeState(nextState, myId, enemyId)
                utility = self.getUtility(nextCategory)
                
                # Track best move
                if utility > bestUtility:
                    bestUtility = utility
                    bestMove = move
                elif utility == bestUtility and random.random() < 0.5:
                    # Random tie-breaking
                    bestMove = move
            
            selectedMove = bestMove if bestMove else random.choice(moves)

        # Record state for TD learning
        self.stateHistory.append((currentCategory, REWARD_STEP))

        return selectedMove
    

    
    ##
    #getAttack
    #Description: Gets the attack to be made from the Player
    #
    #Parameters:
    #   currentState - A clone of the current state (GameState)
    #   attackingAnt - The ant currently making the attack (Ant)
    #   enemyLocation - The Locations of the Enemies that can be attacked (Location[])
    ##
    def getAttack(self, currentState, attackingAnt, enemyLocations):
        #Attack a random enemy.
        return enemyLocations[random.randint(0, len(enemyLocations) - 1)]

    ##
    #registerWin
    #
    # Perform TD updates and save weights
    #
    def registerWin(self, hasWon):
        
        finalReward = REWARD_WIN if hasWon else REWARD_LOSS

        if len(self.stateHistory) > 0:
            # final state gets terminal reward
            finalState = self.stateHistory[-1][0]
            self.stateUtilities[finalState] = finalReward

            # update all previous states
            for i in range(len(self.stateHistory) - 2, -1, -1):
                currentState, stepReward = self.stateHistory[i]
                nextState = self.stateHistory[i + 1][0]
                self.updateUtility(currentState, stepReward, nextState)
        
        # Clear history for next game
        self.stateHistory = []
        
        # Save weights after each game
        self.gamesPlayed += 1
        self.saveWeights()
        
        print(f"Game {self.gamesPlayed} complete. {'Won' if hasWon else 'Lost'}. States learned: {len(self.stateUtilities)}")