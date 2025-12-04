# HW6 TD Learning
# Authors: Chengen, Chris

import random
import sys

import numpy as np

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
WEIGHTS_FILE = os.path.join(AI_DIR, "./v2_lic27_yeec26_weights.txt")

#rewards
REWARD_WIN = 1.0
REWARD_LOSS = -1.0
REWARD_STEP = -0.01

## converts a game state into a category
# returns a tuple
def categorizeState(gameState, myId, enemyId):
    fea = np.array([]).astype(float)
    # Constants
    me = gameState.whoseTurn
    enemy = 1 - me
    myInv = getCurrPlayerInventory(gameState)
    enemyInv = getEnemyInv(me, gameState)
    myWorkers = getAntList(gameState, me, (WORKER,))
    tunnels = myInv.getTunnels()
    anthill = myInv.getAnthill()
    foodList = getConstrList(gameState, None, (FOOD,))
    def on_my_side(coords):
        y = coords[1]
        return y <= 4

    # Enemy ants on my side are threats
    threats = [a for a in getAntList(gameState, enemy, (QUEEN, WORKER, DRONE, SOLDIER, R_SOLDIER)) if on_my_side(a.coords)]
    fea = np.append(fea, min(1.0, np.multiply(len(threats), 0.1)))
    # My attack-capable ants
    defenders = getAntList(gameState, me, (DRONE, SOLDIER, R_SOLDIER))
    fea = np.append(fea, min(1.0, np.multiply(len(defenders), 0.1)))

    # If no threats on my side, defense is perfect
    if not threats:
        # return np.full((7, 1), 1.0, dtype=float)
        fea = np.append(fea, 1.0)
    else:
        fea = np.append(fea, 0.0)
    # If there are threats but no defenders, defense is bad
    if not defenders:
        # return np.full((7, 1), 0.0, dtype=float)
        fea = np.append(fea, 0.0)
    else:
        fea = np.append(fea, 1.0)

    # Encourage defenders to be close to threats
    # 0 distance -> 1.0 score; distance >= maxDist -> 0.1 score
    maxDist = 10.0
    total = 0.0
    if len(threats) != 0 and len(defenders) != 0:
        for t in threats:
            minDist = min(approxDist(d.coords, t.coords) for d in defenders)
            score = 1.0 - min(minDist / maxDist, 10.0)
            total += score

        proximityScore = total / len(threats)
        proximityScore = max(0.0, min(1.0, proximityScore))
        fea = np.append(fea, proximityScore)
    elif len(defenders) == 0:
        fea = np.append(fea, 0.0)
    else:
        fea = np.append(fea, 1.0)

    # return
    utility = 0.0

    # Food Weights - 90% of total utility
    if myInv.foodCount is not None and enemyInv.foodCount is not None:
        foodScore = 0.5
        foodScore += (myInv.foodCount / 11) * 0.5 # This is on a scale of 0 - 1 - good, now multiply by multiplier
        foodScore -= (enemyInv.foodCount / 11) * 0.5
        # print(f"Food Score: {foodScore}")
        utility += foodScore * 0.97
        fea = np.append(fea, min(1.0, max(foodScore, 0.0)))

        # Some help from ChatGPT
        workerScore = 0.0
        # Get my workers
        numWorkers = len(myWorkers)
        # print(f"Workers: {myWorkers}")
        # print(f"Num Workers: {numWorkers}")
        # fea = np.append(fea, np.multiply(len(myWorkers), 0.1))
        fea = np.append(fea, np.multiply(numWorkers, 0.1))

        # If we have no workers, score is 0
        if numWorkers == 0:
            fea = np.append(fea, 0.0)
        else:
            fea = np.append(fea, 1.0)

        # If we have too many workers, aka not good
        if numWorkers > 2:
            utility -= 0.1

        # Avoid division by zero; if no workers, score remains 0
        if numWorkers > 0:
            # Precompute drop sites
            dropSites = []
            if anthill:
                dropSites.append(anthill.coords)
            if tunnels:
                dropSites.extend([t.coords for t in tunnels])

            # Normalization constants keep per-worker contribution in [0,1]
            maxFoodDist = 8.0
            maxDropDist = 8.0

            for i, w in enumerate(myWorkers):
                contrib = 0.0

                if w.carrying:
                    # If at drop site: full contribution
                    if dropSites and any(w.coords == d for d in dropSites):
                        contrib = 1.0
                    else:
                        # Positive baseline for carrying so picking up is attractive
                        if dropSites:
                            closestDrop = min(approxDist(w.coords, d) for d in dropSites)
                            progressToDrop = max(0.0, min(1.0, 1.0 - (closestDrop / maxDropDist)))
                        else:
                            progressToDrop = 0.0
                        # Baseline 0.5 plus progress up to 1.0 max
                        contrib = 0.5 + 0.5 * progressToDrop
                else:
                    # Not carrying: incentivize getting closer to nearest food, but cap at 0.5
                    if foodList:
                        closestFood = min(approxDist(w.coords, f.coords) for f in foodList)
                        towardFood = max(0.0, min(1.0, 1.0 - (closestFood / maxFoodDist)))
                        contrib = 0.5 * towardFood
                    else:
                        contrib = 0.0

                # Clamp and average across workers
                contrib = max(0.0, min(1.0, contrib))
                workerScore += contrib / numWorkers
                # print(f"Worker {i} contrib: {contrib}")

        # print(f"Worker Score: {workerScore}")
        # Ensure workerScore in [0,1]
        workerScore = max(0.0, min(1.0, workerScore))
        fea = np.append(fea, workerScore)
        utility += (workerScore * 0.03)
        utility = min(utility, 1.0)
        fea = np.append(fea, utility)

    return fea


##
#AIPlayer
#Description: The responsibility of this class is to interact with the game by
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
            self.stateUtilities[stateCategory] = np.sum(stateCategory)
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
            # Exploit: choose move leading to the highest utility state
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
