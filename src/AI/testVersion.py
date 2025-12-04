import random
import sys
import os
sys.path.append("..")
from Player import *
from Constants import *
from Construction import CONSTR_STATS
from Ant import UNIT_STATS
from Move import Move
from GameState import *
from AIPlayerUtils import *

##
# AIPlayer - TD Learning Agent
#
# Implements Temporal Difference (TD) Learning with:
# - State categorization to reduce state space
# - Eligibility traces for faster learning
# - Epsilon-greedy exploration/exploitation
# - Persistence of learned utilities across games
##
class AIPlayer(Player):

    def __init__(self, inputPlayerId):
        super(AIPlayer, self).__init__(inputPlayerId, "TD Learning Agent")
        
        # TD Learning parameters
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 0.15  # Exploration rate (15% explore, 85% exploit)
        
        # State utilities dictionary - stores V(s) for each state category
        self.utilities = {}
        
        # Eligibility traces for faster learning (TD-lambda)
        self.traces = {}
        self.trace_decay = 0.9  # Lambda parameter
        
        # Game tracking
        self.game_history = []  # List of state categories for current game
        self.prev_category = None  # Previous state category for TD updates
        
        # Statistics for monitoring
        self.visit_counts = {}  # How often each state is visited
        self.games_played = 0
        self.wins = 0
        
        # File for saving/loading learned utilities (in same directory as this script)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.weights_file = os.path.join(script_dir, "lic27_yeec26_weights.txt")
        
        # Load existing utilities if available
        self.load_utilities()
    
    def categorize_state(self, state):
        """
        Create a state category to reduce the enormous state space.
        
        This function abstracts away irrelevant details while preserving
        important tactical information. The goal is ~10,000-100,000 categories
        with good revisit rates.
        
        Returns a hashable tuple representing the state category.
        """
        me = state.whoseTurn
        enemy = 1 - me
        
        my_inv = state.inventories[me]
        enemy_inv = state.inventories[enemy]
        
        # Basic inventory stats
        my_food = my_inv.foodCount
        enemy_food = enemy_inv.foodCount
        
        # Count ant types
        my_workers = len([a for a in my_inv.ants if a.type == WORKER])
        my_soldiers = len([a for a in my_inv.ants if a.type in [SOLDIER, DRONE, R_SOLDIER]])
        enemy_workers = len([a for a in enemy_inv.ants if a.type == WORKER])
        enemy_soldiers = len([a for a in enemy_inv.ants if a.type in [SOLDIER, DRONE, R_SOLDIER]])
        
        # Check if we have a queen
        my_queen_alive = my_inv.getQueen() is not None
        enemy_queen_alive = enemy_inv.getQueen() is not None
        
        # Get critical locations
        my_tunnel = my_inv.getTunnels()[0] if my_inv.getTunnels() else None
        my_anthill = my_inv.getAnthill()
        enemy_anthill = enemy_inv.getAnthill()
        
        # Analyze worker positions relative to key locations
        workers_carrying = 0
        workers_near_food = 0
        workers_near_deposit = 0
        
        foods = getConstrList(state, None, (FOOD,))
        
        for ant in my_inv.ants:
            if ant.type == WORKER:
                if ant.carrying:
                    workers_carrying += 1
                
                # Distance to nearest food
                if foods:
                    min_food_dist = min([abs(ant.coords[0] - f.coords[0]) + 
                                        abs(ant.coords[1] - f.coords[1]) for f in foods])
                    if min_food_dist <= 4:
                        workers_near_food += 1
                
                # Distance to deposit (tunnel or anthill)
                if my_tunnel:
                    tunnel_dist = abs(ant.coords[0] - my_tunnel.coords[0]) + abs(ant.coords[1] - my_tunnel.coords[1])
                else:
                    tunnel_dist = 100
                    
                if my_anthill:
                    anthill_dist = abs(ant.coords[0] - my_anthill.coords[0]) + abs(ant.coords[1] - my_anthill.coords[1])
                else:
                    anthill_dist = 100
                
                min_deposit_dist = min(tunnel_dist, anthill_dist)
                if min_deposit_dist <= 4:
                    workers_near_deposit += 1
        
        # Analyze soldier positions
        soldiers_near_enemy_ants = 0
        soldiers_on_enemy_side = 0
        
        for ant in my_inv.ants:
            if ant.type in [SOLDIER, DRONE, R_SOLDIER]:
                # Check if near enemy ants
                for enemy_ant in enemy_inv.ants:
                    dist = abs(ant.coords[0] - enemy_ant.coords[0]) + abs(ant.coords[1] - enemy_ant.coords[1])
                    if dist <= 5:
                        soldiers_near_enemy_ants += 1
                        break
                
                # Check if on enemy side
                if ant.coords[1] >= 5:
                    soldiers_on_enemy_side += 1
        
        # Bin continuous values to create discrete categories
        # This greatly reduces the state space
        food_bin = min(my_food // 2, 5)  # 0, 2, 4, 6, 8, 10+
        enemy_food_bin = min(enemy_food // 2, 5)
        worker_bin = min(my_workers, 4)  # 0, 1, 2, 3, 4+
        soldier_bin = min(my_soldiers, 3)  # 0, 1, 2, 3+
        enemy_worker_bin = min(enemy_workers, 3)
        enemy_soldier_bin = min(enemy_soldiers, 3)
        
        # Create state category tuple
        category = (
            food_bin,
            enemy_food_bin,
            worker_bin,
            soldier_bin,
            enemy_worker_bin,
            enemy_soldier_bin,
            min(workers_carrying, 2),
            min(workers_near_food, 2),
            min(workers_near_deposit, 2),
            min(soldiers_near_enemy_ants, 2),
            min(soldiers_on_enemy_side, 2),
            my_queen_alive,
            enemy_queen_alive
        )
        
        # Track visit counts
        if category not in self.visit_counts:
            self.visit_counts[category] = 0
        self.visit_counts[category] += 1
        
        return category
    
    def get_utility(self, category):
        """Get utility value for a state category. Default to 0.0 for unseen states."""
        return self.utilities.get(category, 0.0)
    
    def set_utility(self, category, value):
        """Set utility value for a state category."""
        self.utilities[category] = value
    
    def td_update(self, current_state):
        """
        Perform TD(lambda) learning update.
        
        TD Learning equation:
        V(s) = V(s) + alpha * (reward + gamma * V(s') - V(s))
        
        With eligibility traces for TD(lambda).
        """
        if self.prev_category is None:
            return
        
        current_category = self.categorize_state(current_state)
        
        # Immediate reward (small negative for each step)
        reward = -0.01
        
        # Check for terminal state conditions
        me = current_state.whoseTurn
        enemy = 1 - me
        my_inv = current_state.inventories[me]
        enemy_inv = current_state.inventories[enemy]
        
        if my_inv.foodCount >= FOOD_GOAL or enemy_inv.getQueen() is None:
            reward = 1.0  # Win
        elif enemy_inv.foodCount >= FOOD_GOAL or my_inv.getQueen() is None:
            reward = -1.0  # Loss
        
        # TD error (temporal difference)
        prev_utility = self.get_utility(self.prev_category)
        current_utility = self.get_utility(current_category)
        td_error = reward + self.gamma * current_utility - prev_utility
        
        # Update eligibility trace for previous state
        self.traces[self.prev_category] = self.traces.get(self.prev_category, 0.0) + 1.0
        
        # Update all states with eligibility traces
        for category in list(self.traces.keys()):
            trace = self.traces[category]
            old_util = self.get_utility(category)
            new_util = old_util + self.alpha * td_error * trace
            self.set_utility(category, new_util)
            
            # Decay trace for next step
            self.traces[category] = trace * self.gamma * self.trace_decay
            
            # Remove negligible traces to keep memory efficient
            if self.traces[category] < 0.001:
                del self.traces[category]
        
        # Store current state for next update
        self.prev_category = current_category
        self.game_history.append(current_category)
    
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
    
    def getMove(self, currentState):
        """
        Select next move using TD learning.
        
        Strategy:
        1. Update from previous state
        2. Use epsilon-greedy to balance exploration and exploitation
        3. Exploit: choose move leading to highest utility state
        4. Explore: choose random move
        """
        # Perform TD update from previous state
        self.td_update(currentState)
        
        # Get all legal moves
        moves = listAllLegalMoves(currentState)
        
        # Filter out building moves if we already have 3+ ants (resource management)
        numAnts = len(currentState.inventories[currentState.whoseTurn].ants)
        movable_moves = [m for m in moves if m.moveType != BUILD or numAnts < 3]
        
        if movable_moves:
            moves = movable_moves
        
        if not moves:
            # Fallback - should rarely happen
            return moves[0] if moves else Move(END, [], None)
        
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            # Explore: random move
            selectedMove = random.choice(moves)
        else:
            # Exploit: choose move with highest utility successor state
            best_move = None
            best_utility = float('-inf')
            best_moves = []
            
            for move in moves:
                # Predict state after this move
                next_state = getNextState(currentState, move)
                next_category = self.categorize_state(next_state)
                utility = self.get_utility(next_category)
                
                if utility > best_utility:
                    best_utility = utility
                    best_moves = [move]
                elif utility == best_utility:
                    best_moves.append(move)
            
            # Break ties randomly
            selectedMove = random.choice(best_moves) if best_moves else random.choice(moves)
        
        # Store state for next update
        self.prev_category = self.categorize_state(currentState)
        self.game_history.append(self.prev_category)
        
        return selectedMove
    
    def getAttack(self, currentState, attackingAnt, enemyLocations):
        """Attack the closest enemy to apply pressure."""
        if not enemyLocations:
            return None
        
        closest = min(enemyLocations,
                     key=lambda loc: abs(attackingAnt.coords[0] - loc[0]) + 
                                    abs(attackingAnt.coords[1] - loc[1]))
        return closest
    
    def registerWin(self, hasWon):
        """
        Called at end of game.
        
        Performs final TD update and saves learned utilities.
        This is critical for TD learning to work properly!
        """
        # Perform final TD update with terminal reward
        if self.prev_category is not None:
            # Terminal state update
            reward = 1.0 if hasWon else -1.0
            
            # For terminal state, bootstrap is 0
            prev_utility = self.get_utility(self.prev_category)
            td_error = reward - prev_utility
            
            # Update all traced states one final time
            for category, trace in self.traces.items():
                old_util = self.get_utility(category)
                new_util = old_util + self.alpha * td_error * trace
                self.set_utility(category, new_util)
        
        # Statistics
        self.games_played += 1
        if hasWon:
            self.wins += 1
        win_rate = (self.wins / self.games_played * 100) if self.games_played > 0 else 0
        
        # Print learning statistics
        print(f"Game {self.games_played} - Win: {hasWon} - Win Rate: {win_rate:.1f}% - States Visited: {len(self.visit_counts)} - Utilities Learned: {len(self.utilities)}")
        
        # Reset eligibility traces and game history
        self.traces = {}
        self.game_history = []
        self.prev_category = None
        
        # Save utilities after each game
        self.save_utilities()
    
    def save_utilities(self):
        """Save learned utilities to file for persistence across runs."""
        try:
            with open(self.weights_file, 'w') as f:
                for state_category, utility in self.utilities.items():
                    f.write(f"{state_category}\t{utility}\n")
        except Exception as e:
            print(f"Error saving utilities: {e}")
    
    def load_utilities(self):
        """Load utilities from file if it exists."""
        if os.path.exists(self.weights_file):
            try:
                with open(self.weights_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            parts = line.split('\t')
                            if len(parts) == 2:
                                state_category = eval(parts[0])
                                utility = float(parts[1])
                                self.utilities[state_category] = utility
                print(f"Loaded {len(self.utilities)} learned state utilities from {self.weights_file}")
            except Exception as e:
                print(f"Error loading utilities: {e}")
                self.utilities = {}
        else:
            print(f"No existing utilities file found at {self.weights_file} - starting fresh")
