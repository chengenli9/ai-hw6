# TD Learning Agent Implementation - CS 421 HW6

## Overview
This is a complete Temporal Difference (TD) learning agent for the ReAntics environment that properly implements TD(λ) learning with eligibility traces.

## Key Components

### 1. **State Categorization** (Critical for success)
The agent reduces the massive state space using a 13-dimensional category tuple:
```
(food_bin, enemy_food_bin, worker_bin, soldier_bin, 
 enemy_worker_bin, enemy_soldier_bin, 
 workers_carrying, workers_near_food, workers_near_deposit,
 soldiers_near_enemy, soldiers_on_enemy_side,
 my_queen_alive, enemy_queen_alive)
```

**Binning strategy:**
- Food counts: 0, 1, 2, 3, 4, 5+ (binned into 6 categories)
- Ant counts: 0-4 (binned based on type)
- Boolean features: queen alive/dead
- Positional features: near food, near deposit, near enemy ants

**Expected state space:** ~10,000-100,000 states (ensures good revisit rates)

### 2. **TD(λ) Learning with Eligibility Traces**
The agent uses the correct TD learning equation:
```
V(s) = V(s) + alpha * (reward + gamma * V(s') - V(s))
```

With eligibility traces for faster convergence:
```
For each state s:
  trace[s] = trace[s] * gamma * lambda
  V(s) = V(s) + alpha * TD_error * trace[s]
```

**Parameters:**
- `alpha = 0.1` (learning rate) - can be tuned or decayed over time
- `gamma = 0.9` (discount factor) - balances immediate vs future rewards
- `epsilon = 0.15` (exploration rate) - 15% explore, 85% exploit
- `lambda = 0.9` (trace decay) - speeds up learning without excessive variance

### 3. **Reward Function**
```
Terminal states:
  +1.0 if agent wins (food >= 11 OR enemy queen dies)
  -1.0 if agent loses (enemy food >= 11 OR our queen dies)
  
Non-terminal states:
  -0.01 for each step (encourages faster wins)
```

This ensures strong learning signals at game end while avoiding oscillation during play.

### 4. **Action Selection Strategy (Epsilon-Greedy)**
- **Exploit (85% of time):** Choose move leading to highest utility successor state
- **Explore (15% of time):** Choose random move
- **Tie-breaking:** Random among moves with equal utility

The agent uses `getNextState()` to predict future states, then looks up their utility values.

### 5. **Proper TD Update Timing** (THE FIX from previous code)
```python
def getMove():
    td_update(currentState)  # Update from PREVIOUS state
    # ... select move ...
    prev_category = categorize_state(currentState)  # Store for next update

def registerWin(hasWon):
    # FINAL update with terminal reward
    reward = 1.0 if hasWon else -1.0
    td_error = reward - prev_utility  # Bootstrap from 0
    # Apply update to all traced states
    save_utilities()  # CRITICAL: save after each game
```

**Why this matters:**
- Previous code only received rewards during its own turns
- This code receives rewards from the environment (win/loss signals)
- Final `registerWin()` call ensures terminal state bootstrapping happens

### 6. **Persistence Across Runs**
- Saves `utilities` dictionary after each game to `lic27_yeec26_weights.txt`
- Loads saved utilities on startup if file exists
- Format: `state_category\tutility_value` (one per line)
- Allows training to continue across multiple runs

## How TD Learning Works Here

1. **Game starts:** Agent categorizes state and stores in `prev_category`
2. **Each turn:**
   - Call `td_update()` using current state
   - TD error = `reward + gamma * V(next_state) - V(prev_state)`
   - Update all traced states proportionally
   - Decay eligibility traces
   - Select next move using epsilon-greedy
3. **Game ends:** Call `registerWin()`
   - Final update with terminal reward signal
   - Clear eligibility traces
   - Save utilities to file

## Tuning Recommendations

If learning is too slow:
- Increase `alpha` (0.1 → 0.2 or 0.3)
- Decrease `epsilon` (explore less, exploit more)
- Increase `trace_decay` (0.9 → 0.95)

If learning is unstable:
- Decrease `alpha` (0.1 → 0.05)
- Increase `epsilon` (explore more)
- Decrease `gamma` (0.9 → 0.8)

To debug state categorization:
- Check `visit_counts` dictionary
- If heavily imbalanced, adjust binning thresholds
- If too sparse, reduce number of categories

## Testing Checklist

✓ Agent loads/saves utilities correctly
✓ TD updates happen on every move
✓ Terminal rewards applied only at game end
✓ Eligibility traces decay properly
✓ Win rate should increase over 100+ games
✓ Can beat Random agent > 60% after training
✓ Completes games in < 3 minutes

## Files
- `testVersion.py` - Main TD learning agent
- `lic27_yeec26_weights.txt` - Learned utilities (auto-generated)
