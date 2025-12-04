# Next Steps for Training Your TD Learning Agent

## Step 1: Verify Setup
- Ensure `testVersion.py` is in `src/AI/` directory ✓
- Weights file will be created as `./lic27_yeec26_weights.txt` in working directory
- When testing, delete this file to reset learning

## Step 2: Run Initial Test Games
Start by playing your TD agent against Random or itself to see if it runs:

```bash
# From src directory
python Game.py
# Select: TD Learning Agent vs Random Agent
```

Check the console output after each game:
```
Game 1 - Win: False - Win Rate: 0.0% - States Visited: 247 - Utilities Learned: 185
Game 2 - Win: True - Win Rate: 50.0% - States Visited: 356 - Utilities Learned: 298
```

**What to look for:**
- ✓ States Visited: Should grow to 5,000-50,000 over many games
- ✓ Utilities Learned: Should match or be slightly less than States Visited
- ✓ Game completes in < 3 minutes
- ✓ Win Rate gradually increases

## Step 3: Initial Training (50-100 games)
Run multiple games to establish baseline learning. If win rate is still 0-10% after 50 games, something needs adjustment.

## Step 4: Monitor Learning Progress
Create a simple script to track progress:

```python
# Log game statistics
games_data = []
for game in range(100):
    # Play game, store (game_num, win, states_visited, utilities_learned)
    games_data.append((game, hasWon, len(visit_counts), len(utilities)))
```

Expected progression:
- Games 1-20: 10-20% win rate
- Games 20-50: 30-40% win rate  
- Games 50-100: 40-60% win rate

## Step 5: Tune Learning Parameters
If results are unsatisfactory, try ONE change at a time:

### Agent learns too slowly:
```python
self.alpha = 0.2  # Increase learning rate (was 0.1)
# OR
self.epsilon = 0.05  # Decrease exploration (was 0.15)
# OR
self.trace_decay = 0.95  # Increase trace impact (was 0.9)
```
**Delete weights file and retrain for 50 games to test**

### Agent learns but is unstable:
```python
self.alpha = 0.05  # Decrease learning rate (was 0.1)
self.epsilon = 0.25  # Increase exploration (was 0.15)
# OR
self.gamma = 0.8  # Discount future rewards less (was 0.9)
```

### State space issues:
If visiting same states repeatedly, refine `categorize_state()`:
- Adjust binning thresholds
- Add/remove features
- Check `visit_counts` histogram

## Step 6: Final Validation
Once win rate > 60% vs Random:

1. **Test persistence:**
   - Play 10 games
   - Close program
   - Reopen and play 10 more
   - Should maintain high win rate

2. **Test speed:**
   - Play 10 games consecutively
   - Each should complete in < 3 minutes

3. **Test self-play:**
   - TD Agent vs TD Agent
   - Should always complete without timeout

## Debugging Tips

### If game crashes:
Check the error. Common issues:
- `getNextState()` failure → verify currentState is valid
- File I/O error → check write permissions in working directory
- Memory error → reduce state space granularity

### If learning stalls:
- Print first 10 states visited: `print(list(self.visit_counts.keys())[:10])`
- Check if same states repeated: indicates state abstraction is too coarse
- Verify TD updates happening: add debug print in `td_update()`

### If agent plays poorly despite high win rate:
- Check `epsilon` - might be exploring too much at inference time
- Verify `categorize_state()` is deterministic (same input = same category)
- Check utilities are being loaded correctly

## File Management

**Weights file location:** Same directory as your Game.py runner
```
./lic27_yeec26_weights.txt  (relative to working directory)
```

**To reset learning:**
```bash
# Delete the weights file before running
del lic27_yeec26_weights.txt  # Windows
rm ./lic27_yeec26_weights.txt  # Linux/Mac
```

**Backup good weights:**
```bash
copy lic27_yeec26_weights.txt lic27_yeec26_weights_backup.txt
```

## Expected Final Performance

After 200-300 training games:
- ✓ Beats Random agent 60-70% of the time
- ✓ Completes each game in < 3 minutes
- ✓ Never crashes on self-play
- ✓ Graceful restart from saved weights

## What the Agent Learned

Check what it discovered by examining the weights file:
```bash
# Show utilities for states with food >= 8
grep "(5," ./lic27_yeec26_weights.txt
```

States near winning should have high utility (+0.5 to +1.0).
States with dead queens should have low utility (-0.5 to -1.0).

---

**You now have a proper TD learning agent that's ready to train!** Start with 50 games and monitor the win rate. Good luck! 🚀
