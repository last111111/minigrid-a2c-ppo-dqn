# Training Progress Display Changes

## Summary
Modified the PPO training pipeline to display real-time success rate tracking during training.

## Changes Made

### 1. Modified `torch_ac/algos/base.py`
- Added `self.log_success` list to track episode success (Line 104)
- Track success for each completed episode: success = 1 if episode_return > 0, else 0 (Line 174)
- Calculate rolling 100-episode success rate in logs (Lines 230-242)
- Added `success_rate` and `success_count` to log output
- Keep only last 100 episodes in memory for success calculation (Line 249)

### 2. Modified `scripts/train.py`
- Extract success rate and count from logs (Lines 230-231)
- Add success_rate and success_count to CSV headers and data (Lines 232-233)
- Updated console output format to include success rate display (Lines 235-237)
- Format: `SR {:.1%} ({}/{})` shows success rate percentage and fraction (e.g., "SR 65.0% (65/100)")

## How It Works

1. **Success Definition**: An episode is considered successful if its total return > 0
   - This works for sparse reward environments where goal-reaching gives positive reward
   - For environments without intermediate rewards, final reward indicates success

2. **Rolling Window**: The success rate is calculated over the last 100 completed episodes
   - Initially shows success rate over fewer episodes until 100 are collected
   - `success_count` indicates how many episodes are in the current window

3. **Display Format**:
   ```
   U 10 | E 160 | F 020480 | FPS 2048 | D 42 | rR:μσmM 0.12 0.32 0.00 1.00 |
   F:μσmM 128.0 0.0 128 128 | H 1.234 | V 0.123 | pL 0.012 | vL 0.034 | ∇ 0.567 |
   SR 65.0% (65/100)
   ```
   - `SR 65.0%` = 65% success rate
   - `(65/100)` = 65 successful episodes out of 100 total

## Testing

Run the original training script as before:
```bash
bash train_ppo.sh
```

Or test with fewer frames:
```bash
bash test_train.sh
```

The training logs will now show real-time success rate updates at each logging interval.
