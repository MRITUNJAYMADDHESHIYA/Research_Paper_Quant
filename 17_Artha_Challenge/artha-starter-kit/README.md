# Artha — starter kit

Everything you need to write, test and submit a bot.

```
starter.py       the file to edit
RULES.md         every rule that decides your score
examples/        three bots, from naive to adaptive
```

## Setup

```bash
pip install artha-1.0.0-py3-none-any.whl     # the wheel from the brief page
```

That installs the engine, the six house bots you will be scored against, the
scorer, and 400 training sessions. Everything below runs offline.

## Your first score

```bash
artha run starter.py
```

Two hundred sessions, roughly fifteen seconds, and a full breakdown of where
your cost went. Iterate here. You have twenty submissions a day and there is no
reason to spend them finding out things you could have found out locally.

```bash
artha benchmark
```

Scores the six house bots the same way you are scored, so you can see what a
respectable number looks like before you go chasing one.

## Submitting

```bash
export ARTHA_TOKEN=your-team-token
artha submit starter.py
```

## Getting at the data

The 400 training sessions are yours to pull apart:

```python
from artha.paths import load

for session in load("train"):
    session.volume["ASHVAM"]     # 320 bars of traded volume
    session.base["ASHVAM"]       # 320 midpoints, before anyone trades
    session.mandate              # signed quantity per instrument
    session.shock_at             # bar index, or None
```

What you find in there is up to you. It is a real part of the problem.


"""
Design goals
------------
- No historical volume profile.
- No training-set lookup at runtime.
- No guessed shock bar.
- No fixed TWAP/VWAP blend.
- No manually chosen shock threshold.
- Uses only information available in `state`.
- Uses the competition's 25% participation cap as the only market-rule constant.

Core idea
---------
At every bar the bot recomputes:

1. Required completion rate:
       remaining / bars_left

2. Online liquidity estimate:
       robust combination of the full observed volume history and
       a recent window whose length grows automatically as sqrt(n).

3. Volume surprise:
       latest observed volume / expected volume

4. Regime adjustment:
       recent volume and recent volatility relative to their own
       historical baselines. This is continuous; there is no hard
       "shock / no shock" threshold.

5. Feasibility pressure:
       remaining quantity / estimated remaining executable capacity

6. Multi-leg synchronization:
       legs that are ahead are slowed; lagging legs receive more weight.

The engine still performs the actual 25% current-bar clipping.
"""
