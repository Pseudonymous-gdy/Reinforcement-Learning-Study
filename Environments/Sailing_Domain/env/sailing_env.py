'''
SailingEnv Introduction:

State: (x,y,wind_dir,tack), where:
- wind_dir: direction of wind
- tack: the relative direction of sailing compared
    to wind directions.

Actions (7): neighbor moves; action opposite to wind
    is forbidden. Cost is in [1,8.6], depends on
    action vs. wind direction.

API:
- reset(start=None, goal=None) -> state
- step(state, action) -> (next_state, cost, done,
    info)
- valid_cations(state) -> List[int]

Wind/Tack update: delegate to WindModel
'''
