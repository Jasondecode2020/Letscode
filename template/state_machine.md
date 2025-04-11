### 65. Valid Number

```python
class Solution:
    def isNumber(self, s: str) -> bool:
        # -12.34e+1
        s = s.strip()
        state = [
            {'s': 1, 'd': 2, '.': 3},               # 0 start state
            {'d': 2, '.': 3},                       # 1 sign before dot 
            {'d': 2, '.': 4, 'e': 5},               # 2 digit before dot 
            {'d': 4},                               # 3 digit after dot and no digit before dot 
            {'d': 4, 'e': 5},                       # 4 digit after dot and has digit before do t
            {'s': 6, 'd': 7},                       # 5 e
            {'d': 7},                               # 6 sign after dot
            {'d': 7},                               # 7 digit after dot 
        ]
        x = 0
        for c in s:
            if '0' <= c <= '9': t = 'd'
            elif c in '+-': t = 's'
            elif c in 'eE': t = 'e'
            elif c == '.': t = '.'
            else: t = '#'
            if not t in state[x]: return False 
            x = state[x][t]
        return x in (2, 4, 7)
```