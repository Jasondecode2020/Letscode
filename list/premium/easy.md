### 293. Flip Game

```python
class Solution:
    def generatePossibleNextMoves(self, currentState: str) -> List[str]:
        currentState = list(currentState)
        n, res = len(currentState), []
        for i in range(n - 1):
            if currentState[i] == '+' and currentState[i + 1] == '+':
                currentState[i], currentState[i + 1] = '-', '-'
                res.append(''.join(currentState)) 
                currentState[i], currentState[i + 1] = '+', '+'
        return res
```