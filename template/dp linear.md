### 651. 4 Keys Keyboard

```python
class Solution:
    def maxA(self, n: int) -> int:
        dp = [i for i in range(n + 1)]
        for i in range(4, n + 1):
            for j in range(i - 2):
                dp[i] = max(dp[i], dp[j] * (i - j - 2 + 1))
        return dp[-1]
```

### 650. 2 Keys Keyboard

```python
class Solution:
    def minSteps(self, n: int) -> int:
        dp = [inf] * (n + 1)
        dp[1] = 0
        for i in range(2, n + 1):
            for j in range(2, i + 1):
                if i % j == 0:
                    k = i // j
                    dp[i] = min(dp[i], dp[k] + 1 + j - 1)
        return dp[-1]
```