### 97. Interleaving String

```python
class Solution:
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        R, C = len(s1), len(s2)
        if R + C != len(s3):
            return False
        dp = [[False] * (C + 1) for r in range(R + 1)]
        dp[0][0] = True
        for r in range(1, R + 1):
            dp[r][0] = dp[r - 1][0] and (s1[r - 1] == s3[r - 1])
        for c in range(1, C + 1):
            dp[0][c] = dp[0][c - 1] and (s2[c - 1] == s3[c - 1])
        for r in range(1, R + 1):
            for c in range(1, C + 1):
                dp[r][c] = (dp[r][c - 1] and s3[r + c - 1] == s2[c - 1]) or (dp[r - 1][c] and s3[r + c - 1] == s1[r - 1])
        return dp[-1][-1]
```