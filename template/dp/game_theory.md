## template 1: min-max dp - game theory

### 486. Predict the Winner

```python
class Solution:
    def predictTheWinner(self, nums: List[int]) -> bool:
        @cache
        def dfs(l, r, alice):
            if l > r:
                return 0
            res = -inf if alice else inf
            if alice:
                res = max(dfs(l + 1, r, not alice) + nums[l], dfs(l, r - 1, not alice) + nums[r])
            else:
                res = min(dfs(l + 1, r, not alice), dfs(l, r - 1, not alice))
            return res
        return dfs(0, len(nums) - 1, True) >= sum(nums) / 2
```

### 877. Stone Game

```python
class Solution:
    def stoneGame(self, piles: List[int]) -> bool:
        @cache
        def dfs(l, r, alice):
            if l > r:
                return 0 
            res = -inf if alice else inf 
            if alice:
                res = max(res, dfs(l + 1, r, False) + piles[l], dfs(l, r - 1, False) + piles[r])
            else:
                res = min(res, dfs(l + 1, r, True), dfs(l, r - 1, True))
            return res

        half = sum(piles) // 2
        res = dfs(0, len(piles) - 1, True)
        return res > half
```

### 1140. Stone Game II

```python
class Solution:
    def stoneGameII(self, piles: List[int]) -> int:
        @cache
        def dfs(i, alice, M):
            if i >= n:
                return 0
            res = -inf if alice else inf
            if alice:
                for j in range(i, min(2 * M + i, n)):
                    res = max(res, pre[j + 1] - pre[i] + dfs(j + 1, False, max(M, j + 1 - i)))
            else:
                for j in range(i, min(2 * M + i, n)):
                    res = min(res, dfs(j + 1, True, max(M, j + 1 - i)))
            return res
        n = len(piles)
        pre = list(accumulate(piles, initial = 0))
        return dfs(0, True, 1)
```

### 1406. Stone Game III

```python
class Solution:
    def stoneGameIII(self, stoneValue: List[int]) -> str:
        @cache
        def dfs(i, alice):
            if i >= n:
                return 0
            res = -inf if alice else inf 
            if alice:
                for j in range(i, i + 3):
                    res = max(res, sum(stoneValue[i: j + 1]) + dfs(j + 1, False))
            else:
                for j in range(i, i + 3):
                    res = min(res, dfs(j + 1, True))
            return res
            
        n = len(stoneValue)
        s = sum(stoneValue)
        res = dfs(0, True)
        if res > s - res:
            return 'Alice'
        elif res < s - res:
            return 'Bob'
        else:
            return 'Tie'
```

### 1510. Stone Game IV

```python
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        @cache 
        def dfs(i):
            if i == 0:
                return False
            for j in range(int(sqrt(i)), 0, -1):
                if not dfs(i - j * j):
                    return True
            return False
        return dfs(n)
```

### 1563. Stone Game V

```python
class Solution:
    def stoneGameV(self, stoneValue: List[int]) -> int:
        @lru_cache(None)
        def dfs(l, r):
            if l == r:
                return 0
            totol = pre[r + 1] - pre[l]
            res, sumL = 0, 0
            for i in range(l, r):
                sumL += stoneValue[i]
                sumR = totol - sumL 
                if min(sumL, sumR) * 2 < res: # trim
                    break
                if sumL > sumR:
                    res = max(res, dfs(i + 1, r) + sumR)
                elif sumL < sumR:
                    res = max(res, dfs(l, i) + sumL)
                else:
                    res = max(res, max(dfs(l, i), dfs(i + 1, r)) + sumL)
            return res 
        n = len(stoneValue)
        pre = list(accumulate(stoneValue, initial = 0))
        res = dfs(0, n - 1)
        return res
```

### 1686. Stone Game VI

```python
class Solution:
    def stoneGameVI(self, aliceValues: List[int], bobValues: List[int]) -> int:
        pairs = sorted([(a, b) for a, b in zip(aliceValues, bobValues)], key = lambda x: -sum(x))
        alice, bob = sum([a for a, b in pairs][::2]), sum([b for a, b in pairs][1::2])
        if alice == bob:
            return 0
        elif alice > bob:
            return 1
        else:
            return -1
```

### 1690. Stone Game VII

```python
class Solution:
    def stoneGameVII(self, stones: List[int]) -> int:
        @cache
        def dfs(l, r):
            if l == r:
                return 0
            left = pre[r + 1] - pre[l + 1] - dfs(l + 1, r)
            right = pre[r] - pre[l] - dfs(l, r - 1)
            return max(left, right)

        n = len(stones)
        pre = list(accumulate(stones, initial = 0))
        res = dfs(0, n - 1)
        dfs.cache_clear()
        return res 
```

### 2029. Stone Game IX

```python
class Solution:
    def stoneGameIX(self, stones: List[int]) -> bool:
        cnt = [0] * 3
        for t in stones:
            cnt[t % 3] += 1
        if cnt[0] % 2 == 0 and cnt[1] >= 1 and cnt[2] >= 1:
            return True
        if cnt[0] % 2 == 1 and abs(cnt[1] - cnt[2]) >= 3:
            return True
        return False
```