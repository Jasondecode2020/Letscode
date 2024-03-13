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