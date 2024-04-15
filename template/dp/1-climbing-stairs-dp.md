## Question list

* [70. Climbing Stairs](#70-Climbing-Stairs)
* [509. Fibonacci Number](#509-Fibonacci-Number)
* [1137. N-th Tribonacci Number](#1137-N-th-Tribonacci-Number)
* [746. Min Cost Climbing Stairs](#746-Min-Cost-Climbing-Stairs)
* [377. Combination Sum IV](#377-Combination-Sum-IV)

### 70. Climbing Stairs

```python
class Solution:
    def climbStairs(self, n: int) -> int:
        first, second = 1, 2
        for i in range(3, n + 1):
            first, second = second, first + second 
        return second if n >= 2 else 1
```

### 509. Fibonacci Number

```python
class Solution:
    def fib(self, n: int) -> int:
        first, second = 0, 1
        for i in range(2, n + 1):
            first, second = second, first + second 
        return second if n >= 1 else 0
```

### 1137. N-th Tribonacci Number

```python
class Solution:
    def tribonacci(self, n: int) -> int:
        first, second, third = 0, 1, 1
        for i in range(3, n + 1):
            first, second, third = second, third, first + second + third
        return third if n >= 1 else 0
```

### 746. Min Cost Climbing Stairs

```python
class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        n = len(cost)
        for i in range(2, n):
            cost[i] = min(cost[i - 1], cost[i - 2]) + cost[i]
        return min(cost[-1], cost[-2])
```

### 377. Combination Sum IV

```python
class Solution:
    def combinationSum4(self, nums: List[int], target: int) -> int:
        nums.sort(reverse = True)
        @cache
        def dfs(t):
            if t > target:
                return 0
            if t == target:
                return 1
            return sum(dfs(t + n) for n in nums)
        return dfs(0)
```