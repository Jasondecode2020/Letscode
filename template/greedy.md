# greedy

## simulation

### 455. Assign Cookies

```python
class Solution:
    def findContentChildren(self, g: List[int], s: List[int]) -> int:
        i, j, res = 0, 0, 0
        g, s = sorted(g), sorted(s)
        while i < len(g) and j < len(s):
            if s[j] >= g[i]:
                res += 1
                i += 1
                j += 1
            else:
                j += 1
        return res
```

### 860. Lemonade Change

```python
class Solution:
    def lemonadeChange(self, bills: List[int]) -> bool:
        d = {5: 0, 10: 0}
        for bill in bills:
            if bill == 5:
                d[bill] += 1
            elif bill == 10:
                d[bill] += 1
                d[5] -= 1
            elif bill == 20:
                if d[10] > 0:
                    d[10] -= 1
                    d[5] -= 1
                else:
                    d[5] -= 3
            if d[5] < 0:
                return False
        return True
```

## sorting

### 179. Largest Number

```python
def isSwap(s1, s2):
            return int(s1 + s2) < int(s2 + s1)
        origin = nums[::]
        nums = [str(n) for n in nums]
        n = len(nums)
        for i in range(n):
            for j in range(i + 1, n):
                if isSwap(nums[i], nums[j]):
                    nums[i], nums[j] = nums[j], nums[i]
        return ''.join(nums) if sum(origin) != 0 else '0'
```

## intervals

## regret

### 767. Reorganize String

```python
class Solution:
    def reorganizeString(self, s: str) -> str:
        c = Counter(s)
        mx = max(c.values())
        if mx > len(s) // 2 + 1 and len(s) % 2:
            return ''
        if mx > len(s) // 2 and len(s) % 2 == 0:
            return ''
        res = ''
        pq = []
        for k, v in c.items():
            heappush(pq, (-v, k))
        while pq:
            v, k = heappop(pq)
            res += k
            if pq:
                v2, k2 = heappop(pq)
                res += k2 
                v2 = -v2
                if v2 - 1 != 0:
                    heappush(pq, (-(v2 - 1), k2))
            v = -v
            if v - 1 != 0:
                heappush(pq, (-(v - 1), k))
        return res
```

### 945. Minimum Increment to Make Array Unique

```python
class Solution:
    def minIncrementForUnique(self, nums: List[int]) -> int:
        nums.sort()
        res = 0
        for i in range(1, len(nums)):
            if nums[i] <= nums[i - 1]:
                res += nums[i - 1] - nums[i] + 1
                nums[i] = nums[i - 1] + 1
        return res
```

### 1090. Largest Values From Labels

```python
class Solution:
    def largestValsFromLabels(self, values: List[int], labels: List[int], numWanted: int, useLimit: int) -> int:
        d = defaultdict(list)
        for v, l in zip(values, labels):
            d[l].append(v)
        res = []
        for k, v in d.items():
            v.sort(reverse = True)
            res += v[:useLimit]
        
        res.sort(reverse = True)
        return sum(res[: numWanted])
```