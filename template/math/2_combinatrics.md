## combinatorics

### Multiplication Principle

* [2125. Number of Laser Beams in a Bank](#2125-number-of-laser-beams-in-a-bank) 1280
* [3128. Right Triangles](#3128-right-triangles) 1541
* [1573. Number of Ways to Split a String](#1573-number-of-ways-to-split-a-string)
* [2750. Ways to Split Array Into Good Subarrays](#2750-ways-to-split-array-into-good-subarrays)
* [2550. Count Collisions of Monkeys on a Polygon](#2550-count-collisions-of-monkeys-on-a-polygon)
* [1922. Count Good Numbers](#1922-count-good-numbers)
* [3067. Count Pairs of Connectable Servers in a Weighted Tree Network](#3067-count-pairs-of-connectable-servers-in-a-weighted-tree-network)
* [2147. Number of Ways to Divide a Long Corridor](#2147-number-of-ways-to-divide-a-long-corridor)
* [2963. Count the Number of Good Partitions](#2963-count-the-number-of-good-partitions)
* [2450. Number of Distinct Binary Strings After Applying Operations](#2450-number-of-distinct-binary-strings-after-applying-operations)

### Counting Combinations

* [62. Unique Paths](#62-unique-paths)
* [357. Count Numbers with Unique Digits](#357-count-numbers-with-unique-digits)
* [3179. Find the N-th Value After K Seconds](#3179-find-the-n-th-value-after-k-seconds)
* [1359. Count All Valid Pickup and Delivery Options](#1359-count-all-valid-pickup-and-delivery-options)
* [2400. Number of Ways to Reach a Position After Exactly k Steps](#2400-number-of-ways-to-reach-a-position-after-exactly-k-steps)
* [2514. Count Anagrams](#2514-count-anagrams)
* [634. Find the Derangement of An Array](#634-find-the-derangement-of-an-array)
s
* [2928. Distribute Candies Among Children I](#2928-distribute-candies-among-children-i)
* [2929. Distribute Candies Among Children II](#2929-distribute-candies-among-children-ii)
* [2927. Distribute Candies Among Children III](#2927-distribute-candies-among-children-iii)

* [1569. Number of Ways to Reorder Array to Get Same BST](#1569-number-of-ways-to-reorder-array-to-get-same-bst)
* [1411. Number of Ways to Paint N × 3 Grid](#1411-number-of-ways-to-paint-n--3-grid)


### 2125. Number of Laser Beams in a Bank

```python
class Solution:
    def numberOfBeams(self, bank: List[str]) -> int:
        nums = [item.count('1') for item in bank if item.count('1') > 0]
        res = 0
        for a, b in pairwise(nums):
            res += a * b 
        return res 
```

### 3128. Right Triangles

```python
class Solution:
    def numberOfRightTriangles(self, grid: List[List[int]]) -> int:
        R, C = len(grid), len(grid[0])
        row, col = [0] * R, [0] * C 
        for r in range(R):
            for c in range(C):
                if grid[r][c] == 1:
                    row[r] += 1
                    col[c] += 1
        return sum((row[r] - 1) * (col[c] - 1) for r in range(R) for c in range(C) if grid[r][c])
```

### 62. Unique Paths

```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        R, C = m, n
        dp = [[1] * C for i in range(R)]
        for r in range(1, m):
            for c in range(1, n):
                dp[r][c] = dp[r - 1][c] + dp[r][c - 1]
        return dp[-1][-1]
```

### 357. Count Numbers with Unique Digits

> if n = 0, return 1
> n = 1, can choose from 1 to 9, return 9 for 1 digit
> n = 2, for 2 digit, first digit can not choose 0, 9 cases, second digit can not choose
> first one, 9 cases, after than will be, 8, 7, 6, ...

```python
class Solution:
    def countNumbersWithUniqueDigits(self, n: int) -> int:
        def check(n):
            if n == 0:
                return 1
            res = 9
            for i in range(9, 9 - n + 1, -1):
                res *= i
            return res
        return sum(check(i) for i in range(n + 1))
```

### 3179. Find the N-th Value After K Seconds

```python
class Solution:
    def valueAfterKSeconds(self, n: int, k: int) -> int:
        mod = 10 ** 9 + 7
        arr = [1] * n
        while k:
            arr = list(accumulate(arr))
            k -= 1
        return arr[-1] % mod
```

### 1359. Count All Valid Pickup and Delivery Options

```python
class Solution:
    def countOrders(self, n: int) -> int:
        # 1 (P1, D1)             1
        # 2 (P1,P2,D1,D2)        6
        # 3 (P1,P2,P3,D1,D2,D3)  90
        mod = 10 ** 9 + 7
        @cache
        def dfs(n):
            if n == 1:
                return 1
            return dfs(n - 1) * n * (2 * n - 1) % mod
        return dfs(n)
```

### 2400. Number of Ways to Reach a Position After Exactly k Steps

```python
class Solution:
    def numberOfWays(self, startPos: int, endPos: int, k: int) -> int:
        mod = 10 ** 9 + 7
        @cache
        def f(pos, step):
            if step == 0:
                return pos == endPos
            if abs(pos - endPos) > k:
                return 0
            return (f(pos + 1, step - 1) + f(pos - 1, step - 1)) % mod 
        return f(startPos, k)
```

### 2514. Count Anagrams

```python
m = 10 ** 9 + 7
@lru_cache(None)
def modInverse(i):
    return pow(i, m - 2, m)

@lru_cache(None)
def Factorial(i):
    if i == 0:
        return 1
    return (i * Factorial(i - 1)) % m

class Solution:
    def countAnagrams(self, s: str) -> int:
        def check(c):
            res = 1
            for k, v in c.items():
                res *= modInverse(Factorial(v))
                res %= m
            return res

        res = 1
        for w in s.split(' '):
            n = len(w)
            c = Counter(w)
            res *= (Factorial(n) * check(c)) % m
            res %= m
        return res
```

### 634. Find the Derangement of An Array

```python
class Solution:
    def findDerangement(self, n: int) -> int:
        if n == 1: return 0
        mod = 10 ** 9 + 7
        f = [0] * (n + 1)
        f[2] = 1
        for i in range(3, n + 1):
            f[i] = ((i - 1) * (f[i - 1] + f[i - 2])) % mod
        return f[-1]
```

## math combinations

- use plate to find 

### 2928. Distribute Candies Among Children I
### 2929. Distribute Candies Among Children II
### 2927. Distribute Candies Among Children III

```python
def C2(n):
    return n * (n - 1) // 2 if n >= 1 else 0
class Solution:
    def distributeCandies(self, n: int, limit: int) -> int:
        return C2(n + 2) - 3 * C2(n + 2 - (limit + 1)) + 3 * C2(n + 2 - 2 * (limit + 1)) - C2(n + 2 - 3 * (limit + 1))
```

### 1573. Number of Ways to Split a String

```python
class Solution:
    def numWays(self, s: str) -> int:
        mod = 10 ** 9 + 7
        n = s.count('1')
        if n % 3:
            return 0
        
        N = len(s) - 1
        if n == 0:
            return (N * (N - 1) // 2) % mod

        l, r = 0, 0
        cnt = 0
        t = n // 3
        for c in s:
            if c == '1':
                cnt += 1
            if cnt == t:
                l += 1
            if cnt == 2 * t:
                r += 1
        return (l * r) % mod
```

### 2750. Ways to Split Array Into Good Subarrays

```python
class Solution:
    def numberOfGoodSubarraySplits(self, nums: List[int]) -> int:
        while nums and nums[-1] == 0:
            nums.pop()
        if not nums:
            return 0
        mod = 10 ** 9 + 7
        res = 1
        i = 0
        n = len(nums)
        while i < n:
            if nums[i] == 1:
                j = i + 1
                while j < n and nums[j] == 0:
                    j += 1
                res = (res * (j - i)) % mod 
                i = j 
            else:
                i += 1
        return res 

class Solution:
    def numberOfGoodSubarraySplits(self, nums: List[int]) -> int:
        if not sum(nums):
            return 0
        mod = 10 ** 9 + 7
        res = 1
        arr = [i for i, n in enumerate(nums) if n == 1]
        for a, b in pairwise(arr):
            res = (res * (b - a)) % mod 
        return res
```

### 2550. Count Collisions of Monkeys on a Polygon

```python
class Solution:
    def monkeyMove(self, n: int) -> int:
        mod = 10 ** 9 + 7
        return (pow(2, n, mod) - 2) % mod
```

### 1922. Count Good Numbers

```python
class Solution:
    def countGoodNumbers(self, n: int) -> int:
        mod = 10 ** 9 + 7
        if n % 2 == 0:
            return pow(5, n // 2, mod) * pow(4, n // 2, mod) % mod 
        return pow(5, n // 2 + 1, mod) * pow(4, n // 2, mod) % mod 
```

### 3067. Count Pairs of Connectable Servers in a Weighted Tree Network

```python
class Solution:
    def countPairsOfConnectableServers(self, edges: List[List[int]], signalSpeed: int) -> List[int]:
        n = len(edges) + 1
        g = defaultdict(list)
        for u, v, w in edges:
            g[u].append((v, w))
            g[v].append((u, w))

        def dfs(x, pa, t):
            cnt = 0 if t % signalSpeed else 1
            for y, w in g[x]:
                if y != pa:
                    cnt += dfs(y, x, t + w)
            return cnt 

        res = [0] * n 
        for i, a in g.items():
            if len(a) > 1:
                s = 0
                for x, w in a:
                    cnt = dfs(x, i, w)
                    res[i] += s * cnt 
                    s += cnt 
        return res 
```

### 2147. Number of Ways to Divide a Long Corridor

```python
class Solution:
    def numberOfWays(self, corridor: str) -> int:
        mod = 10 ** 9 + 7
        res, cnt_s, pre = 1, 0, 0
        for i, c in enumerate(corridor):
            if c == 'S':
                cnt_s += 1
                if cnt_s >= 3 and cnt_s % 2:
                    res = res * (i - pre) % mod 
                pre = i 
        return res if cnt_s and cnt_s % 2 == 0 else 0
```

### 2963. Count the Number of Good Partitions

```python
class Solution:
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        mod = 10 ** 9 + 7
        d = defaultdict(list)
        for i, n in enumerate(nums):
            if n in d:
                d[n][1] = i 
            else:
                d[n] = [i, i]
        
        a = sorted(d.values())
        res = 0
        r = a[0][1]
        for left, right in a[1:]:
            if left > r:
                res += 1
            r = max(r, right)
        return pow(2, res, mod)
```

### 2450. Number of Distinct Binary Strings After Applying Operations

```python
class Solution:
    def countDistinctStrings(self, s: str, k: int) -> int:
        mod = 10 ** 9 + 7
        n = len(s)
        return pow(2, (n - k + 1), mod)
```

### 1569. Number of Ways to Reorder Array to Get Same BST

```python
class Solution:
    def numOfWays(self, nums: List[int]) -> int:
        mod = 10 ** 9 + 7
        def f(nums):
            if len(nums) <= 2:
                return 1
            val = nums[0]
            l = [n for n in nums if n < val]
            r = [n for n in nums if n > val]
            return comb(len(l) + len(r), len(l)) * f(l) * f(r)
        return (f(nums) - 1) % mod
```


### 1411. Number of Ways to Paint N × 3 Grid

```python
class Solution:
    def numOfWays(self, n: int) -> int:
        mod = 10 ** 9 + 7
        @cache
        def dfs(i, p):
            if i == n:
                return 1
            if i == 0:
                return 6 * (dfs(i + 1, 0) + dfs(i + 1, 1)) % mod
            else:
                if p == 0:
                    return 2 * (dfs(i + 1, 0) + dfs(i + 1, 1)) % mod
                else:
                    return (2 * dfs(i + 1, 0) + 3 * dfs(i + 1, 1)) % mod 
        return dfs(0, -1)
```

### 1359. Count All Valid Pickup and Delivery Options

```python
class Solution:
    def countOrders(self, n: int) -> int:
        mod = 10 ** 9 + 7
        @cache
        def dfs(n):
            if n == 1:
                return 1
            if n == 2:
                return 6
            return dfs(n - 1) * ((2 * n - 1) + (2 * (n - 1) * (2 * n - 1) // 2))
        return dfs(n) % mod
```

### 1922. Count Good Numbers

```python
class Solution:
    def countGoodNumbers(self, n: int) -> int:
        mod = 10 ** 9 + 7
        if n % 2 == 0:
            return pow(5, n // 2, mod) * pow(4, n // 2, mod) % mod
        return pow(5, n // 2 + 1, mod) * pow(4, n // 2, mod) % mod
```