## combinatorics

* [3128. Right Triangles](#3128-right-triangles)
* [357. Count Numbers with Unique Digits](#357-count-numbers-with-unique-digits)
* [2928. Distribute Candies Among Children I](#2928-distribute-candies-among-children-i)
* [2929. Distribute Candies Among Children II](#2929-distribute-candies-among-children-ii)
* [2927. Distribute Candies Among Children III](#2927-distribute-candies-among-children-iii)
* [1573. Number of Ways to Split a String](#1573-number-of-ways-to-split-a-string)
* [1569. Number of Ways to Reorder Array to Get Same BST](#1569-number-of-ways-to-reorder-array-to-get-same-bst)
* [1411. Number of Ways to Paint N × 3 Grid](#1411-number-of-ways-to-paint-n--3-grid)

```python
class Solution:
    def numberOfRightTriangles(self, grid: List[List[int]]) -> int:
        R, C = len(grid), len(grid[0])
        row, col = [0] * R, [0] * C
        for r in range(R):
            for c in range(C):
                if grid[r][c]:
                    row[r] += 1
                    col[c] += 1
        res = 0
        for r in range(R):
            for c in range(C):
                if grid[r][c] == 1:
                    res += (row[r] - 1) * (col[c] - 1)
        return res
```

* [357. Count Numbers with Unique Digits](#357-Count-Numbers-with-Unique-Digits)

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
        n = len(s)
        mod = 10 ** 9 + 7
        ones = s.count('1')
        if ones % 3:
            return 0
        k = ones // 3
        if k == 0:
            return ((n - 1) * (n - 2) // 2) % mod
        i = 0
        res = 1
        count = 0
        s = s.strip('0')
        n = len(s)
        while i < n:
            if s[i] == '1':
                count += 1
            if count == k:
                count = 0
                j = i + 1
                while j < n and s[j] != '1':
                    j += 1
                res *= (j - i)
                i = j
            else:
                i += 1
        return res % mod
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