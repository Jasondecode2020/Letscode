### 50. Pow(x, n)

```python
class Solution:
    def myPow(self, x: float, n: int) -> float:
        def dfs(x, n):
            if n == 1:
                return x
            res = dfs(x, n // 2)
            if n % 2 == 0:
                return res * res 
            return x * res * res 
        if n == 0:
            return 1
        if n > 0:
            return dfs(x, n)
        else:
            return 1 / dfs(x, abs(n))
```

### 2961. Double Modular Exponentiation

```python
class Solution:
    def getGoodIndices(self, variables: List[List[int]], target: int) -> List[int]:
        res = []
        for i, (a, b, c, m) in enumerate(variables):
            A = pow(a, b, 10)
            if pow(A, c, m) == target:
                res.append(i)
        return res
```

### 829. Consecutive Numbers Sum

```python
class Solution:
    def consecutiveNumbersSum(self, n: int) -> int:
        res, n = 0, 2 * n 
        for i in range(1, int(sqrt(n)) + 1):
            if n % i == 0 and (n // i - (i - 1)) % 2 == 0:
                res += 1
        return res
```

### 1785. Minimum Elements to Add to Form a Given Sum

```python
class Solution:
    def minElements(self, nums: List[int], limit: int, goal: int) -> int:
        res = goal - sum(nums)
        res = abs(res)
        return ceil(res / limit)
```

### 1093. Statistics from a Large Sample

```python
class Solution:
    def sampleStats(self, count: List[int]) -> List[float]:
        mn, mx, mean, median, mode = inf, -inf, 0, 0, 0
        mx_f = max(count)
        total_num, total_val = 0, 0
        for i, f in enumerate(count):
            if f:
                mn = min(mn, i)
                mx = max(mx, i)
                total_num += f
                total_val += f * i 
            if f == mx_f:
                mode = i
        cnt = 0
        a, b = -1, -1
        for i, f in enumerate(count):
            if f:
                cnt += f
                if total_num % 2 == 0:
                    if a == -1 and cnt >= total_num // 2:
                        a = i 
                    if b == -1 and cnt >= total_num // 2 + 1:
                        b = i 
                    if b != -1:
                        break
                else:
                    if cnt >= total_num // 2 + 1:
                        median = i 
                        break
        if a != -1 and b != -1:
            median = (a + b) / 2
        mean = total_val / total_num
        return [mn, mx, mean, median, mode]
```

### 1247. Minimum Swaps to Make Strings Equal

```python
class Solution:
    def minimumSwap(self, s1: str, s2: str) -> int:
        c = Counter(s1 + s2)
        if c['x'] % 2 == 1 or c['y'] % 2 == 1:
            return -1
        res = 0
        d = defaultdict(int)
        for x, y in zip(s1, s2):
            if x + y == 'xy':
                d['xy'] += 1
            if x + y == 'yx':
                d['yx'] += 1
        xy = d['xy']
        yx = d['yx']
        res += d['xy'] // 2 if d['xy'] % 2 == 0 else d['xy'] // 2 + 1
        res += d['yx'] // 2 if d['yx'] % 2 == 0 else d['yx'] // 2 + 1
        return res
```

### 670. Maximum Swap

```python
class Solution:
    def maximumSwap(self, num: int) -> int:
        s = [int(i) for i in str(num)]
        n = len(s)
        for i in range(n):
            flag = False
            mx = s[i]
            mx_idx = 0
            for j in range(i + 1, n):
                if s[j] >= mx and s[j] != s[i]:
                    mx = s[j]
                    mx_idx = j 
                    flag = True
            
            if flag:
                s[mx_idx], s[i] = s[i], s[mx_idx]
                break 
        s = [str(i) for i in s]
        return int(''.join(s))
```

### 400. Nth Digit

```python
class Solution:
    def findNthDigit(self, n: int) -> int:
        cur, base = 1, 9
        while n - cur * base > 0:
            n -= cur * base 
            cur += 1
            base *= 10
        n -= 1
        num = 10 ** (cur - 1) + n // cur 
        idx = n % cur 
        return int(str(num)[idx])
```

### 1104. Path In Zigzag Labelled Binary Tree

```python
class Solution:
    def pathInZigZagTree(self, label: int) -> List[int]:
        row = int(log2(label)) + 1
        res = [0] * row
        while row:
            res[row-1] = label
            label = (2 ** row - 1 - label + 2 ** (row - 1)) // 2
            row -= 1
        return res
```

### 1963. Minimum Number of Swaps to Make the String Balanced

```python
class Solution:
    def minSwaps(self, s: str) -> int:
        diff = 0
        res = 0
        for c in s:
            if c == ']':
                diff += 1
            else:
                diff -= 1
            if diff > 0:
                if diff % 2 == 0:
                    res = max(res, abs(diff // 2))
                else:
                    res = max(res, abs(diff // 2) + 1)
        return res 
```

### 3195. Find the Minimum Area to Cover All Ones I

```python 
class Solution:
    def minimumArea(self, grid: List[List[int]]) -> int:
        R, C = len(grid), len(grid[0])
        top, bottom, left, right = inf, -inf, inf, -inf
        for r in range(R):
            for c in range(C):
                if grid[r][c] == 1:
                    top = min(top, c)
                    bottom = max(bottom, c)
                    left = min(left, r)
                    right = max(right, r)
        return (bottom - top + 1) * (right - left + 1)
```