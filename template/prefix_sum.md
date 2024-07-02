## 1 basics

* [303. Range Sum Query - Immutable](#303-range-sum-query---immutable)
* [2559. Count Vowel Strings in Ranges](#2559-count-vowel-strings-in-ranges)
* [2389. Longest Subsequence With Limited Sum](#2389-longest-subsequence-with-limited-sum)
* [3152. Special Array II](#3152-special-array-ii)
* [2438. Range Product Queries of Powers](#2438-range-product-queries-of-powers)
* [2055. Plates Between Candles](#2055-plates-between-candles)
* [1744. Can You Eat Your Favorite Candy on Your Favorite Day?](#1744-can-you-eat-your-favorite-candy-on-your-favorite-day)

## 2 prefix + hash

* [930. Binary Subarrays With Sum](#930-binary-subarrays-with-sum)
* [560. Subarray Sum Equals K](#560-subarray-sum-equals-k)
* [1524. Number of Sub-arrays With Odd Sum](#1524-number-of-sub-arrays-with-odd-sum)
* [974. Subarray Sums Divisible by K](#974-subarray-sums-divisible-by-k)
* [523. Continuous Subarray Sum](#523-continuous-subarray-sum)
* [437. Path Sum III](#437-path-sum-iii)
* [525. Contiguous Array](#525-contiguous-array)
* [1546. Maximum Number of Non-Overlapping Subarrays With Sum Equals Target](#1546-maximum-number-of-non-overlapping-subarrays-with-sum-equals-target)

## 3 prefix + distance

* [1685. Sum of Absolute Differences in a Sorted Array](#2588-count-the-number-of-beautiful-subarrays)
* [2615. Sum of Distances](#2615-sum-of-distances)
* [2602. Minimum Operations to Make All Array Elements Equal](#1177-can-make-palindrome-from-substring)

## 4 prefix + xor

* [2588. Count the Number of Beautiful Subarrays](#2588-count-the-number-of-beautiful-subarrays)
* [1310. XOR Queries of a Subarray](#1310-xor-queries-of-a-subarray)
* [1177. Can Make Palindrome from Substring](#1177-can-make-palindrome-from-substring)
* [1371. Find the Longest Substring Containing Vowels in Even Counts](#1371-find-the-longest-substring-containing-vowels-in-even-counts)
* [1915. Number of Wonderful Substrings](#1915-number-of-wonderful-substrings)

## 5 2d prefix

* [304. Range Sum Query 2D - Immutable](#304-range-sum-query-2d---immutable)
* [1314. Matrix Block Sum](#1314-matrix-block-sum)
* [3070. Count Submatrices with Top-Left Element and Sum Less Than k](#3070-count-submatrices-with-top-left-element-and-sum-less-than-k)
* [1292. Maximum Side Length of a Square with Sum Less than or Equal to Threshold](#1292-maximum-side-length-of-a-square-with-sum-less-than-or-equal-to-threshold)
* [221. Maximal Square](#221-maximal-square)
* [1277. Count Square Submatrices with All Ones](#1277-count-square-submatrices-with-all-ones)
* [1504. Count Submatrices With All Ones](#1504-count-submatrices-with-all-ones)
* [1074. Number of Submatrices That Sum to Target](#1074-number-of-submatrices-that-sum-to-target)
* [1738. Find Kth Largest XOR Coordinate Value](#1738-find-kth-largest-xor-coordinate-value)
* [3148. Maximum Difference Score in a Grid](#3148-maximum-difference-score-in-a-grid)

### 303. Range Sum Query - Immutable

```python
class NumArray:

    def __init__(self, nums: List[int]):
        self.a = list(accumulate(nums, initial = 0))

    def sumRange(self, left: int, right: int) -> int:
        return self.a[right + 1] - self.a[left]
```

### 2559. Count Vowel Strings in Ranges

```python
class Solution:
    def vowelStrings(self, words: List[str], queries: List[List[int]]) -> List[int]:
        v = 'aeiou'
        def check(s):
            return 1 if s[0] in v and s[-1] in v else 0
        nums = [check(s) for s in words]
        pre = list(accumulate(nums, initial = 0))
        return [pre[b + 1] - pre[a] for a, b in queries]
```

### 2389. Longest Subsequence With Limited Sum

```python
class Solution:
    def answerQueries(self, nums: List[int], queries: List[int]) -> List[int]:
        nums.sort()
        a = list(accumulate(nums, initial = 0))
        res = [0] * len(queries)
        for i, q in enumerate(queries):
            j = bisect_left(a, q)
            if j < len(a) and a[j] == q:
                res[i] = j 
            else:
                res[i] = j - 1
        return res
```

### 3152. Special Array II

```python
from itertools import accumulate
class Solution:
    def isArraySpecial(self, nums: List[int], queries: List[List[int]]) -> List[bool]:
        pre = []
        for a, b in pairwise(nums):
            if (a + b) % 2 == 0:
                pre.append(1)
            else:
                pre.append(0)
        pre = list(accumulate(pre, initial = 0))
        return [pre[b] - pre[a] == 0 for a, b in queries]
```

### 2438. Range Product Queries of Powers

```python
class Solution:
    def productQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        # n = 15
        # [1111]
        # [1, 2, 4, 8]
        mod = 10 ** 9 + 7
        s = bin(n)[2:]
        res = []
        for i, c in enumerate(s[::-1]):
            if c != '0':
                res.append(int(c) * 2 ** i)
        pre = [1] * (len(res) + 1)
        for i in range(len(res)):
            pre[i + 1] = pre[i] * res[i]
        return [(pre[b + 1] // pre[a]) % mod for a, b in queries]
```

### 2055. Plates Between Candles

```python
class Solution:
    def platesBetweenCandles(self, s: str, queries: List[List[int]]) -> List[int]:
        right_closest, left_closest = inf, inf 
        n = len(s)
        left = [0] * n
        for i, c in enumerate(s):
            if c == '|':
                left_closest = i 
            left[i] = left_closest
        right = [0] * n
        for i in range(n - 1, -1, -1):
            if s[i] == '|':
                right_closest = i 
            right[i] = right_closest
        
        pre = [0] * n
        for i, c in enumerate(s):
            if c == '*':
                pre[i] = 1
        pre = list(accumulate(pre, initial = 0))
        res = [0] * len(queries)
        for i, (a, b) in enumerate(queries):
            if left[b] != inf and right[a] != inf and pre[left[b] + 1] - pre[right[a]] > 0:
                res[i] = pre[left[b] + 1] - pre[right[a]]
        return res
```

### 1744. Can You Eat Your Favorite Candy on Your Favorite Day?

```python
class Solution:
    def canEat(self, candiesCount: List[int], queries: List[List[int]]) -> List[bool]:
        pre = list(accumulate(candiesCount, initial = 0))
        res = [False] * len(queries)
        for i, (t, d, c) in enumerate(queries):
            earliest, latest = pre[t] // c,  pre[t + 1] - 1
            if earliest <= d <= latest:
                res[i] = True
        return res
```


### 930. Binary Subarrays With Sum

```python
class Solution:
    def numSubarraysWithSum(self, nums: List[int], goal: int) -> int:
        n, total, res = len(nums), 0, 0
        pre = defaultdict(int)
        pre[0] = 1 
        for v in nums:
            total += v 
            res += pre[total - goal]
            pre[total] += 1
        return res 
```

### 560. Subarray Sum Equals K


```python
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        pre = defaultdict(int)
        pre[0] = 1
        res, total = 0, 0
        for v in nums:
            total += v 
            res += pre[total - k]
            pre[total] += 1
        return res 
```

### 1524. Number of Sub-arrays With Odd Sum

```python
class Solution:
    def numOfSubarrays(self, arr: List[int]) -> int:
        pre_odd, pre_even = 0, 0
        pre = list(accumulate(arr))
        res = 0
        mod = 10 ** 9 + 7
        for n in pre:
            if n % 2 == 1:
                res += pre_even + 1
                pre_odd += 1
            else:
                res += pre_odd
                pre_even += 1
        return res % mod
```

### 974. Subarray Sums Divisible by K

```python
class Solution:
    def subarraysDivByK(self, nums: List[int], k: int) -> int:
        res, presum = 0, 0
        d = defaultdict(int)
        d[0] = 1
        for n in nums:
            presum += n 
            presum %= k
            if presum in d:
                res += d[presum]
            d[presum] = d[presum] + 1
        return res
```

### 523. Continuous Subarray Sum

```python
class Solution:
    def checkSubarraySum(self, nums: List[int], k: int) -> bool:
        presum, d = 0, {0: -1}
        for i, n in enumerate(nums):
            presum += n 
            if presum % k in d and i - d[presum % k] >= 2:
                return True
            if presum % k not in d:
                d[presum % k] = i
        return False
```

### 437. Path Sum III

```python
class Solution:
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> int:
        def dfs(node, presum):
            if not node:
                return 
            curr =  node.val + presum
            self.res += prefix[curr - targetSum]
            prefix[curr] += 1
            dfs(node.left, presum + node.val)
            dfs(node.right, presum + node.val)
            prefix[curr] -= 1
        
        self.res = 0
        prefix = defaultdict(int)
        prefix[0] = 1
        dfs(root, 0)
        return self.res
```

### 525. Contiguous Array

```python
class Solution:
    def findMaxLength(self, nums: List[int]) -> int:
        res, count, d = 0, 0, {0: -1}
        for i, n in enumerate(nums):
            if n:
                count += 1
            else:
                count -= 1
            if count in d:
                res = max(res, i - d[count])
            else:
                d[count] = i
        return res
```

### 1546. Maximum Number of Non-Overlapping Subarrays With Sum Equals Target

```python
class Solution:
    def maxNonOverlapping(self, nums: List[int], target: int) -> int:
        s, res = set([0]), 0
        for a in accumulate(nums):
            if a - target in s:
                s.clear()
                res += 1
            s.add(a)
        return res
```

### 1685. Sum of Absolute Differences in a Sorted Array

```python
class Solution:
    def getSumAbsoluteDifferences(self, nums: List[int]) -> List[int]:
        # [0, 2, 5, 10]
        pre = list(accumulate(nums, initial = 0))
        res = []
        n = len(pre)
        for i in range(1, n):
            ans = (pre[-1] - pre[i] - (n - 1 - i) * nums[i - 1]) + ((i - 1) * nums[i - 1] - pre[i - 1])
            res.append(ans)
        return res
```

### 2588. Count the Number of Beautiful Subarrays

```python
class Solution:
    def beautifulSubarrays(self, nums: List[int]) -> int:
        d = defaultdict(int)
        d[0] = 1
        total = 0
        res = 0
        for n in nums:
            total ^= n
            res += d[total]
            d[total] += 1
        return res 
```

### 2615. Sum of Distances

```python
class Solution:
    def distance(self, nums: List[int]) -> List[int]:
        # [0, 0, 2, 5]
        d = defaultdict(list)
        for i, n in enumerate(nums):
            d[n].append(i)
        n = len(nums)
        res = [0] * n 
        for arr in d.values():
            pre = list(accumulate(arr, initial = 0))
            m = len(pre)
            for i in range(1, m):
                res[arr[i - 1]] = (pre[-1] - pre[i] - (m - 1 - i) * arr[i - 1]) + (i - 1) * arr[i - 1] - pre[i - 1]
        return res
```

### 2602. Minimum Operations to Make All Array Elements Equal

```python
class Solution:
    def minOperations(self, nums: List[int], queries: List[int]) -> List[int]:
        # [1, 3, 6, 8]
        # [0, 1, 4, 10, 18]
        nums.sort()
        pre = list(accumulate(nums, initial = 0))
        n = len(queries)
        m = len(pre)
        res = [0] * n 
        for i, q in enumerate(queries):
            j = bisect_left(nums, q)
            ans = pre[-1] - pre[j] - (m - 1 - j) * q + q * j - pre[j]
            res[i] = ans 
        return res
```

### 1310. XOR Queries of a Subarray

```python
class Solution:
    def xorQueries(self, arr: List[int], queries: List[List[int]]) -> List[int]:
        arr = [0] + arr
        for i in range(1, len(arr)):
            arr[i] ^= arr[i - 1]
        res = []
        for s, e in queries:
            res.append(arr[e + 1] ^ arr[s])
        return res
```


### 1177. Can Make Palindrome from Substring

```python
class Solution:
    def canMakePaliQueries(self, s: str, queries: List[List[int]]) -> List[bool]:
        n = len(s)
        dp = [[0] * 26 for i in range(n + 1)]
        for i, c in enumerate(s, start = 1):
            dp[i][ord(c) - ord('a')] += 1
        for i in range(1, n + 1):
            for j in range(26):
                dp[i][j] += dp[i - 1][j]

        res = [False] * len(queries)
        for j, (l, r, k) in enumerate(queries):
            odd = 0
            for i in range(26):
                if (dp[r + 1][i] - dp[l][i]) % 2 == 1:
                    odd += 1
            if odd // 2 <= k:
                res[j] = True
        return res
```


### 1371. Find the Longest Substring Containing Vowels in Even Counts

```python
class Solution:
    def findTheLongestSubstring(self, s: str) -> int:
        idx = [-1] + [inf] * ((1 << 5) - 1)
        res, mask = 0, 0
        vowel = set(list('aeiou'))
        d = Counter()
        for i, c in enumerate('aeoui'):
            d[c] = i
        for i, c in enumerate(s):
            if c in vowel:
                mask ^= 1 << d[c]
            res = max(res, i - idx[mask])
            idx[mask] = min(i, idx[mask])
        return res 
```


### 1915. Number of Wonderful Substrings

```python
class Solution:
    def wonderfulSubstrings(self, word: str) -> int:
        index = [1] + [0] * ((1 << 10) - 1)
        res, mask = 0, 0
        for i, c in enumerate(word):
            mask ^= 1 << (ord(c) - ord('a'))
            res += index[mask]
            for j in range(10):
                res += index[mask ^ (1 << j)]
            index[mask] += 1
        return res
```

### 304. Range Sum Query 2D - Immutable

```python
class NumMatrix:

    def __init__(self, matrix: List[List[int]]):
        self.mat = matrix
        R, C = len(matrix), len(matrix[0])
        for r in range(1, R):
            self.mat[r][0] += self.mat[r - 1][0]
        for c in range(1, C):
            self.mat[0][c] += self.mat[0][c - 1]
        for r in range(1, R):
            for c in range(1, C):
                self.mat[r][c] += self.mat[r - 1][c] + self.mat[r][c - 1] - self.mat[r - 1][c - 1]
    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        left = self.mat[row2][col1 - 1] if col1 - 1 >= 0 else 0
        top = self.mat[row1 - 1][col2] if row1 - 1 >= 0 else 0
        topLeft = self.mat[row1 - 1][col1 - 1] if row1 >= 1 and col1 >= 1 else 0
        return self.mat[row2][col2] - left - top + topLeft
```

### 1314. Matrix Block Sum

```python
class Solution:
    def matrixBlockSum(self, mat: List[List[int]], k: int) -> List[List[int]]:
        R, C = len(mat), len(mat[0])
        for r in range(1, R):
            mat[r][0] += mat[r - 1][0]
        for c in range(1, C):
            mat[0][c] += mat[0][c - 1]
        for r in range(1, R):
            for c in range(1, C):
                mat[r][c] += mat[r - 1][c] + mat[r][c - 1] - mat[r - 1][c - 1]
        res = [[0] * C for r in range(R)]
        for r in range(R):
            for c in range(C):
                left = mat[min(R - 1, r + k)][c - k - 1] if c - k - 1 >= 0 else 0
                top = mat[r - k - 1][min(C - 1, c + k)] if r - k - 1 >= 0 else 0
                topLeft = mat[r - k - 1][c - k - 1] if r >= k + 1 and c >= k + 1 else 0
                res[r][c] = mat[min(R - 1, r + k)][min(C - 1, c + k)] - left - top + topLeft
        return res 
```

### 3070. Count Submatrices with Top-Left Element and Sum Less Than k

```python
class Solution:
    def countSubmatrices(self, grid: List[List[int]], k: int) -> int:
        R, C = len(grid), len(grid[0])
        dp = [[0] * C for r in range(R)]
        dp[0][0] = grid[0][0]
        for c in range(1, C):
            dp[0][c] += dp[0][c - 1] + grid[0][c]
        for r in range(1, R):
            dp[r][0] += dp[r - 1][0] + grid[r][0]
        for r in range(1, R):
            for c in range(1, C):
                dp[r][c] = dp[r - 1][c] + dp[r][c - 1] + grid[r][c] - dp[r - 1][c - 1]
        return sum(1 for r in range(R) for c in range(C) if dp[r][c] <= k)
```

### 1292. Maximum Side Length of a Square with Sum Less than or Equal to Threshold

```python
class Solution:
    def maxSideLength(self, mat: List[List[int]], threshold: int) -> int:
        R, C = len(mat), len(mat[0])
        dp = [[0] * (C + 1) for r in range(R + 1)]
        for r in range(1, R + 1):
            for c in range(1, C + 1):
                dp[r][c] = dp[r - 1][c] + dp[r][c - 1] - dp[r - 1][c - 1] + mat[r - 1][c - 1]

        def check(m):
            for r in range(m, R + 1):
                for c in range(m, C + 1):
                    s = dp[r][c] - dp[r - m][c] - dp[r][c - m] + dp[r - m][c - m]
                    if s <= threshold:
                        return True
            return False

        l, r, res = 0, 10 ** 9, 0
        while l <= r:
            m = l + (r - l) // 2
            if check(m):
                res = m 
                l = m + 1
            else:
                r = m - 1
        return res
```

### 221. Maximal Square

```python
class Solution:
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        R, C, res = len(matrix), len(matrix[0]), 0
        dp=[[0] * (C + 1) for r in range(R + 1)]
        for i in range(1, R + 1):
            for j in range(1, C + 1):
                if(matrix[i - 1][j - 1] == "1"):
                    dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]) + 1
                    res = max(dp[i][j], res)
        return res * res
```

### 1277. Count Square Submatrices with All Ones

```python
class Solution:
    def countSquares(self, matrix: List[List[int]]) -> int:
        # same as largest aubsquare
        # 1 init a dp
        dp = [[0] * (len(matrix[0]) + 1)  for i in range(len(matrix) + 1)]
        # 2 write function: T = mn
        for i in range(1, len(dp)): # rows
            for j in range(1, len(dp[0])): # cols
                if matrix[i-1][j-1] == 1:
                    dp[i][j] = min([dp[i-1][j-1], dp[i-1][j], dp[i][j-1]]) + 1
        
        return sum([sum(item) for item in dp])
```

### 1504. Count Submatrices With All Ones

```python
class Solution:
    def numSubmat(self, mat: List[List[int]]) -> int:
        R, C, res = len(mat), len(mat[0]), 0
        hist = [0] * (C + 1) # design hist for hist[-1] to be 0
        for i in range(R):
            stack, dp = [-1], [0] * (C + 1) # design of stack for easy calc
            for j in range(C):
                hist[j] = 0 if mat[i][j] == 0 else hist[j] + 1
                while hist[j] < hist[stack[-1]]:
                    stack.pop()
                dp[j] = dp[stack[-1]] + hist[j] * (j - stack[-1]) # Important!!
                stack.append(j)
            res += sum(dp)
        return res
```

### 1074. Number of Submatrices That Sum to Target

```python
class Solution:
    def numSubmatrixSumTarget(self, matrix: List[List[int]], target: int) -> int:
        R, C = len(matrix), len(matrix[0])
        for row in matrix:
            for c in range(C - 1):
                row[c + 1] += row[c]

        res = 0
        for i in range(C):
            for j in range(i, C):
                nums = [matrix[k][j] - (matrix[k][i - 1] if i > 0 else 0) for k in range(R)]
                presum, d = 0, {0: 1}
                for n in nums:
                    presum += n
                    if presum - target in d:
                        res += d[presum - target]
                    d[presum] = d.get(presum, 0) + 1
        return res
```

### 1738. Find Kth Largest XOR Coordinate Value

```python
class Solution:
    def kthLargestValue(self, matrix: List[List[int]], k: int) -> int:
        R, C = len(matrix), len(matrix[0])
        res = []
        for r in range(1, R):
            matrix[r][0] ^= matrix[r - 1][0]
        for c in range(1, C):
            matrix[0][c] ^= matrix[0][c - 1]
        for r in range(1, R):
            for c in range(1, C):
                matrix[r][c] ^= matrix[r - 1][c] ^ matrix[r][c - 1] ^ matrix[r - 1][c - 1]
        for r in range(R):
            for c in range(C):
                res.append(matrix[r][c])
        res.sort(reverse = True)
        return res[k - 1]
```

### 3148. Maximum Difference Score in a Grid

```python
class Solution:
    def maxScore(self, grid: List[List[int]]) -> int:
        R, C = len(grid), len(grid[0])
        f = [[inf] * (C + 1) for r in range(R + 1)]
        res = -inf
        for r in range(1, R + 1):
            for c in range(1, C + 1):
                mn = min(f[r - 1][c], f[r][c - 1])
                res = max(res, grid[r - 1][c - 1] - mn)
                f[r][c] = min(mn, grid[r - 1][c - 1])
        return res
```

### 643. Maximum Average Subarray I

```python
class Solution:
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        prefix = list(accumulate(nums, initial = 0))
        res = -inf
        for i in range(k, len(prefix)):
            res = max(res, (prefix[i] - prefix[i - k]) / k)
        return res
```

### 1176. Diet Plan Performance (same as 643)

```python
class Solution:
    def dietPlanPerformance(self, calories: List[int], k: int, lower: int, upper: int) -> int:
        prefix = list(accumulate(calories, initial = 0))
        res = 0
        for i in range(k, len(prefix)):
            if prefix[i] - prefix[i - k] > upper:
                res += 1
            elif prefix[i] - prefix[i - k] < lower:
                res -= 1
        return res
```

### 1763. Longest Nice Substring

```python
class Solution:
    def longestNiceSubstring(self, s: str) -> str:
        def check(s):
            s = set(list(s))
            for c in s:
                if c.lower() not in s or c.upper() not in s:
                    return False
            return True

        n, res = len(s), ''
        for i in range(n):
            for j in range(i + 1, n):
                if check(s[i: j + 1]) and len(s[i: j + 1]) > len(res):
                    res = s[i: j + 1]
        return res
```

### 1984. Minimum Difference Between Highest and Lowest of K Scores

```python
class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        res = inf
        nums.sort()
        for i in range(k - 1, len(nums)):
            res = min(res, nums[i] - nums[i - k + 1]) 
        return res
```

### 2269. Find the K-Beauty of a Number

```python
class Solution:
    def divisorSubstrings(self, num: int, k: int) -> int:
        n = num
        num, res = str(num), 0
        for i in range(k - 1, len(num)):
            if int(num[i - k + 1: i + 1]) == 0:
                continue
            if n % int(num[i - k + 1: i + 1]) == 0:
                res += 1
        return res
```

### 2379. Minimum Recolors to Get K Consecutive Black Blocks

```python
class Solution:
    def minimumRecolors(self, blocks: str, k: int) -> int:
        d, res, l = defaultdict(int), inf, 0
        count = 0
        for r, c in enumerate(blocks):
            if c == 'W':
                count += 1
            if r - l + 1 == k:
                res = min(res, count)
                if blocks[l] == 'W':
                    count -= 1
                l += 1
        return res
```

### 1343. Number of Sub-arrays of Size K and Average Greater than or Equal to Threshold

```python
class Solution:
    def numOfSubarrays(self, arr: List[int], k: int, threshold: int) -> int:
        arr = list(accumulate(arr, initial = 0))
        res = 0
        for i in range(k, len(arr)):
            if arr[i] - arr[i - k] >= threshold * k:
                res += 1
        return res
```

## base case

- 2574. Left and Right Sum Differences
prefix sum and suffix sum

- 1588. Sum of All Odd Length Subarrays
prefix sum
follow up: enumeration

-  2485. Find the Pivot Integer
prefix sum

- 1732. Find the Highest Altitude
simulation

- 303. Range Sum Query - Immutable
prefix sum

- 1480. Running Sum of 1d Array
prefix sum

## Easy

### 2574. Left and Right Sum Differences

```python
class Solution:
    def leftRightDifference(self, nums: List[int]) -> List[int]:
        leftSum = list(accumulate(nums, initial = 0))[:-1]
        rightSum = list(accumulate(nums[::-1], initial = 0))[:-1]
        rightSum.reverse()
        return [abs(a - b) for a, b in zip(leftSum, rightSum)]
```

### 1588. Sum of All Odd Length Subarrays

```python
class Solution:
    def sumOddLengthSubarrays(self, arr: List[int]) -> int:
        arr = list(accumulate(arr, initial = 0))
        n = len(arr)
        res = 0
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % 2 == 1:
                    res += arr[j] - arr[i]
        return res
```

- O(n) follow up

```python
class Solution:
    def sumOddLengthSubarrays(self, arr: List[int]) -> int:
        # include the current number for both left and right
        n, res = len(arr), 0
        for i in range(n): 
            l, r = i + 1, n - i
            l_even, l_odd = l // 2, (l + 1) // 2
            r_even, r_odd = r // 2, (r + 1) // 2
            res += (l_even * r_even + l_odd * r_odd) * arr[i]
        return res
```

### 2485. Find the Pivot Integer

```python
class Solution:
    def pivotInteger(self, n: int) -> int:
        res = -1
        arr = list(accumulate([i for i in range(1, n + 1)], initial = 0))
        for i in range(1, len(arr)):
            if arr[i] == arr[-1] - arr[i - 1]:
                res = i
                break
        return res
```

### 1732. Find the Highest Altitude

```python
class Solution:
    def largestAltitude(self, gain: List[int]) -> int:
        res = [0]
        for g in gain:
            res.append(g + res[-1])
        return max(res)
```

### 1480. Running Sum of 1d Array

```python
class Solution:
    def runningSum(self, nums: List[int]) -> List[int]:
        return list(accumulate(nums))
```

###  2848. Points That Intersect With Cars

```python
class Solution:
    def numberOfPoints(self, nums: List[List[int]]) -> int:
        s = set()
        for l, r in nums:
            for i in range(l, r + 1):
                s.add(i)
        return len(s)
```

- diff array

```python
class Solution:
    def numberOfPoints(self, nums: List[List[int]]) -> int:
        max_end = max(end for start, end in nums)
        diff = [0] * (max_end + 2)
        for s, e in nums:
            diff[s] += 1
            diff[e + 1] -= 1
        return sum(s > 0 for s in accumulate(diff))
```

### 1413. Minimum Value to Get Positive Step by Step Sum

```python
class Solution:
    def minStartValue(self, nums: List[int]) -> int:
        arr = list(accumulate(nums, initial = 0))
        minNum = min(arr)
        return 1 - minNum
```

### 1991. Find the Middle Index in Array

```python
class Solution:
    def findMiddleIndex(self, nums: List[int]) -> int:
        arr = list(accumulate(nums, initial = 0))
        n = len(arr)
        for i in range(1, n):
            if arr[i - 1] == arr[n - 1] - arr[i]:
                return i - 1
        return -1
```

### 724. Find Pivot Index (same as 1991)

```python
class Solution:
    def pivotIndex(self, nums: List[int]) -> int:
        arr = list(accumulate(nums, initial = 0))
        n = len(arr)
        for i in range(1, n):
            if arr[i - 1] == arr[n - 1] - arr[i]:
                return i - 1
        return -1
```

### 1893. Check if All the Integers in a Range Are Covered

```python
class Solution:
    def isCovered(self, ranges: List[List[int]], left: int, right: int) -> bool:
        diff = [0] * 52
        for s, e in ranges:
            diff[s] += 1
            diff[e + 1] -= 1
        return all(v > 0 for i, v in enumerate(accumulate(diff)) if left <= i <= right)
```


## Medium

### 497. Random Point in Non-overlapping Rectangles

- random.choices

```python
class Solution:

    def __init__(self, rects: List[List[int]]):
        self.rects = rects
        self.w = [(y2-y1+1) * (x2-x1+1) for x1, y1, x2, y2 in self.rects]

    def pick(self) -> List[int]:
        n = len(self.rects)
        i = random.choices(range(n), self.w)[0]
        rect = self.rects[i]
        y = randint(rect[1], rect[-1])
        x = randint(rect[0], rect[2])
        return [x, y]
```

### 2391. Minimum Amount of Time to Collect Garbage

```python
class Solution:
    def garbageCollection(self, garbage: List[str], travel: List[int]) -> int:
        n = len(garbage)
        total = sum(len(i) for i in garbage)
        lastG, lastM, lastP = 0, 0, 0
        for i in range(n - 1, -1, -1):
            if 'G' in garbage[i]:
                lastG = i
                break
        for i in range(n - 1, -1, -1):
            if 'M' in garbage[i]:
                lastM = i
                break
        for i in range(n - 1, -1, -1):
            if 'P' in garbage[i]:
                lastP = i
                break
        arr = list(accumulate(travel))
        res = 0
        for i in [lastG, lastM, lastP]:
            if i - 1 >= 0:
                res += arr[i - 1]
        return res + total
```

### 1442. Count Triplets That Can Form Two Arrays of Equal XOR

- xor

```python
class Solution:
    def countTriplets(self, arr: List[int]) -> int:
        res = 0
        arr = [0] + arr
        n = len(arr)
        for i in range(1, n):
            arr[i] = arr[i] ^ arr[i - 1]
        for i in range(1, n):
            for j in range(i + 1, n):
                for k in range(j, n):
                    if arr[j - 1] ^ arr[i - 1] == arr[j - 1] ^ arr[k]:
                        res += 1
        return res
```

- O(n^2)

```python
class Solution:
    def countTriplets(self, arr: List[int]) -> int:
        res = 0
        arr = [0] + arr
        n = len(arr)
        for i in range(1, n):
            arr[i] = arr[i] ^ arr[i - 1]
        for i in range(1, n):
            for k in range(i + 1, n):
                if arr[i - 1] == arr[k]:
                    res += k - i
        return res
```

- O(n) later

### 2640. Find the Score of All Prefixes of an Array

```python
class Solution:
    def findPrefixScore(self, nums: List[int]) -> List[int]:
        res = []
        origin = nums[::]
        for i in range(1, len(nums)):
            nums[i] = max(nums[i], nums[i - 1])
        cur = 0
        for i, n in enumerate(nums):
            cur += n + origin[i]
            res.append(cur)
        return res
```





## sliding window

### 209. Minimum Size Subarray Sum

```python
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        for i in range(1, len(nums)):
            nums[i] += nums[i - 1]
        
        nums = [0] + nums
        l, r, res = 0, 1, inf
        while r < len(nums):
            if nums[r] - nums[l] >= target:
                res = min(res, r - l)
                l += 1
            else:
                r += 1
        return res if res != inf else 0
```

### 1004. Max Consecutive Ones III

```python
class Solution:
    def longestOnes(self, nums: List[int], k: int) -> int:
        for i in range(1, len(nums)):
            nums[i] += nums[i - 1]

        nums = [0] + nums
        l, res = 0, 0
        for r in range(1, len(nums)):
            if nums[r] - nums[l] + k >= r - l:
                res = max(res, r - l)
            else:
                l += 1
        return res
```

### 525. Contiguous Array

```python
class Solution:
    def findMaxLength(self, nums: List[int]) -> int:
        res, count, d = 0, 0, {0: -1}
        for i, n in enumerate(nums):
            if n:
                count += 1
            else:
                count -= 1
            if count in d:
                res = max(res, i - d[count])
            else:
                d[count] = i
        return res
```

## binary search

### 209. Minimum Size Subarray Sum

```python
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        for i in range(1, len(nums)):
            nums[i] += nums[i - 1]
        
        nums, res = [0] + nums, inf
        for l, n in enumerate(nums):
            r = bisect_left(nums, n + target)
            if r < len(nums):
                res = min(res, r - l)
        return res if res != inf else 0
```

## subarray

### 325. Maximum Size Subarray Sum Equals k

```python
class Solution:
    def maxSubArrayLen(self, nums: List[int], k: int) -> int:
        res, presum, d = 0, 0, {0: -1}
        for i, n in enumerate(nums):
            presum += n
            if presum - k in d:
                res = max(res, i - d[presum - k])
            if presum not in d:
                d[presum] = i
        return res
```





### 1074. Number of Submatrices That Sum to Target

```python
class Solution:
    def numSubmatrixSumTarget(self, matrix: List[List[int]], target: int) -> int:
        R, C = len(matrix), len(matrix[0])
        for row in matrix:
            for c in range(C - 1):
                row[c + 1] += row[c]

        res = 0
        # turn each column as 1d array, and find all combinations
        for i in range(C):
            for j in range(i, C):
                # solve 1d subarray problem
                nums = [matrix[k][j] - (matrix[k][i - 1] if i > 0 else 0) for k in range(R)]
                presum, d = 0, {0: 1}
                for n in nums:
                    presum += n
                    if presum - target in d:
                        res += d[presum - target]
                    d[presum] = d.get(presum, 0) + 1
        return res
```

### 363. Max Sum of Rectangle No Larger Than K

```python
from sortedcontainers import SortedList
class Solution:
    def maxSumSubmatrix(self, matrix: List[List[int]], k: int) -> int:
        res, R, C = -inf, len(matrix), len(matrix[0])
        for row in matrix:
            for c in range(C - 1):
                row[c + 1] += row[c]
        for i in range(C):
            for j in range(i, C):
                nums = [matrix[k][j] - (matrix[k][i - 1] if i > 0 else 0) for k in range(R)]
                presum, sortedList = 0, SortedList([0])
                for n in nums:
                    presum += n
                    idx = sortedList.bisect_left(presum - k)
                    if idx < len(sortedList):
                        res = max(res, presum - sortedList[idx])
                    sortedList.add(presum)
        return res
```

### 2428. Maximum Sum of an Hourglass

```python
class Solution:
    def maxSum(self, grid: List[List[int]]) -> int:
        def check(r, c):
            return grid[r][c] + sum(grid[r - 1][c - 1: c + 2]) + sum(grid[r + 1][c - 1: c + 2])
        R, C = len(grid), len(grid[0])
        res = 0
        for r in range(1, R - 1):
            for c in range(1, C - 1):
                res = max(res, check(r, c))
        return res
```

### 1685. Sum of Absolute Differences in a Sorted Array

```python
class Solution:
    def getSumAbsoluteDifferences(self, nums: List[int]) -> List[int]:
        # [0, 2, 5, 10]
        res = []
        presum = list(accumulate(nums, initial = 0))
        n = len(presum)
        for i in range(1, n):
            res.append((presum[-1] - presum[i - 1]) - nums[i - 1] * (n - i) + nums[i - 1] * (i - 1) - presum[i - 1])
        return res
```

### 848. Shifting Letters

```python
class Solution:
    def shiftingLetters(self, s: str, shifts: List[int]) -> str:
        prefix = list(accumulate(shifts[::-1]))[::-1]
        res = ''
        for c, n in zip(s, prefix):
            res += chr((n + ord(c) - 97) % 26 + 97)
        return res
```

### 2955. Number of Same-End Substrings

```python
class Solution:
    def sameEndSubstringCount(self, s: str, queries: List[List[int]]) -> List[int]:
        n = len(s)
        dp = [[0] * (n + 1) for r in range(26)]
        for i in range(1, n + 1):
            dp[ord(s[i - 1]) - ord('a')][i] += 1
        for i, arr in enumerate(dp):
            dp[i] = list(accumulate(arr))
        
        res = []
        for s, e in queries:
            s, e = s + 1, e + 1
            ans = 0
            for i in range(26):
                m = dp[i][e] - dp[i][s - 1]
                ans += m * (m + 1) // 2
            res.append(ans)
        return res
```

### 2207. Maximize Number of Subsequences in a String

```python
class Solution:
    def maximumSubsequenceCount(self, text: str, pattern: str) -> int:
        res = ''
        for t in text:
            if t in pattern:
                res += t

        res1, res2 = pattern[0] + res, res + pattern[1]
        if pattern[0] == pattern[1]:
            m = res1.count(pattern[0])
            return m * (m - 1) // 2
            
        def check(s):
            res = 0
            b = s.count(pattern[1])
            for c in s:
                if c == pattern[0]:
                    res += b
                else:
                    b -= 1
            return res
        return max(check(res1), check(res2))
```

### 2789. Largest Element in an Array after Merge Operations

```python
class Solution:
    def maxArrayValue(self, nums: List[int]) -> int:
        n = len(nums)
        i = n - 1
        while i > 0:
            j = i - 1
            while j >= 0 and nums[j] <= nums[j + 1]:
                nums[j] += nums[j + 1]
                j -= 1
            i = j      
        return max(nums)
```

### 1894. Find the Student that Will Replace the Chalk

```python
class Solution:
    def chalkReplacer(self, chalk: List[int], k: int) -> int:
        s = sum(chalk)
        k %= s 
        for i, n in enumerate(chalk):
            k -= n
            if k < 0:
                return i 
```

### 2270. Number of Ways to Split Array

```python
class Solution:
    def waysToSplitArray(self, nums: List[int]) -> int:
        prefix = list(accumulate(nums, initial = 0))
        res, n = 0, len(nums)
        for i in range(1, n):
            if prefix[i] >= prefix[-1] - prefix[i]:
                res += 1
        return res
```

### 2587. Rearrange Array to Maximize Prefix Score

```python
class Solution:
    def maxScore(self, nums: List[int]) -> int:
        prefix = list(accumulate(sorted(nums, reverse = True)))
        return sum(n > 0 for n in prefix)
```

### 2222. Number of Ways to Select Buildings

```python
class Solution:
    def numberOfWays(self, s: str) -> int:
        zero, one = s.count('0'), s.count('1')
        def check(b, total):
            left, right, res = 0, 0, 0
            for i, c in enumerate(s):
                if c == b:
                    right = total - left
                    res += left * right
                else:
                    left += 1
            return res 
        return check('0', one) + check('1', zero)
```

### 1895. Largest Magic Square

```python
class Solution:
    def largestMagicSquare(self, grid: List[List[int]]) -> int:
        R, C = len(grid), len(grid[0])
        def valid(mat):
            n = len(mat)
            if not mat:
                return False
            val = sum(mat[0])
            for m in mat:
                if sum(m) != val:
                    return False
            R, C = len(mat), len(mat[0])
            for c in range(C):
                sumCol = 0
                for r in range(R):
                    sumCol += mat[r][c]
                if sumCol != val:
                    return False 
            diag1 = sum(mat[r][c] for r in range(R) for c in range(C) if r == c)
            diag2 = sum(mat[r][c] for r in range(R) for c in range(C) if r + c + 1 == n)
            if diag1 != val or diag2 != val:
                return False
            return True

        def check(m):
            for r in range(m - 1, R):
                for c in range(m - 1, C):
                    res = []
                    g = grid[r - m + 1: r + 1][:]
                    for item in g:
                        res.append(item[c - m + 1: c + 1])
                    if valid(res):
                        return True
            return False

        for i in range(min(R, C), -1, -1):
            if check(i):
                return i
```

### 1546. Maximum Number of Non-Overlapping Subarrays With Sum Equals Target

```python
class Solution:
    def maxNonOverlapping(self, nums: List[int], target: int) -> int:
        s, res = set([0]), 0
        for a in accumulate(nums):
            if a - target in s:
                s.clear()
                res += 1
            s.add(a)
        return res
```

### 1930. Unique Length-3 Palindromic Subsequences

```python
class Solution:
    def countPalindromicSubsequence(self, s: str) -> int:
        n = len(s)
        dp = [[0] * n for r in range(26)]
        for i, c in enumerate(s):
            dp[ord(c) - ord('a')][i] += 1
        for i, arr in enumerate(dp):
            dp[i] = list(accumulate(arr))
    
        res = 0
        for i in range(26):
            seen = set()
            for j in range(1, n - 1):
                left, right = dp[i][j - 1], dp[i][-1] - dp[i][j]
                if left and right and s[j] not in seen:
                    seen.add(s[j])
                    res += 1
        return res
```

### 1652. Defuse the Bomb

```python
class Solution:
    def decrypt(self, code: List[int], k: int) -> List[int]:
        n = len(code)
        code = code * 2
        prefix = list(accumulate(code, initial = 0))
        res = []
        if k >= 0:
            for i in range(1, n + 1):
                res.append(prefix[i + k] - prefix[i])
        else:
            for i in range(n + 1, 2 * n + 1):
                res.append(prefix[i - 1] - prefix[i + k - 1])
        return res
```

## Maximum Subarray template

### 53. Maximum Subarray

```python
'''
template
'''
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        n = len(nums)
        pre = [nums[0]] * n 
        for i in range(1, n):
            pre[i] = max(nums[i], pre[i - 1] + nums[i])
        return max(pre)
```

### 2606. Find the Substring With Maximum Cost

```python
'''
based on 53
'''
class Solution:
    def maximumCostSubstring(self, s: str, chars: str, vals: List[int]) -> int:
        n = len(s)
        d = Counter()
        for c, v in zip(chars, vals):
            d[c] = v 
        nums = [0] * n 
        for i, c in enumerate(s):
            if c not in d:
                nums[i] = ord(c) - ord('a') + 1
            else:
                nums[i] = d[c]
        pre = [nums[0]] * n 
        for i in range(1, n):
            pre[i] = max(nums[i], pre[i - 1] + nums[i])
        return max(0, max(pre))
```

### 918. Maximum Sum Circular Subarray

```python
'''
based on 53
'''
class Solution:
    def maxSubarraySumCircular(self, nums: List[int]) -> int:
        n = len(nums)
        maxNum, minNum = [nums[0]] * n, [nums[0]] * n
        for i in range(1, n):
            maxNum[i] = max(nums[i], nums[i] + maxNum[i - 1])
        for i in range(1, n):
            minNum[i] = min(nums[i], nums[i] + minNum[i - 1])
        if sum(nums) - min(minNum) == 0:
            return max(maxNum)
        return max(max(maxNum), sum(nums) - min(minNum))
```

### 2321. Maximum Score Of Spliced Array

```python
'''
based on 53
'''
class Solution:
    def maximumsSplicedArray(self, nums1: List[int], nums2: List[int]) -> int:
        arr1 = [n1 - n2 for n1, n2 in zip(nums1, nums2)]
        arr2 = [n2 - n1 for n1, n2 in zip(nums1, nums2)]
        n = len(arr1)
        pre1, pre2 = [arr1[0]] * n, [arr2[0]] * n 
        for i in range(1, n):
            pre1[i] = max(arr1[i], pre1[i - 1] + arr1[i])
            pre2[i] = max(arr2[i], pre2[i - 1] + arr2[i])
        return max(max(pre1) + sum(nums2), max(pre2) + sum(nums1))
```

## base case

### 724. Find Pivot Index

```python
class Solution:
    def pivotIndex(self, nums: List[int]) -> int:
        nums = list(accumulate(nums, initial = 0))
        for i, n in enumerate(nums):
            if i > 0 and nums[i - 1] == nums[-1] - nums[i]:
                return i - 1
        return -1
```

### 1480. Running Sum of 1d Array

```python
class Solution:
    def runningSum(self, nums: List[int]) -> List[int]:
        return list(accumulate(nums))
```

## sliding window

### 209. Minimum Size Subarray Sum

```python
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        for i in range(1, len(nums)):
            nums[i] += nums[i - 1]
        
        nums = [0] + nums
        l, r, res = 0, 1, inf
        while r < len(nums):
            if nums[r] - nums[l] >= target:
                res = min(res, r - l)
                l += 1
            else:
                r += 1
        return res if res != inf else 0
```

### 1004. Max Consecutive Ones III

```python
class Solution:
    def longestOnes(self, nums: List[int], k: int) -> int:
        for i in range(1, len(nums)):
            nums[i] += nums[i - 1]

        nums = [0] + nums
        l, res = 0, 0
        for r in range(1, len(nums)):
            if nums[r] - nums[l] + k >= r - l:
                res = max(res, r - l)
            else:
                l += 1
        return res
```

### 525. Contiguous Array

```python
class Solution:
    def findMaxLength(self, nums: List[int]) -> int:
        res, count, d = 0, 0, {0: -1}
        for i, n in enumerate(nums):
            if n:
                count += 1
            else:
                count -= 1
            if count in d:
                res = max(res, i - d[count])
            else:
                d[count] = i
        return res
```

## binary search

### 209. Minimum Size Subarray Sum

```python
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        for i in range(1, len(nums)):
            nums[i] += nums[i - 1]
        
        nums, res = [0] + nums, inf
        for l, n in enumerate(nums):
            r = bisect_left(nums, n + target)
            if r < len(nums):
                res = min(res, r - l)
        return res if res != inf else 0
```

## subarray

### 325. Maximum Size Subarray Sum Equals k

```python
class Solution:
    def maxSubArrayLen(self, nums: List[int], k: int) -> int:
        res, presum, d = 0, 0, {0: -1}
        for i, n in enumerate(nums):
            presum += n
            if presum - k in d:
                res = max(res, i - d[presum - k])
            if presum not in d:
                d[presum] = i
        return res
```

### 560. Subarray Sum Equals K

```python
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        res, presum, d = 0, 0, {0: 1} # [1, 2, 3]
        for n in nums:
            presum += n
            if presum - k in d:
                res += d[presum - k]
            d[presum] = d.get(presum, 0) + 1
        return res
```

### 974. Subarray Sums Divisible by K

```python
class Solution:
    def subarraysDivByK(self, nums: List[int], k: int) -> int:
        res, presum, d = 0, 0, {0: 1}
        for n in nums:
            presum += n 
            if presum % k in d:
                res += d[presum % k]
            d[presum % k] = d.get(presum % k, 0) + 1
        return res
```

### 523. Continuous Subarray Sum

```python
class Solution:
    def checkSubarraySum(self, nums: List[int], k: int) -> bool:
        presum, d = 0, {0: -1}
        for i, n in enumerate(nums):
            presum += n 
            if presum % k in d and i - d[presum % k] >= 2:
                return True
            if presum % k not in d:
                d[presum % k] = i
        return False
```

### 1074. Number of Submatrices That Sum to Target

```python
class Solution:
    def numSubmatrixSumTarget(self, matrix: List[List[int]], target: int) -> int:
        R, C = len(matrix), len(matrix[0])
        for row in matrix:
            for c in range(C - 1):
                row[c + 1] += row[c]

        res = 0
        # turn each column as 1d array, and find all combinations
        for i in range(C):
            for j in range(i, C):
                # solve 1d subarray problem
                nums = [matrix[k][j] - (matrix[k][i - 1] if i > 0 else 0) for k in range(R)]
                presum, d = 0, {0: 1}
                for n in nums:
                    presum += n
                    if presum - target in d:
                        res += d[presum - target]
                    d[presum] = d.get(presum, 0) + 1
        return res
```

### 363. Max Sum of Rectangle No Larger Than K

```python
from sortedcontainers import SortedList
class Solution:
    def maxSumSubmatrix(self, matrix: List[List[int]], k: int) -> int:
        res, R, C = -inf, len(matrix), len(matrix[0])
        for row in matrix:
            for c in range(C - 1):
                row[c + 1] += row[c]
        for i in range(C):
            for j in range(i, C):
                nums = [matrix[k][j] - (matrix[k][i - 1] if i > 0 else 0) for k in range(R)]
                presum, sortedList = 0, SortedList([0])
                for n in nums:
                    presum += n
                    idx = sortedList.bisect_left(presum - k)
                    if idx < len(sortedList):
                        res = max(res, presum - sortedList[idx])
                    sortedList.add(presum)
        return res
```
