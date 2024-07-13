## Hash

### 1. 2 sum(16)

* [1. Two Sum](#1-two-sum)  
* [1512. Number of Good Pairs](#1512-number-of-good-pairs)
* [2815. Max Pair Sum in an Array](#2815-max-pair-sum-in-an-array)
* [2748. Number of Beautiful Pairs](#2748-number-of-beautiful-pairs)
* [219. Contains Duplicate II](#219-contains-duplicate-ii)
* [121. Best Time to Buy and Sell Stock](#121-best-time-to-buy-and-sell-stock)
* [2342. Max Sum of a Pair With Equal Sum of Digits](#2342-max-sum-of-a-pair-with-equal-sum-of-digits)
* [1010. Pairs of Songs With Total Durations Divisible by 60](#1010-pairs-of-songs-with-total-durations-divisible-by-60)
* [3185. Count Pairs That Form a Complete Day II](#3185-count-pairs-that-form-a-complete-day-ii)
* [454. 4Sum II](#454-4sum-ii)
* [2874. Maximum Value of an Ordered Triplet II](#2874-maximum-value-of-an-ordered-triplet-ii)
* [1014. Best Sightseeing Pair](#1014-best-sightseeing-pair)
* [1214. Two Sum BSTs](#1214-two-sum-bsts)
* [2971. Find Polygon With the Largest Perimeter](#2971-find-polygon-with-the-largest-perimeter)
* [1679. Max Number of K-Sum Pairs](#1679-max-number-of-k-sum-pairs)
* [2964. Number of Divisible Triplet Sums](#2964-number-of-divisible-triplet-sums)

### 1. Two Sum

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        d = defaultdict()
        for i, v in enumerate(nums):
            res = target - v
            if res in d:
                return [d[res], i]
            d[v] = i
```

### 1512. Number of Good Pairs

```python
class Solution:
    def numIdenticalPairs(self, nums: List[int]) -> int:
        d = defaultdict(int)
        res = 0
        for v in nums:
            if v in d:
                res += d[v]
            d[v] += 1
        return res
```

### 2815. Max Pair Sum in an Array

```python
class Solution:
    def maxSum(self, nums: List[int]) -> int:
        res = -1
        d = defaultdict(lambda: -inf)
        for v in nums:
            mx_d = max(map(int, str(v)))
            res = max(res, v + d[mx_d])
            d[mx_d] = max(d[mx_d], v)
        return res 
```

### 2748. Number of Beautiful Pairs

```python
class Solution:
    def countBeautifulPairs(self, nums: List[int]) -> int:
        res = 0
        cnt = [0] * 10
        for x in nums:
            for y, c in enumerate(cnt):
                if gcd(y, x % 10) == 1:
                    res += c 
            cnt[int(str(x)[0])] += 1
        return res 
```

### 219. Contains Duplicate II

```python
class Solution:
    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        d = defaultdict(int)
        for i, n in enumerate(nums):
            if n in d:
                if i - d[n] <= k:
                    return True
            d[n] = i
        return False
```

### 121. Best Time to Buy and Sell Stock

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        profit, lowest = 0, prices[0]
        n = len(prices)
        for i in range(1, n):
            lowest = min(lowest, prices[i])
            profit = max(profit, prices[i] - lowest)
        return profit
```

### 2342. Max Sum of a Pair With Equal Sum of Digits

```python
class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        d = defaultdict(int)
        res = -1
        for n in nums:
            total = sum(map(int, str(n)))
            if total in d:
                res = max(res, n + d[total])
            d[total] = max(d[total], n)
        return res
```

### 1010. Pairs of Songs With Total Durations Divisible by 60

```python
class Solution:
    def numPairsDivisibleBy60(self, time: List[int]) -> int:
        d = defaultdict(int)
        res = 0
        for t in time:
            ans = 60 - (t % 60)
            if ans in d:
                res += d[ans]
            d[t % 60] += 1
        m = d[0]
        res += m * (m - 1) // 2
        return res
```

### 3185. Count Pairs That Form a Complete Day II

```python
class Solution:
    def countCompleteDayPairs(self, hours: List[int]) -> int:
        d = defaultdict(int)
        res = 0
        for h in hours:
            ans = 24 - (h % 24)
            if ans in d:
                res += d[ans]
            d[h % 24] += 1
        m = d[0]
        res += m * (m - 1) // 2
        return res
```

### 454. 4Sum II

```python
class Solution:
    def fourSumCount(self, nums1: List[int], nums2: List[int], nums3: List[int], nums4: List[int]) -> int:
        d = defaultdict(int)
        n = len(nums1)
        res = 0
        for i in range(n):
            for j in range(n):
                two = nums1[i] + nums2[j]
                d[two] += 1

        for i in range(n):
            for j in range(n):
                two = nums3[i] + nums4[j]
                two = -two 
                if two in d:
                    res += d[two]
        return res
```

### 2874. Maximum Value of an Ordered Triplet II

```python
class Solution:
    def maximumTripletValue(self, nums: List[int]) -> int:
        mx, diff, res = 0, 0, 0
        for n in nums:
            res = max(res, diff * n)
            mx = max(mx, n)
            diff = max(diff, mx - n)
        return res 
```

### 1014. Best Sightseeing Pair

```python
class Solution:
    def maxScoreSightseeingPair(self, values: List[int]) -> int:
        mx, res = 0, 0
        for i, v in enumerate(values):
            res = max(res, mx + v - i)
            mx = max(mx, i + v)
        return res
```

### 1814. Count Nice Pairs in an Array

```python
class Solution:
    def countNicePairs(self, nums: List[int]) -> int:
        mod = 10 ** 9 + 7
        def reversePosNum(x):
            res = 0
            while x:
                res = res * 10 + x % 10
                x //= 10
            return res 

        d = defaultdict(int)
        res = 0
        for i, n in enumerate(nums):
            val = n - reversePosNum(n)
            if val in d:
                res += d[val]
            d[val] += 1
        return res % mod
```

### 1214. Two Sum BSTs

'''
Given the roots of two binary search trees, root1 and root2, return true if and only if there is a node in the first tree and a node in the second tree whose values sum up to a given integer target.

Example 1:

Input: root1 = [2,1,4], root2 = [1,0,3], target = 5
Output: true
Explanation: 2 and 3 sum up to 5.
'''

```python
class Solution:
    def twoSumBSTs(self, root1: Optional[TreeNode], root2: Optional[TreeNode], target: int) -> bool:
        d = defaultdict(int)
        def dfs(node):
            if node:
                val = node.val 
                d[val] += 1
                dfs(node.left)
                dfs(node.right)
        def dfs2(node):
            if node:
                val = node.val 
                ans = target - val
                if ans in d:
                    return True
                return dfs2(node.left) or dfs2(node.right)
            return False

        dfs(root1)
        self.res = False
        return dfs2(root2)
```

### 2971. Find Polygon With the Largest Perimeter

```python
class Solution:
    def largestPerimeter(self, nums: List[int]) -> int:
        # [1, 1, 2, 3, 5, 12, 50]
        nums.sort()
        res = -1
        pre = list(accumulate(nums))
        n = len(pre)
        for i in range(2, n):
            if pre[i - 1] > nums[i]:
                res = pre[i]
        return res
```

### 1679. Max Number of K-Sum Pairs

```python
class Solution:
    def maxOperations(self, nums: List[int], k: int) -> int:
        nums.sort()
        l, r = 0, len(nums) - 1
        res = 0
        while l < r:
            two = nums[l] + nums[r]
            if two == k:
                res += 1
                l += 1
                r -= 1
            elif two > k:
                r -= 1
            else:
                l += 1
        return res 
```

### 2964. Number of Divisible Triplet Sums

```python
class Solution:
    def divisibleTripletCount(self, nums: List[int], d: int) -> int:
        n = len(nums)
        c = defaultdict(int)
        res = 0
        for i in range(n):
            if i > 1:
                ans = d - nums[i] % d
                if ans in c:
                    res += c[ans]
                if ans - d in c:
                    res += c[ans - d]
            for j in range(i):
                two = nums[i] + nums[j]
                two %= d
                c[two] += 1
        return res 
```

### 336. Palindrome Pairs

```python
class Solution:
    def palindromePairs(self, words: List[str]) -> List[List[int]]:
        lookup = {w: i for i, w in enumerate(words)}
        res = []
        for i, w in enumerate(words):
            for j in range(len(w) + 1):
                pre, suf = w[:j], w[j:]
                if pre[::-1] == pre and suf[::-1] != w and suf[::-1] in lookup:
                    res.append([lookup[suf[::-1]], i])
                if suf[::-1] == suf and pre[::-1] != w and pre[::-1] in lookup and j != len(w):
                    # j != len(w)，j = w的情况已经出现过, avoid duplicate
                    res.append([i, lookup[pre[::-1]]])
        return res
```

### 594. Longest Harmonious Subsequence

```python
class Solution:
    def findLHS(self, nums: List[int]) -> int:
        d = Counter(nums)
        res = 0
        for k in d:
            if k - 1 in d:
                res = max(res, d[k] + d[k - 1])
            if k + 1 in d:
                res = max(res, d[k] + d[k + 1])
        return res
```

### 1577. Number of Ways Where Square of Number Is Equal to Product of Two Numbers

```python
class Solution:
    def numTriplets(self, nums1: List[int], nums2: List[int]) -> int:
        def check(nums1, nums2):
            c1, c2 = Counter(nums1), Counter(nums2)
            res = 0
            for n1 in c1:
                for n2 in c2:
                    if (n1 * n1) % n2 == 0:
                        if n1 * n1 // n2 in c2 and n1 == n2:
                            res += c1[n1] * c2[n2] * (c2[n2] - 1)
                        if n1 * n1 // n2 in c2 and n1 != n2:
                            res += c1[n1] * c2[n2] * c2[n1 * n1 // n2]
            return res // 2
        return check(nums1, nums2) + check(nums2, nums1)
```

### 1647. Minimum Deletions to Make Character Frequencies Unique

```python
class Solution:
    def minDeletions(self, s: str) -> int:
        c = Counter(s)
        seen = set()
        res = 0
        for v in sorted(c.values()):
            if v not in seen:
                seen.add(v)
            else:
                while v in seen and v - 1 >= 0:
                    v -= 1
                    res += 1
                seen.add(v)
        return res
```

### 890. Find and Replace Pattern

```python
class Solution:
    def findAndReplacePattern(self, words: List[str], pattern: str) -> List[str]:
        def check(w):
            c1, c2 = Counter(), Counter()
            for a, b in zip(w, pattern):
                c1[a] = b 
                c2[b] = a 
            res1, res2 = '', ''
            for c in w:
                res1 += c1[c]
            for c in pattern:
                res2 += c2[c]
            return res1 == pattern and res2 == w
        res = []
        for w in words:
            if check(w):
                res.append(w)
        return res
```

### 2225. Find Players With Zero or One Losses

```python
class Solution:
    def findWinners(self, matches: List[List[int]]) -> List[List[int]]:
        c_lose = Counter()
        s = set()
        for a, b in matches:
            c_lose[b] += 1
            s.add(a)
            s.add(b)
        res1, res2 = [], []
        for i in s:
            if i not in c_lose:
                res1.append(i)
            elif c_lose[i] == 1:
                res2.append(i)
        return [sorted(res1), sorted(res2)]
```

### 2244. Minimum Rounds to Complete All Tasks

```python
class Solution:
    def minimumRounds(self, tasks: List[int]) -> int:
        c = Counter(tasks)
        res = 0
        for v in c.values():
            if v % 3 == 0:
                res += v // 3
            else:
                if v > 3:
                    res += v // 3 + 1
                else:
                    if v == 1:
                        return -1
                    else:
                        res += 1
        return res
```

### 2482. Difference Between Ones and Zeros in Row and Column

```python
class Solution:
    def onesMinusZeros(self, grid: List[List[int]]) -> List[List[int]]:
        R, C = len(grid), len(grid[0])
        rows, cols = defaultdict(int), defaultdict(int)
        for r in range(R):
            for c in range(C):
                if grid[r][c] == 1:
                    rows[r] += 1
                    cols[c] += 1
        dp = [[0] * C for r in range(R)]
        for r in range(R):
            for c in range(C):
                dp[r][c] = rows[r] + cols[c] - (R + C - (rows[r] + cols[c]))
        return dp
```

### 1224. Maximum Equal Frequency

```python
class Solution:
    def maxEqualFreq(self, nums: List[int]) -> int:
        n = len(nums)
        count = Counter(nums)
        countFreq = Counter(count.values())
        for i in range(n - 1, -1, -1):
            if len(countFreq) == 1:
                freq, num = list(countFreq.items())[0]
                if freq == 1 or num == 1:
                    return i + 1
            elif len(countFreq) == 2:
                lst = list(sorted(countFreq.items()))  
                freq1, num1 = lst[0]
                freq2,num2 = lst[1]
                if (freq1 == 1 and num1 == 1) or (num2 == 1 and freq2 == freq1 + 1):
                    return i + 1
            f = count[nums[i]]
            count[nums[i]] -= 1
            if count[nums[i]] == 0:
                del count[nums[i]]
            countFreq[f] -= 1
            if countFreq[f] == 0:
                del countFreq[f]
            if f - 1 > 0:
                countFreq[f - 1] += 1
        return 2
```
