# Hash

## 2 sum
- The idea is 2 sum, enumerate right index and check left index, and use s hash table to store the checked state.

* [1. Two Sum](#1-two-sum)  
* [2441. Largest Positive Integer That Exists With Its Negative](#2441-largest-positive-integer-that-exists-with-its-negative)
* [1512. Number of Good Pairs](#1512-number-of-good-pairs)
* [2001. Number of Pairs of Interchangeable Rectangles](#2001-number-of-pairs-of-innterchangeable-rectangles)
* [1128. Number of Equivalent Domino Pairs](#1128-number-of-equivalent-domino-pairs)

* [219. Contains Duplicate II](#219-contains-duplicate-ii)
* [121. Best Time to Buy and Sell Stock](#121-best-time-to-buy-and-sell-stock)
* [2016. Maximum Difference Between Increasing Elements](#2016-maximum-difference-between-increasing-elements)
* [2260. Minimum Consecutive Cards to Pick Up](#2260-minimum-consecutive-cards-to-pick-up)
* [2815. Max Pair Sum in an Array](#2815-max-pair-sum-in-an-array)

* [167. Two Sum II - Input Array Is Sorted](#167-two-sum-ii---input-array-is-sorted)
* [170. Two Sum III - Data structure design](#170-two-sum-iii---data-structure-design)
* [653. Two Sum IV - Input is a BST](#653-two-sum-iv---input-is-a-bst)
* [1214. Two Sum BSTs](#1214-two-sum-bsts)
* [2964. Number of Divisible Triplet Sums](#2964-number-of-divisible-triplet-sums)

* [2342. Max Sum of a Pair With Equal Sum of Digits](#2342-max-sum-of-a-pair-with-equal-sum-of-digits)
* [1679. Max Number of K-Sum Pairs](#1679-max-number-of-k-sum-pairs)
* [16-24. Pairs With Sum LCCI](#16-24-pairs-with-sum-lcci)
* [3623. Count Number of Trapezoids I](#3623-count-number-of-trapezoids-i)
* [624. Maximum Distance in Arrays](#624-maximum-distance-in-arrays)

* [2364. Count Number of Bad Pairs](#2364-count-number-of-bad-pairs)
* [1014. Best Sightseeing Pair](#1014-best-sightseeing-pair)
* [3371. Identify the Largest Outlier in an Array]()
* [2748. Number of Beautiful Pairs](#2748-number-of-beautiful-pairs)
* [1814. Count Nice Pairs in an Array](#1814-count-nice-pairs-in-an-array)

* [3584. Maximum Product of First and Last Elements of a Subsequence](#3584-maximum-product-of-first-and-last-elements-of-a-subsequence)
* [2905. Find Indices With Index and Value Difference II](#2905-find-indices-with-index-and-value-difference-ii)
* [1010. Pairs of Songs With Total Durations Divisible by 60](#1010-pairs-of-songs-with-total-durations-divisible-by-60)
* [3185. Count Pairs That Form a Complete Day II](#3185-count-pairs-that-form-a-complete-day-ii)
* [454. 4Sum II](#454-4sum-ii)

* [2874. Maximum Value of an Ordered Triplet II](#2874-maximum-value-of-an-ordered-triplet-ii)
* [2971. Find Polygon With the Largest Perimeter](#2971-find-polygon-with-the-largest-perimeter)
* [533. Lonely Pixel II](#533-lonely-pixel-ii)

## Diagonal

* [1329. Sort the Matrix Diagonally]()
* [3446. Sort Matrix by Diagonals](#3446-sort-matrix-by-diagonals)
* [498. Diagonal Traverse](#498-diagonal-traverse)

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

```js
var twoSum = function(nums, target) {
    const m = new Map();
    for (let i = 0; i < nums.length; i ++) {
        const res = target - nums[i];
        if (m.has(res)) {
            return [m.get(res), i];
        }
        m.set(nums[i], i);
    }
    return [];
};
```

```js
var twoSum = function(nums, target) {
    const obj = {};
    for (let i = 0; i < nums.length; i++) {
        const res = target - nums[i];
        if (res in obj) {
            return [obj[res], i];
        }
        obj[nums[i]] = i;
    }
    return [];
};
```

```java
class Solution {
    public int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> m = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            int res = target - nums[i];
            if (m.containsKey(res)) {
                return new int[]{m.get(res), i};
            }
            m.put(nums[i], i);
        }
        return new int[0];
    }
}
```

### 2441. Largest Positive Integer That Exists With Its Negative

```python
class Solution:
    def findMaxK(self, nums: List[int]) -> int:
        s, res = set(), -1
        for i, n in enumerate(nums):
            ans = -n 
            if ans in s:
                res = max(res, abs(ans))
            s.add(n)
        return res 
```


### 1512. Number of Good Pairs

```python
class Solution:
    def numIdenticalPairs(self, nums: List[int]) -> int:
        d, res = defaultdict(int), 0
        for i, n in enumerate(nums):
            res += d[n]
            d[n] += 1
        return res 
```

### 2001. Number of Pairs of Interchangeable Rectangles

```python
class Solution:
    def interchangeableRectangles(self, rectangles: List[List[int]]) -> int:
        d, res = defaultdict(int), 0
        for w, h in rectangles:
            res += d[w / h]
            d[w / h] += 1
        return res 
```

### 1128. Number of Equivalent Domino Pairs

```python
class Solution:
    def numEquivDominoPairs(self, dominoes: List[List[int]]) -> int:
        d, res = defaultdict(int), 0
        for a, b in dominoes:
            if a > b:
                a, b = b, a 
            res += d[(a, b)]
            d[(a, b)] += 1
        return res 
```

### 219. Contains Duplicate II

```python
class Solution:
    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        d = defaultdict(int)
        for i, n in enumerate(nums):
            if n in d and i - d[n] <= k:
                return True
            d[n] = i 
        return False
```


### 121. Best Time to Buy and Sell Stock

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        res, lowest = 0, prices[0]
        for p in prices:
            lowest = min(lowest, p)
            res = max(res, p - lowest)
        return res 
```

### 2016. Maximum Difference Between Increasing Elements

```python
class Solution:
    def maximumDifference(self, nums: List[int]) -> int:
        res, lowest = -1, inf 
        for i, n in enumerate(nums):
            lowest = min(lowest, n)
            res = max(res, n - lowest)
        return res if res != 0 else -1
```

### 2260. Minimum Consecutive Cards to Pick Up

```python
class Solution:
    def minimumCardPickup(self, cards: List[int]) -> int:
        d, res = defaultdict(int), -1
        for i, n in enumerate(cards):
            if n in d:
                res = max(res, i - d[n] + 1)
            d[n] = i 
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

### 167. Two Sum II - Input Array Is Sorted

```python
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        l, r = 0, len(numbers) - 1
        while l <= r:
            total = numbers[l] + numbers[r]
            if total > target:
                r -= 1
            elif total < target:
                l += 1
            else:
                return [l + 1, r + 1]
```

### 170. Two Sum III - Data structure design

Design a data structure that accepts a stream of integers and checks if it has a pair of integers that sum up to a particular value.

Implement the TwoSum class:

TwoSum() Initializes the TwoSum object, with an empty array initially.
void add(int number) Adds number to the data structure.
boolean find(int value) Returns true if there exists any pair of numbers whose sum is equal to value, otherwise, it returns false.
 

Example 1:

Input
["TwoSum", "add", "add", "add", "find", "find"]
[[], [1], [3], [5], [4], [7]]
Output
[null, null, null, null, true, false]

Explanation
TwoSum twoSum = new TwoSum();
twoSum.add(1);   // [] --> [1]
twoSum.add(3);   // [1] --> [1,3]
twoSum.add(5);   // [1,3] --> [1,3,5]
twoSum.find(4);  // 1 + 3 = 4, return true
twoSum.find(7);  // No two integers sum up to 7, return false
 

Constraints:

-105 <= number <= 105
-231 <= value <= 231 - 1
At most 104 calls will be made to add and find.

```python
class TwoSum:

    def __init__(self):
        self.arr = []

    def add(self, number: int) -> None:
        self.arr.append(number)

    def find(self, value: int) -> bool:
        s = set()
        for i, n in enumerate(self.arr):
            res = value - n 
            if res in s:
                return True
            s.add(n)
        return False
```

### 653. Two Sum IV - Input is a BST

```python
class Solution:
    def findTarget(self, root: Optional[TreeNode], k: int) -> bool:
        s = set()
        def dfs(node):
            if node:
                res = k - node.val
                if res in s:
                    return True
                s.add(node.val)
                return dfs(node.left) or dfs(node.right)
            return False
        return dfs(root)
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

### 1010. Pairs of Songs With Total Durations Divisible by 60

```python
class Solution:
    def numPairsDivisibleBy60(self, time: List[int]) -> int:
        d, res = defaultdict(int), 0
        for i, n in enumerate(time):
            ans = 60 - n % 60
            res += d[ans] + d[ans - 60]
            d[n % 60] += 1
        return res 
```

### 3185. Count Pairs That Form a Complete Day II

```python
class Solution:
    def countCompleteDayPairs(self, hours: List[int]) -> int:
        d, res = defaultdict(int), 0
        for i, n in enumerate(hours):
            ans = 24 - n % 24
            res += d[ans] + d[ans - 24]
            d[n % 24] += 1
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

### 3371. Identify the Largest Outlier in an Array

```python
class Solution:
    def getLargestOutlier(self, nums: List[int]) -> int:
        # x + 2y = total
        res, total = -inf, sum(nums)
        d = Counter(nums)
        for i, x in enumerate(nums):
            d[x] -= 1
            y = (total - x) / 2
            if y in d and d[y] > 0:    
                res = max(res, x)
            d[x] += 1
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
            res += d[val]
            d[val] += 1
        return res % mod
```

### 3584. Maximum Product of First and Last Elements of a Subsequence

```python
class Solution:
    def maximumProduct(self, nums: List[int], m: int) -> int:
        res = mx = -inf
        mn = inf
        for i in range(m - 1, len(nums)):
            y = nums[i - m + 1]
            mn = min(mn, y)
            mx = max(mx, y)
            x = nums[i]
            res = max(res, x * mx, x * mn)
        return res 
```

### 2905. Find Indices With Index and Value Difference II

```python
class Solution:
    def findIndices(self, nums: List[int], indexDifference: int, valueDifference: int) -> List[int]:
        max_idx = min_idx = 0
        for j in range(indexDifference, len(nums)):
            i = j - indexDifference
            if nums[i] > nums[max_idx]:
                max_idx = i 
            elif nums[i] < nums[min_idx]:
                min_idx = i 
            if nums[max_idx] - nums[j] >= valueDifference:
                return [max_idx, j]
            if nums[j] - nums[min_idx] >= valueDifference:
                return [min_idx, j]
        return [-1, -1]
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

```python
class Solution:
    def twoSumBSTs(self, root1: Optional[TreeNode], root2: Optional[TreeNode], target: int) -> bool:
        s = set()
        def dfs(node):
            if node:
                s.add(node.val)
                dfs(node.left)
                dfs(node.right)
        dfs(root1)

        def dfs2(node):
            if node:
                res = target - node.val
                return res in s or dfs2(node.left) or dfs2(node.right)
            return False

        dfs(root1)
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

### 2964. Number of Divisible Triplet Sums

Given a 0-indexed integer array nums and an integer d, return the number of triplets (i, j, k) such that i < j < k and (nums[i] + nums[j] + nums[k]) % d == 0.
 

Example 1:

Input: nums = [3,3,4,7,8], d = 5
Output: 3
Explanation: The triplets which are divisible by 5 are: (0, 1, 2), (0, 2, 4), (1, 2, 4).
It can be shown that no other triplet is divisible by 5. Hence, the answer is 3.
Example 2:

Input: nums = [3,3,3,3], d = 3
Output: 4
Explanation: Any triplet chosen here has a sum of 9, which is divisible by 3. Hence, the answer is the total number of triplets which is 4.
Example 3:

Input: nums = [3,3,3,3], d = 6
Output: 0
Explanation: Any triplet chosen here has a sum of 9, which is not divisible by 6. Hence, the answer is 0.
 

Constraints:

1 <= nums.length <= 1000
1 <= nums[i] <= 109
1 <= d <= 109

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

### 1679. Max Number of K-Sum Pairs

```python
class Solution:
    def maxOperations(self, nums: List[int], k: int) -> int:
        d, res = defaultdict(int), 0
        for i, n in enumerate(nums):
            ans = k - n 
            if ans in d and d[ans] > 0:
                res += 1
                d[ans] -= 1
            else:
                d[n] += 1
        return res 
```

### 16-24. Pairs With Sum LCCI

```python
class Solution:
    def pairSums(self, nums: List[int], target: int) -> List[List[int]]:
        d, res = defaultdict(int), []
        for n in nums:
            ans = target - n 
            if ans in d and d[ans] > 0:
                res.append([ans, n])
                d[ans] -= 1
            else:
                d[n] += 1
        return res 
```

### 3623. Count Number of Trapezoids I

```python
class Solution:
    def countTrapezoids(self, points: List[List[int]]) -> int:
        mod = 10 ** 9 + 7
        d = defaultdict(int)
        for x, y in points:
            d[y] += 1

        res = 0
        b = [v for v in d.values() if v >= 2]
        b = [x * (x - 1) // 2 for x in b]
        total = sum(b)
        for a in b:
            x = total - a 
            res += a * x
            total -= a
        return res % mod

class Solution:
    def countTrapezoids(self, points: List[List[int]]) -> int:
        mod = 10 ** 9 + 7
        d = Counter(y for x, y in points)
        res = s = 0
        for x in d.values():
            y = x * (x - 1) // 2
            res += s * y 
            s += y
        return res % mod
```

### 624. Maximum Distance in Arrays

```python
class Solution:
    def maxDistance(self, arrays: List[List[int]]) -> int:
        res = 0
        mn, mx = inf, -inf 
        for a in arrays:
            res = max(res, mx - a[0], a[-1] - mn)
            mn = min(mn, a[0])
            mx = max(mx, a[-1])
        return res 
```

### 2364. Count Number of Bad Pairs

```python
class Solution:
    def countBadPairs(self, nums: List[int]) -> int:
        # j - i != nums[j] - nums[i] => i - nums[i] = j - nums[j]
        d, res = defaultdict(int), 0
        for i, n in enumerate(nums):
            ans = i - n 
            if ans not in d:
                res += i 
            else:
                res += i - d[ans]
            d[ans] += 1
        return res 
```

### 533. Lonely Pixel II

```python
class Solution:
    def findBlackPixel(self, picture: List[List[str]], target: int) -> int:
        rows, cols = Counter(), Counter()
        R, C = len(picture), len(picture[0])
        for r in range(R):
            for c in range(C):
                if picture[r][c] == 'B':
                    rows[r] += 1
                    cols[c] += 1
        d = defaultdict(bool)
        for r1, row1 in enumerate(picture):
            for r2, row2 in enumerate(picture):
                d[(r1, r2)] = row1 == row2
                
        res = 0
        for r in range(R):
            for c in range(C):
                if rows[r] == cols[c] == target:
                    state = True
                    for i in range(R):
                        if picture[i][c] == 'B':
                            if not d[(r, i)]:
                                state = False
                                break 
                    if state:
                        res += 1
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

### 1329. Sort the Matrix Diagonally

```python
class Solution:
    def diagonalSort(self, mat: List[List[int]]) -> List[List[int]]:
        R, C, d = len(mat), len(mat[0]), defaultdict(list)
        for r in range(R):
            for c in range(C):
                d[r - c].append(mat[r][c])
        for k, v in d.items():
            d[k].sort()
        
        for r in range(R):
            for c in range(C):
                mat[r][c] = d[r - c].pop(0)
        return mat
```

### 3446. Sort Matrix by Diagonals

```python
class Solution:
    def sortMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        n, d = len(grid), defaultdict(list)
        for r in range(n):
            for c in range(n):
                d[r - c].append(grid[r][c])
        for k, v in d.items():
            d[k].sort()
        
        for r in range(n):
            for c in range(n):
                if r < c:
                    grid[r][c] = d[r - c].pop(0)
                else:
                    grid[r][c] = d[r - c].pop()
        return grid
```

### 498. Diagonal Traverse

```python
class Solution:
    def findDiagonalOrder(self, mat: List[List[int]]) -> List[int]:
        R, C, d = len(mat), len(mat[0]), defaultdict(list)
        for r in range(R):
            for c in range(C):
                d[r + c].append(mat[r][c])
        
        res = []
        for k, v in d.items():
            if k % 2 == 0:
                res.extend(v[::-1])
            else:
                res.extend(v)
        return res 
```