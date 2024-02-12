## template 1: loop one array/string

```python
def fn(arr):
    res, l, r = 0, 0, len(arr) - 1
    while l < r:
        if CONDITION: # according to problem
            l += 1
        else:
            r -= 1  
    return res
```

## template 2: loop two arrays/strings

```python
def fn(arr1, arr2):
    i = j = res = 0
    while i < len(arr1) and j < len(arr2):
        # some code
        if CONDITION:
            i += 1
        else:
            j += 1
    while i < len(arr1):
        # some code
        i += 1
    while j < len(arr2):
        # some code
        j += 1
    return res
```

* [167. Two Sum II - Input Array Is Sorted](#167-Two-Sum-II---Input-Array-Is-Sorted)
* [1099. Two Sum Less Than K](#1099-Two-Sum-Less-Than-K)
* [15. 3Sum](#15-3Sum)
* [16. 3Sum Closest](#16-3Sum-Closest)
* [259. 3Sum Smaller](#259-3Sum-Smaller)
* [18. 4Sum](#18-4Sum)
* [11. Container With Most Water](#11-Container-With-Most-Water)

### 1. Two Sum

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        d = {}
        for i, v in enumerate(nums):
            res = target - v
            if res in d:
                return [d[res], i]
            d[v] = i
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

### 167. Two Sum II - Input Array Is Sorted

```python
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        l, r = 0, len(numbers) - 1
        while l < r:
            res = numbers[l] + numbers[r]
            if res > target:
                r -= 1
            elif res < target:
                l += 1
            else:
                return [l + 1, r + 1]
```

### 15. 3Sum

```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        n, res = len(nums), []
        for i in range(n - 2):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            if sum(nums[i: i + 3]) > 0:
                break
            if nums[i] + nums[-1] + nums[-2] < 0:
                continue
            l, r = i + 1, n - 1
            while l < r:
                three = sum([nums[i], nums[l], nums[r]])
                if three > 0:
                    r -= 1
                elif three < 0:
                    l += 1
                else:
                    res.append((nums[i], nums[l], nums[r]))
                    l += 1
                    while l < r and nums[l] == nums[l - 1]:
                        l += 1
                    r -= 1
                    while l < r and nums[r] == nums[r + 1]:
                        r -= 1
        return res
```

### 16. 3Sum Closest

```python
class Solution:
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        nums.sort()
        n, res = len(nums), inf
        for i in range(n):
            l, r = i + 1, n - 1
            while l < r:
                three = nums[i] + nums[l] + nums[r]
                if abs(three - target) < abs(res - target):
                    res = three
                if three > target:
                    r -= 1
                else:
                    l += 1
        return res
```

### 18. 4Sum

```python
class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        nums.sort()
        n, res = len(nums), []
        for j in range(n - 3):
            if j > 0 and nums[j] == nums[j - 1]:
                continue
            if sum(nums[j: j + 4]) > target:
                break
            if nums[j] + nums[-1] + nums[-2] + nums[-3] < target:
                continue
            for i in range(j + 1, n):
                if i > j + 1 and nums[i] == nums[i - 1]:
                    continue
                l, r = i + 1, n - 1
                while l < r:
                    four = sum([nums[j], nums[i], nums[l], nums[r]])
                    if four > target:
                        r -= 1
                    elif four < target:
                        l += 1
                    else:
                        res.append((nums[j], nums[i], nums[l], nums[r]))
                        l += 1
                        while l < r and nums[l] == nums[l - 1]:
                            l += 1
                        r -= 1
                        while l < r and nums[r] == nums[r + 1]:
                            r -= 1
        return res
```

### 259. 3Sum Smaller

```python
class Solution:
    def threeSumSmaller(self, nums: List[int], target: int) -> int:
        nums.sort()
        res, n = 0, len(nums)
        for i in range(n):
            if nums[i] >= 1 / 3 * target:
                continue
            l, r = i + 1, n - 1
            while l < r:
                three = nums[i] + nums[l] + nums[r]
                if three >= target:
                    r -= 1
                else: # [-1, -1, -1, 0, 1, 1, 1], 2
                    res += r - l
                    l += 1
        return res
```

### 1099. Two Sum Less Than K

```python
class Solution:
    def twoSumLessThanK(self, nums: List[int], k: int) -> int:
        l, r, res = 0, len(nums) - 1, -1
        nums.sort()
        while l < r:
            two = nums[l] + nums[r]
            if two >= k:
                r -= 1
            else:
                res = max(res, two)
                l += 1
        return res
```

### 11. Container With Most Water

```python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        res, l, r = 0, 0, len(height) - 1
        while l < r:
            res = max(res, min(height[l], height[r]) * (r - l))
            if height[l] < height[r]:
                l += 1
            else:
                r -= 1
        return res
```

### 42. Trapping Rain Water

```python
class Solution:
    def trap(self, height: List[int]) -> int:
        res, l, r = 0, 0, len(height) - 1
        prefix, suffix = 0, 0
        while l <= r:
            prefix = max(prefix, height[l])
            suffix = max(suffix, height[r])
            if prefix < suffix:
                res += prefix - height[l]
                l += 1
            else:
                res += suffix - height[r]
                r -= 1
        return res
```

### 392. Is Subsequence

```python
class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        len_s, len_t = len(s), len(t)
        i, j = 0, 0
        for j in range(len_t):
            if i >= len_s:
                break
            if t[j] == s[i]:
                i += 1
        return i == len_ss
```

### 524. Longest Word in Dictionary through Deleting

```python
class Solution:
    def findLongestWord(self, s: str, dictionary: List[str]) -> str:
        res = ""
        for t in dictionary:
            i = j = 0
            while i < len(t) and j < len(s):
                if t[i] == s[j]:
                    i += 1
                j += 1
            if i == len(t):
                if len(t) > len(res) or (len(t) == len(res) and t < res):
                    res = t
        return res
```

### 792. Number of Matching Subsequences

```python
class Solution:
    def numMatchingSubseq(self, s: str, words: List[str]) -> int:
        def check(word, s):
            n, m = len(word), len(s)
            i, j = 0, 0 
            while i < n and j < m:
                if word[i] == s[j]:
                    i += 1
                j += 1
            return i == n

        res = 0
        d = Counter(words)
        for word, cnt in d.items():
            if check(word, s):
                res += cnt
        return res
```

### 1023. Camelcase Matching

```python
class Solution:
    def camelMatch(self, queries: List[str], pattern: str) -> List[bool]:
        def check(word, s):
            n, m = len(word), len(s)
            i, j = 0, 0 
            while i < n and j < m:
                if word[i] == s[j]:
                    i += 1
                j += 1
            return i == n

        upper = set(list(ascii_uppercase))
        res = [0] * len(queries)
        patternCount = sum(c in upper for c in pattern)
        for i, q in enumerate(queries):
            capitalCount = sum(c in upper for c in q)
            if capitalCount != patternCount:
                res[i] = False
                continue
            if check(pattern, q):
                res[i] = True
            else:
                res[i] = False
        return res
```

### 2570. Merge Two 2D Arrays by Summing Values

```python
class Solution:
    def mergeArrays(self, nums1: List[List[int]], nums2: List[List[int]]) -> List[List[int]]:
        d = defaultdict(int)
        for i, v in nums1 + nums2:
            d[i] += v
        return sorted([(k, v) for k, v in d.items()])
```

### 541. Reverse String II

```python
class Solution:
    def reverseStr(self, s: str, k: int) -> str:
        n, res = len(s), ''
        for i in range(0, n, 2 * k):
            res += s[i: i + k][::-1] + s[i + k: i + 2 * k]
        return res
```

### 2540. Minimum Common Value

```python
class Solution:
    def getCommon(self, nums1: List[int], nums2: List[int]) -> int:
        m, n = len(nums1), len(nums2)
        i, j = 0, 0
        while i < m and j < n:
            if nums1[i] == nums2[j]:
                return nums1[i]
            elif nums1[i] < nums2[j]:
                i += 1
            else:
                j += 1
        return -1
```

### 1768. Merge Strings Alternately

```python
class Solution:
    def mergeAlternately(self, word1: str, word2: str) -> str:
        q1, q2 = deque(list(word1)), deque(list(word2))
        res = ''
        while q1 and q2:
            c1, c2 = q1.popleft(), q2.popleft()
            res += c1 + c2 
        res += ''.join(q1) if q1 else ''.join(q2)
        return res
```

### 2697. Lexicographically Smallest Palindrome

```python
class Solution:
    def makeSmallestPalindrome(self, s: str) -> str:
        s = list(s)
        l, r = 0, len(s) - 1
        while l < r:
            if s[l] < s[r]:
                s[r] = s[l]
            elif s[l] > s[r]:
                s[l] = s[r]
            l, r = l + 1, r - 1
        return ''.join(s)
```

### 2825. Make String a Subsequence Using Cyclic Increments

```python
class Solution:
    def canMakeSubsequence(self, str1: str, str2: str) -> bool:
        j = 0
        for i, c in enumerate(str1):
            b = chr(ord(c) + 1) if c != 'z' else 'a'
            if str2[j] in [b, c]:
                 j += 1
                 if j == len(str2):
                     return True
        return False
```

### 392. Is Subsequence

```python
class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        i, j = 0, 0
        l_s, l_t = len(s), len(t)
        while j < l_t:
            if i < l_s and s[i] == t[j]:
                i += 1
            j += 1
        if i == l_s:
            return True
        return False
```

### 881. Boats to Save People

```python
class Solution:
    def numRescueBoats(self, people: List[int], limit: int) -> int:
        people.sort()
        l, r = 0, len(people) - 1
        res = 0
        while l <= r:
            if people[l] + people[r] <= limit:
                l += 1
            r -= 1  
            res += 1
        return res
```