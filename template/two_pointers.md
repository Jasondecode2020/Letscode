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

* [344. Reverse String](#344-reverse-string)
* [125. Valid Palindrome](#125-valid-palindrome)
* [1750. Minimum Length of String After Deleting Similar Ends](#1750-minimum-length-of-string-after-deleting-similar-ends)
* [977. Squares of a Sorted Array](#977-squares-of-a-sorted-array)
* [2105. Watering Plants II](#2105-watering-plants-ii)

* [167. Two Sum II - Input Array Is Sorted](#167-Two-Sum-II---Input-Array-Is-Sorted)
* [633. Sum of Square Numbers](#633-sum-of-square-numbers)
* [1471. The k Strongest Values in an Array](#1471-the-k-strongest-values-in-an-array)
* [658. Find K Closest Elements](#658-find-k-closest-elements)
* [15. 3Sum](#15-3Sum)

* [16. 3Sum Closest](#16-3Sum-Closest)
* [18. 4Sum](#18-4Sum)
* [259. 3Sum Smaller](#259-3Sum-Smaller)
* [2824. Count Pairs Whose Sum is Less than Target](#259-3sum-smaller)
* [1099. Two Sum Less Than K](#1099-Two-Sum-Less-Than-K)

* [11. Container With Most Water](#11-Container-With-Most-Water)
* [1679. Max Number of K-Sum Pairs](#1679-max-number-of-k-sum-pairs)
* [881. Backspace String Compare](#881-boats-to-save-people)
* [360. Sort Transformed Array](#360-sort-transformed-array)
* [2422. Merge Operations to Turn Array Into a Palindrome](#2422-merge-operations-to-turn-array-into-a-palindrome)

* [611. Valid Triangle Number](#611-valid-triangle-number)
* [844. Backspace String Compare](#844-backspace-string-compare)
* [1813. Sentence Similarity III](#1813-sentence-similarity-iii)
* [1616. Split Two Strings to Make Palindrome](#1616-split-two-strings-to-make-palindrome)
* [1498. Number of Subsequences That Satisfy the Given Sum Condition](#1498-number-of-subsequences-that-satisfy-the-given-sum-condition)

## same directon

* [1574. Shortest Subarray to be Removed to Make Array Sorted](#1574-shortest-subarray-to-be-removed-to-make-array-sorted)
* [2972. Count the Number of Incremovable Subarrays II](#2972-count-the-number-of-incremovable-subarrays-ii)

## in-place

* [26. Remove Duplicates from Sorted Array](#26-remove-duplicates-from-sorted-array)
* [80. Remove Duplicates from Sorted Array II](#80-remove-duplicates-from-sorted-array-ii)
* [283. Move Zeroes](#283-move-zeroes)
* [905. Sort Array By Parity](#905-sort-array-by-parity)
* [922. Sort Array By Parity II](#922-sort-array-by-parity-ii)

### 344. Reverse String

```python 
class Solution:
    def reverseString(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        l, r = 0, len(s) - 1
        while l < r:
            s[r], s[l] = s[l], s[r]
            l += 1
            r -= 1
```

### 125. Valid Palindrome

```python 
class Solution:
    def isPalindrome(self, s: str) -> bool:
        res = ''
        for c in s:
            if c.isalnum():
                res += c.lower()
        l, r = 0, len(res) - 1
        while l < r:
            if res[l] != res[r]:
                return False 
            l += 1
            r -= 1
        return True
```

### 1750. Minimum Length of String After Deleting Similar Ends

```python 
class Solution:
    def minimumLength(self, s: str) -> int:
        n = len(s)
        l, r = 0, n - 1
        while l < r:
            if s[l] == s[r]:
                c = s[l]
                while l < n and s[l] == c: 
                    l += 1
                while r >= 0 and s[r] == c:
                    r -= 1
            else:
                break
        return max(0, r - l + 1)
```

### 977. Squares of a Sorted Array

```python 
class Solution:
    def sortedSquares(self, nums: List[int]) -> List[int]:
        nums.sort(key = lambda x: x ** 2)
        nums = [i ** 2 for i in nums]
        return nums
```

### 2105. Watering Plants II

```python 
class Solution:
    def minimumRefill(self, plants: List[int], capacityA: int, capacityB: int) -> int:
        l, r = 0, len(plants) - 1
        originA, originB = capacityA, capacityB
        res = 0
        while l <= r:
            if l == r:
                if max(capacityA, capacityB) < plants[l]:
                    res += 1
                else:
                    break
            else:
                if capacityA < plants[l]:
                    capacityA = originA
                    res += 1
                capacityA -= plants[l]
                    
                if capacityB < plants[r]:
                    capacityB = originB 
                    res += 1
                capacityB -= plants[r]   
            l += 1
            r -= 1
        return res
```

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

### 633. Sum of Square Numbers

```python 
class Solution:
    def judgeSquareSum(self, c: int) -> bool:
        a, b = 0, isqrt(c)
        while a <= b:
            s = a * a + b * b
            if s == c:
                return True
            if s < c:
                a += 1
            else:
                b -= 1
        return False
```

### 1471. The k Strongest Values in an Array

```python 
class Solution:
    def getStrongest(self, arr: List[int], k: int) -> List[int]:
        n = len(arr)
        arr.sort()
        median = arr[(n - 1) // 2]
        res = []
        for n in arr:
            res.append((abs(n - median), n))
        res.sort(reverse = True)
        return [item[1] for item in res[:k]]
```

### 658. Find K Closest Elements

```python 
class Solution:
    def findClosestElements(self, arr: List[int], k: int, x: int) -> List[int]:
        l, left, right, total = 0, -1, -1, 0
        ans = inf
        for r, n in enumerate(arr):
            total += abs(n - x)
            if r - l + 1 == k:
                if total < ans:
                    ans = total
                    left, right = l, r
                total -= abs(arr[l] - x)
                l += 1
        return arr[left: right + 1]
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

### 2824. Count Pairs Whose Sum is Less than Target

```python 
class Solution:
    def countPairs(self, nums: List[int], target: int) -> int:
        n = len(nums)
        res = 0
        for i in range(n):
            for j in range(i + 1, n):
                if nums[i] + nums[j] < target:
                    res += 1
        return res

class Solution:
    def countPairs(self, nums: List[int], target: int) -> int:
        nums.sort()
        n = len(nums)
        res = 0
        l, r = 0, len(nums) - 1
        while l < r:
            s = nums[l] + nums[r]
            if s < target:
                res += r - l
                l += 1
            else:
                r -= 1
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

### 360. Sort Transformed Array

```python 
class Solution:
    def sortTransformedArray(self, nums: List[int], a: int, b: int, c: int) -> List[int]:
        res = []
        for x in nums:
            y = a * x * x + b * x + c 
            res.append(y)
        res.sort()
        return res 
```

### 2422. Merge Operations to Turn Array Into a Palindrome

```python 
class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        n = len(nums)
        l, r = 0, n - 1
        res = 0
        while l < r:
            if nums[l] == nums[r]:
                l += 1
                r -= 1
            elif nums[l] < nums[r]:
                res += 1
                nums[l + 1] += nums[l]
                l += 1
            elif nums[l] > nums[r]:
                res += 1
                nums[r - 1] += nums[r]
                r -= 1
        return res
```

### 611. Valid Triangle Number

```python 
class Solution:
    def triangleNumber(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        res = 0
        for i in range(2, n):
            l, r = 0, i - 1
            while l < r:
                if nums[l] + nums[r] > nums[i]:
                    res += r - l 
                    r -= 1
                else:
                    l += 1
        return res
```

### 2486. Append Characters to String to Make Subsequence

```python
class Solution:
    def appendCharacters(self, s: str, t: str) -> int:
        n = len(t)
        i, j = 0, 0
        while i < len(s):
            if j < n and s[i] == t[j]:
                j += 1
            i += 1
        return n - j
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

### 1877. Minimize Maximum Pair Sum in Array

```python
class Solution:
    def minPairSum(self, nums: List[int]) -> int:
        nums.sort()
        res = 0
        l, r = 0, len(nums) - 1
        while l < r:
            res = max(res, nums[l] + nums[r])
            l += 1
            r -= 1
        return res
```

### 1561. Maximum Number of Coins You Can Get

```python
class Solution:
    def maxCoins(self, piles: List[int]) -> int:
        # [1, 2, 2, 4, ,7. 8]
        piles.sort()
        l, r = 0, len(piles) - 1
        res = 0
        while l < r:
            res += piles[r - 1]
            l += 1
            r -= 2
        return res
```


### 844. Backspace String Compare

```python
class Solution:
    def backspaceCompare(self, s: str, t: str) -> bool:
        i, j = len(s) - 1, len(t) - 1
        skipS = 0
        skipT = 0
        while i >= 0 or j >= 0:
            while i >= 0:
                if s[i] == '#':
                    skipS += 1
                    i -= 1
                elif skipS > 0:
                    skipS -= 1
                    i -= 1
                else:
                    break 
            while j >= 0:
                if t[j] == '#':
                    skipT += 1
                    j -= 1
                elif skipT > 0:
                    skipT -= 1
                    j -= 1
                else:
                    break 
            if i >= 0 and j >= 0 and s[i] != t[j]:
                return False
            i -= 1
            j -= 1
        return i == j
```

### 1813. Sentence Similarity III

```python
class Solution:
    def areSentencesSimilar(self, sentence1: str, sentence2: str) -> bool:
        sentence1 = sentence1.split(' ')
        sentence2 = sentence2.split(' ')
        if len(sentence1) < len(sentence2):
            sentence1, sentence2 = sentence2, sentence1 
        l1, r1 = 0, len(sentence1) - 1
        l2, r2 = 0, len(sentence2) - 1
        while l2 <= r2:
            flag = False
            if sentence2[l2] == sentence1[l1]:
                l2 += 1
                l1 += 1
                flag = True
            if sentence2[r2] == sentence1[r1]:
                r2 -= 1
                r1 -= 1
                flag = True
            if not flag:
                return False
        return True
```

### 1616. Split Two Strings to Make Palindrome

```python 
class Solution:
    def checkPalindromeFormation(self, a: str, b: str) -> bool:
        def check(a, b):
            i, j = 0, len(b) - 1
            while i < j and a[i] == b[j]:
                i, j = i + 1, j - 1
            return i >= j or isPalindrome(a, i, j) or isPalindrome(b, i, j)

        def isPalindrome(s, i, j) -> bool:
            return s[i: j + 1] == s[i: j + 1][::-1]

        return check(a, b) or check(b, a)
```

### 1498. Number of Subsequences That Satisfy the Given Sum Condition

```python 
class Solution:
    def numSubseq(self, nums: List[int], target: int) -> int:
        nums.sort()
        l, r = 0, len(nums) - 1
        mod = 10 ** 9 + 7 
        res = 0
        while l <= r:
            ans = nums[l] + nums[r]
            if ans > target:
                r -= 1
            else:
                res += 2 ** (r - l)
                l += 1
        return res % mod
```

### 1574. Shortest Subarray to be Removed to Make Array Sorted

```python
class Solution:
    def findLengthOfShortestSubarray(self, arr: List[int]) -> int:
        n = len(arr)
        i, j = 0, n - 1
        while i + 1 < n and arr[i] <= arr[i + 1]:
            i += 1
        while j - 1 >= 0 and arr[j] >= arr[j - 1]:
            j -= 1
        res = min(n - i - 1, j)
        if i >= j:
            return 0
        
        r = j 
        for l in range(i + 1):
            while r < n and arr[l] > arr[r]:
                r += 1
            res = min(res, r - l - 1)
        return res 
```

### 2972. Count the Number of Incremovable Subarrays II

```python 
class Solution:
    def incremovableSubarrayCount(self, nums: List[int]) -> int:
        n = len(nums)
        i = 0
        while i + 1 < n and nums[i] < nums[i + 1]:
            i += 1
        if i == n - 1:
            return n * (n + 1) // 2
        res = i + 2 
        j = n - 1
        while j == n - 1 or nums[j] < nums[j + 1]:
            while i >= 0 and nums[i] >= nums[j]:
                i -= 1
            res += i + 2
            j -= 1
        return res 
```

### 26. Remove Duplicates from Sorted Array

```python 
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        k = 1
        for i in range(1, len(nums)):
            if nums[i] != nums[i - 1]:
                nums[k] = nums[i]
                k += 1
        return k
```

### 80. Remove Duplicates from Sorted Array II

```python 
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        i = 0
        for n in nums:
            if i < 2 or nums[i - 2] != n:
                nums[i] = n 
                i += 1
        return i
```

### 283. Move Zeroes

```python 
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        l = 0
        for n in nums:
            if n:
                nums[l] = n 
                l += 1
        for i in range(l, len(nums)):
            nums[i] = 0
```

### 905. Sort Array By Parity

```python 
class Solution:
    def sortArrayByParity(self, nums: List[int]) -> List[int]:
        l, r = 0, len(nums) - 1
        while l < r: 
            if nums[l] % 2 == 0:
                l += 1
            elif nums[r] % 2 == 1:
                r -= 1
            else:
                nums[l], nums[r] = nums[r], nums[l]
                l += 1
                r -= 1
        return nums
```

### 922. Sort Array By Parity II

```python 
class Solution:
    def sortArrayByParityII(self, nums: List[int]) -> List[int]:
        i, j = 0, 1
        while i < len(nums):
            if nums[i] % 2 == 0:
                i += 2
            elif nums[j] % 2 == 1:
                j += 2 
            else:
                nums[i], nums[j] = nums[j], nums[i]
                i += 2
                j += 2
        return nums
```

### 1163. Last Substring in Lexicographical Order

```python 
class Solution:
    def lastSubstring(self, s: str) -> str:
        i, j, k = 0, 1, 0
        while j + k < len(s):
            if s[i + k] == s[j + k]:
                k += 1
            elif s[i + k] < s[j + k]:
                t = i 
                i = j 
                j = max(j + 1, t + k + 1)
                k = 0
            else:
                j += 1
                k = 0
        return s[i:]
```


### 3403. Find the Lexicographically Largest String From the Box I

```python 
class Solution:
    def answerString(self, word: str, numFriends: int) -> str:
        s = word 
        i, j, k = 0, 1, 0
        while j + k < len(s):
            if s[i + k] == s[j + k]:
                k += 1
            elif s[i + k] < s[j + k]:
                t = i 
                i = j 
                j = max(j + 1, t + k + 1)
                k = 0
            else:
                j += 1
                k = 0
        mx_len = len(word) - numFriends + 1
        if mx_len == len(word):
            return s 
        return s[i:][:mx_len]
```