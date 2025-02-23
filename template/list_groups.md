## template 1: start without condition

### 1446

```python
class Solution:
    def maxPower(self, s: str) -> int:
        res, i, n = 0, 0, len(s)
        while i < n:
            j = i + 1
            while j < n and s[j] == s[j - 1]:
                j += 1
            res = max(res, j - i)
            i = j 
        return res
```

## template 2: start with condition

### 2760

```python
class Solution:
    def longestAlternatingSubarray(self, nums: List[int], threshold: int) -> int:
        res, i, n = 0, 0, len(nums)
        while i < n:
            if nums[i] % 2 or nums[i] > threshold:
                i += 1
                continue
            start = i
            while i < n - 1 and nums[i] % 2 != nums[i + 1] % 2 and nums[i + 1] <= threshold:
                i += 1
            res = max(res, i - start + 1)
            i += 1
        return res
```

### list groups(45)

* [1446. Consecutive Characters](#1446-consecutive-characters)
* [1869. Longer Contiguous Segments of Ones than Zeros](#1869-longer-contiguous-segments-of-ones-than-zeros)
* [1957. Delete Characters to Make Fancy String](#1957-delete-characters-to-make-fancy-string)
* [2414. Length of the Longest Alphabetical Continuous Substring](#2414-length-of-the-longest-alphabetical-continuous-substring)
* [674. Longest Continuous Increasing Subsequence](#674-longest-continuous-increasing-subsequence)

* [978. Longest Turbulent Subarray](#978-longest-turbulent-subarray)
* [2110. Number of Smooth Descent Periods of a Stock](#2110-number-of-smooth-descent-periods-of-a-stock)
* [228. Summary Ranges](#228-summary-ranges)
* [2760. Longest Even Odd Subarray With Threshold](#2760-longest-even-odd-subarray-with-threshold)
* [1887. Reduction Operations to Make the Array Elements Equal](#1887-reduction-operations-to-make-the-array-elements-equal)

* [845. Longest Mountain in Array](#845-longest-mountain-in-array)
* [2038. Remove Colored Pieces if Both Neighbors are the Same Color](#2038-remove-colored-pieces-if-both-neighbors-are-the-same-color)
* [1759. Count Number of Homogenous Substrings](#1759-count-number-of-homogenous-substrings)
* [3011. Find if Array Can Be Sorted](#3011-find-if-array-can-be-sorted)
* [1578. Minimum Time to Make Rope Colorful](#1578-minimum-time-to-make-rope-colorful)

* [1839. Longest Substring Of All Vowels in Order](#1839-longest-substring-of-all-vowels-in-order)
* [2765. Longest Alternating Subarray](#2765-longest-alternating-subarray)
* [3255. Find the Power of K-Size Subarrays II](#3255-find-the-power-of-k-size-subarrays-ii)
* [3350. Adjacent Increasing Subarrays Detection II](#3350-adjacent-increasing-subarrays-detection-ii)
* [3105. Longest Strictly Increasing or Strictly Decreasing Subarray](#3105-longest-strictly-increasing-or-strictly-decreasing-subarray)

* [2393. Count Strictly Increasing Subarrays](#2393-count-strictly-increasing-subarrays)
* [2436. Minimum Split Into Subarrays With GCD Greater Than One](#2436-minimum-split-into-subarrays-with-gcd-greater-than-one)
* [3063. Linked List Frequency](#3063-linked-list-frequency)
* [2495. Number of Subarrays Having Even Product](#2495-number-of-subarrays-having-even-product)
* [467. Unique Substrings in Wraparound String](#467-unique-substrings-in-wraparound-string)

* [2593. Find Score of an Array After Marking All Elements](#2593-find-score-of-an-array-after-marking-all-elements)
* [186. Reverse Words in a String II](#186-reverse-words-in-a-string-ii)
* [1784. Check if Binary String Has at Most One Segment of Ones](#1784-check-if-binary-string-has-at-most-one-segment-of-ones)
* [551. Student Attendance Record I](#551-student-attendance-record-i)
* [1933. Check if String Is Decomposable Into Value-Equal Substrings](#1933-check-if-string-is-decomposable-into-value-equal-substrings)

* [408. Valid Word Abbreviation](#408-valid-word-abbreviation)
* [1513. Number of Substrings With Only 1s](#1513-number-of-substrings-with-only-1s)
* [2348. Number of Zero-Filled Subarrays](#2348-number-of-zero-filled-subarrays)
* [2405. Optimal Partition of String](#2405-optimal-partition-of-string)
* [1529. Minimum Suffix Flips](#1529-minimum-suffix-flips)

* [758. Bold Words in String](#408-valid-word-abbreviation)
* [2419. Longest Subarray With Maximum Bitwise AND](#2419-longest-subarray-with-maximum-bitwise-and)
* [2645. Minimum Additions to Make Valid String](#2645-minimum-additions-to-make-valid-string)
* [809. Expressive Words](#809-expressive-words)
* [1807. Evaluate the Bracket Pairs of a String](#1807-evaluate-the-bracket-pairs-of-a-string)

* [830. Positions of Large Groups](#830-positions-of-large-groups)
* [2943. Maximize Area of Square Hole in Grid](#2943-maximize-area-of-square-hole-in-grid)
* [2981. Find Longest Special Substring That Occurs Thrice I](#2981-find-longest-special-substring-that-occurs-thrice-i)
* [2982. Find Longest Special Substring That Occurs Thrice II](#2982-find-longest-special-substring-that-occurs-thrice-ii)
* [2948. Make Lexicographically Smallest Array by Swapping Elements](#2948-make-lexicographically-smallest-array-by-swapping-elements)

### 1446. Consecutive Characters

```python
class Solution:
    def maxPower(self, s: str) -> int:
        res, i, n = 0, 0, len(s)
        while i < n:
            j = i + 1
            while j < n and s[j] == s[j - 1]:
                j += 1
            res = max(res, j - i)
            i = j 
        return res
```

### 1869. Longer Contiguous Segments of Ones than Zeros

```python
class Solution:
    def checkZeroOnes(self, s: str) -> bool:
        d = Counter()
        i, n = 0, len(s)
        while i < n:
            j = i + 1
            while j < n and s[j] == s[j - 1]:
                j += 1
            d[s[i]] = max(d[s[i]], j - i)
            i = j 
        return d['1'] > d['0']
```

### 1957. Delete Characters to Make Fancy String

```python
class Solution:
    def makeFancyString(self, s: str) -> str:
        res, i, n = '', 0, len(s)
        while i < n:
            j = i + 1
            while j < n and s[j] == s[j - 1]:
                j += 1
            res += s[i: min(i + 2, j)]
            i = j 
        return res
```

### 2414. Length of the Longest Alphabetical Continuous Substring

```python
class Solution:
    def longestContinuousSubstring(self, s: str) -> int:
        res, i, n = 0, 0, len(s)
        while i < n:
            j = i + 1
            while j < n and ord(s[j]) == ord(s[j - 1]) + 1:
                j += 1
            res = max(res, j - i)
            i = j 
        return res
```

### 674. Longest Continuous Increasing Subsequence

```python 
class Solution:
    def findLengthOfLCIS(self, nums: List[int]) -> int:
        res, i, n = 0, 0, len(nums)
        while i < n:
            j = i + 1
            while j < n and nums[j] > nums[j - 1]:
                j += 1
            res = max(res, j - i)
            i = j 
        return res
```

### 978. Longest Turbulent Subarray

```python 
class Solution:
    def maxTurbulenceSize(self, arr: List[int]) -> int:
        if len(arr) <= 1:
            return len(arr) 
            
        nums = [1 if a > b else -1 if a < b else 0 for a, b in pairwise(arr)]
        res, i, n = 0, 0, len(nums)
        while i < n:
            if nums[i] == 0:
                i += 1
                continue
            j = i + 1
            while j < n and nums[j] == -nums[j - 1]:
                j += 1
            res = max(res, j - i)
            i = j 
        return res + 1
```

### 2110. Number of Smooth Descent Periods of a Stock

```python
class Solution:
    def getDescentPeriods(self, prices: List[int]) -> int:
        res, i, n = 0, 0, len(prices)
        while i < n:
            j = i + 1
            while j < n and prices[j] == prices[j - 1] - 1:
                j += 1
            x = j - i
            res += x * (x + 1) // 2
            i = j 
        return res
```

### 228. Summary Ranges

```python
class Solution:
    def summaryRanges(self, nums: List[int]) -> List[str]:
        res, i, n = [], 0, len(nums)
        while i < n:
            j = i + 1
            while j < n and nums[j] == nums[j - 1] + 1:
                j += 1
            if i < j - 1:
                res.append(str(nums[i]) + '->' + str(nums[j - 1]))
            else:
                res.append(str(nums[i]))
            i = j 
        return res
```


### 2760. Longest Even Odd Subarray With Threshold

```python
class Solution:
    def longestAlternatingSubarray(self, nums: List[int], threshold: int) -> int:
        res, i, n = 0, 0, len(nums)
        while i < n:
            if nums[i] % 2 or nums[i] > threshold:
                i += 1
                continue
            start = i
            while i < n - 1 and nums[i] % 2 != nums[i + 1] % 2 and nums[i + 1] <= threshold:
                i += 1
            res = max(res, i - start + 1)
            i += 1
        return res
```

### 1887. Reduction Operations to Make the Array Elements Equal

```python 
class Solution:
    def reductionOperations(self, nums: List[int]) -> int:
        res, i, n = 0, 0, len(nums)
        nums.sort()
        k = 0 
        while i < n:
            j = i + 1
            while j < n and nums[j] == nums[j - 1]:
                j += 1
            res += (j - i) * k 
            k += 1
            i = j 
        return res 
```

### 845. Longest Mountain in Array

```python 
class Solution:
    def longestMountain(self, arr: List[int]) -> int:
        def check(i):
            l, r = i, i 
            while r + 1 < n and arr[r + 1] < arr[r]:
                r += 1
            while l - 1 >= 0 and arr[l - 1] < arr[l]:
                l -= 1
            return r - l + 1
        n, res = len(arr), 0
        idx_arr = []
        for i in range(1, n - 1):
            if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
                idx_arr.append(i)
        for i in idx_arr:
            res = max(res, check(i))
        return res if res >= 3 else 0
```

### 2038. Remove Colored Pieces if Both Neighbors are the Same Color

```python
class Solution:
    def winnerOfGame(self, colors: str) -> bool:
        cntA, cntB = 0, 0
        i, n = 0, len(colors)
        while i < n:
            j = i + 1
            while j < n and colors[j] == colors[j - 1]:
                j += 1
            if colors[i] == 'A' and j - i > 2:
                cntA += j - i - 2
            if colors[i] == 'B' and j - i > 2:
                cntB += j - i - 2
            i = j
        return cntA > cntB
```

### 1759. Count Number of Homogenous Substrings

```python
class Solution:
    def countHomogenous(self, s: str) -> int:
        mod = 10 ** 9 + 7
        res, i, n = 0, 0, len(s)
        while i < n:
            j = i + 1
            while j < n and s[j] == s[j - 1]:
                j += 1
            x = j - i
            res += x * (x + 1) // 2
            i = j 
        return res % mod
```

### 3011. Find if Array Can Be Sorted

```python 
class Solution:
    def canSortArray(self, nums: List[int]) -> bool:
        i, n, res = 0, len(nums), []
        while i < n:
            j = i + 1
            while j < n and nums[j].bit_count() == nums[j - 1].bit_count():
                j += 1
            res.extend(sorted(nums[i: j]))
            i = j 
        return res == sorted(nums)
```

### 1578. Minimum Time to Make Rope Colorful

```python
class Solution:
    def minCost(self, colors: str, neededTime: List[int]) -> int:
        res, i, n = 0, 0, len(colors)
        total, mx = 0, 0
        while i < n:
            j = i + 1
            total, mx = neededTime[i], neededTime[i]
            while j < n and colors[j] == colors[j - 1]:
                total += neededTime[j]
                mx = max(mx, neededTime[j])
                j += 1
            res += total - mx 
            i = j 
        return res
```

### 1839. Longest Substring Of All Vowels in Order

```python
class Solution:
    def longestBeautifulSubstring(self, word: str) -> int:  
        res, i, n = 0, 0, len(word)
        while i < n:
            j = i + 1
            s = set([word[i]])  
            while j < n and word[j] >= word[j - 1]:
                s.add(word[j])
                j += 1
            if len(s) == 5:
                res = max(res, j - i)
            i = j 
        return res
```

### 2765. Longest Alternating Subarray

```python
class Solution:
    def alternatingSubarray(self, nums: List[int]) -> int:
        nums = [b - a for a, b in pairwise(nums)]
        res, i, n = 0, 0, len(nums)
        while i < n:
            j = i + 1
            if nums[i] == 1:
                while j < n and nums[j] == -nums[j - 1]:
                    j += 1
                res = max(res, j - i)
            i = j 
        return res + 1 if res > 0 else -1
```

### 3255. Find the Power of K-Size Subarrays II

```python 
class Solution:
    def resultsArray(self, nums: List[int], k: int) -> List[int]:
        if k == 1:
            return nums
        i, n = 0, len(nums)
        res = [-1] * (n - k + 1)
        while i < n:
            j = i + 1
            while j < n and nums[j] == nums[j - 1] + 1:
                if j - i >= k - 1:
                    res[j - k + 1] = nums[j]
                j += 1
            i = j 
        return res
```

### 3350. Adjacent Increasing Subarrays Detection II

```python 
class Solution:
    def maxIncreasingSubarrays(self, nums: List[int]) -> int:
        res, i, n = [], 0, len(nums)
        while i < n:
            j = i + 1
            while j < n and nums[j] >= nums[j - 1] + 1:
                j += 1
            res.append(j - i)
            i = j 
        mx, ans = max(res), 0
        for a, b in pairwise(res):
            ans = max(ans, min(a, b))
        return max(ans, mx // 2)
```

### 3105. Longest Strictly Increasing or Strictly Decreasing Subarray

```python 
class Solution:
    def longestMonotonicSubarray(self, nums: List[int]) -> int:
        res1, i, n = 0, 0, len(nums)
        while i < n:
            j = i + 1
            while j < n and nums[j] > nums[j - 1]:
                j += 1
            res1 = max(res1, j - i)
            i = j 
        
        res2, i = 0, 0
        while i < n:
            j = i + 1
            while j < n and nums[j] < nums[j - 1]:
                j += 1
            res2 = max(res2, j - i)
            i = j 
        return max(res1, res2)
```

### 2393. Count Strictly Increasing Subarrays

```python 
class Solution:
    def countSubarrays(self, nums: List[int]) -> int:
        res, i, n = 0, 0, len(nums)
        while i < n:
            j = i + 1
            while j < n and nums[j] > nums[j - 1]:
                j += 1
            x = j - i
            res += x * (x + 1) // 2
            i = j 
        return res
```

### 2436. Minimum Split Into Subarrays With GCD Greater Than One

```python 
class Solution:
    def minimumSplits(self, nums: List[int]) -> int:
        n = len(nums)
        res = 0
        for i in range(1, n):
            if gcd(nums[i - 1], nums[i]) > 1:
                nums[i] = gcd(nums[i - 1], nums[i])
            else:
                res += 1
        return res + 1
```

### 3063. Linked List Frequency

```python 
class Solution:
    def frequenciesOfElements(self, head: Optional[ListNode]) -> Optional[ListNode]:
        d = defaultdict(int)
        p = dummy = ListNode()
        while head:
            d[head.val] += 1
            head = head.next 
        for v in d.values():
            p.next = ListNode(v)
            p = p.next 
        return dummy.next 
```

### 2495. Number of Subarrays Having Even Product

```python 
class Solution:
    def evenProduct(self, nums: List[int]) -> int:
        n = len(nums)
        last_even_idx = -1
        res = 0
        for i in range(n):
            if nums[i] % 2 == 0:
                last_even_idx = i
            if last_even_idx >= 0:
                res += last_even_idx + 1
        return res
```

### 467. Unique Substrings in Wraparound String

```python 
class Solution:
    def findSubstringInWraproundString(self, s: str) -> int:
        d = Counter()
        res, i, n = 0, 0, len(s)
        while i < n:
            j = i + 1
            d[s[i]] = max(d[s[i]], 1)
            while j < n and (ord(s[j]) == ord(s[j - 1]) + 1 or ord(s[j]) == ord(s[j - 1]) - 25):
                d[s[j]] = max(d[s[j]], j - i + 1)
                j += 1
            i = j
        return sum(d.values())
```

### 2593. Find Score of an Array After Marking All Elements

```python 
class Solution:
    def findScore(self, nums: List[int]) -> int:
        a = sorted([(i, n) for i, n in enumerate(nums)], key = lambda x: x[1])
        res, s = 0, set()
        for i, n in a:
            if i not in s:
                res += n 
                s.add(i)
                s.add(i - 1)
                s.add(i + 1)
        return res 
```

### 186. Reverse Words in a String II

```python
class Solution:
    def reverseWords(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        def subReverse(l, r):
            while l < r:
                s[l], s[r] = s[r], s[l]
                l += 1
                r -= 1
    
        s.reverse()
        i = 0
        while i < len(s):
            if s[i] != ' ':
                start = i
                j = start 
                while j < len(s) and s[j] != ' ':
                    j += 1
                subReverse(start, j - 1)
                i = j 
            else:
                i += 1
        return s
```

### 1784. Check if Binary String Has at Most One Segment of Ones

```python
class Solution:
    def checkOnesSegment(self, s: str) -> bool:
        count = 0
        i = 0
        while i < len(s):
            if s[i] == '1':
                start = i
                j = start
                while j < len(s) and s[j] == '1':
                    j += 1
                count += 1
                i = j 
            i += 1
        return count <= 1
```

### 551. Student Attendance Record I

```python
class Solution:
    def checkRecord(self, s: str) -> bool:
        i = 0
        late, absent = 0, s.count('A')
        while i < len(s):
            start = i
            j = start
            while j < len(s) and s[j] == s[start]:
                j += 1
            if s[start] == 'L':
                late = max(late, j - start)
            i = j 
        return absent < 2 and late < 3
```

### 1933. Check if String Is Decomposable Into Value-Equal Substrings

```python
class Solution:
    def isDecomposable(self, s: str) -> bool:
        i = 0
        res = []
        while i < len(s):
            start = i 
            j = start
            while j < len(s) and s[start] == s[j]:
                j += 1
            if (j - start) % 3 == 1:
                return False
            res.append((j - start) % 3)
            i = j
        ans = 0
        for n in res:
            if n == 2:
                ans += 1
        return ans == 1
```

### 408. Valid Word Abbreviation

```python
class Solution:
    def validWordAbbreviation(self, word: str, abbr: str) -> bool:
        res = ''
        i = 0
        count = 0
        while i < len(abbr):
            if abbr[i] == '0':
                return False
            if abbr[i].isdigit():
                start = i
                j = start 
                while j < len(abbr) and abbr[j].isdigit():
                    j += 1
                res += word[count: count + int(abbr[start: j])]
                i = j 
                count += int(abbr[start: j])
                if count > len(res):
                    return False
            else:
                res += abbr[i]
                count += 1
                i += 1
        return res == word
```

### 758. Bold Words in String

```python
class Solution:
    def boldWords(self, words: List[str], s: str) -> str:
        n = len(s)
        res = [0] * n 
        for i in range(n):
            for w in words:
                if s[i:].startswith(w):
                    for j in range(i, i + len(w)):
                        res[j] = 1

        i, ans = 0, ''
        while i < n:
            start = i 
            j = start
            while j < n and res[j] == res[start]:
                j += 1
            if res[start] == 0:
                ans += s[start: j]
            else:
                ans += '<b>' + s[start: j] + '</b>'
            i = j
        return ans
```

### 1513. Number of Substrings With Only 1s

```python
class Solution:
    def numSub(self, s: str) -> int:
        arr = []
        i = 0
        while i < len(s):
            if s[i] == '1':
                start = i 
                j = start 
                while j < len(s) and s[j] == s[start]:
                    j += 1
                arr.append(j - start)
                i = j
            else:
                i += 1
        res = 0
        mod = 10 ** 9 + 7
        for n in arr:
            res += n * (n + 1) // 2
        return res % mod
```

### 2348. Number of Zero-Filled Subarrays

```python
class Solution:
    def zeroFilledSubarray(self, nums: List[int]) -> int:
        arr = []
        i = 0
        while i < len(nums):
            if nums[i] == 0:
                start = i 
                j = start 
                while j < len(nums) and nums[j] == 0:
                    j += 1
                arr.append(j - start)
                i = j 
            else:
                i += 1
        res = 0
        for n in arr:
            res += n * (n + 1) // 2
        return res
```

### 2405. Optimal Partition of String

```python
class Solution:
    def partitionString(self, s: str) -> int:
        i = 0
        res = 0
        while i < len(s):
            j = i
            visited = set()
            while j < len(s) and s[j] not in visited:
                visited.add(s[j])
                j += 1
            res += 1
            i = j
        return res
```

### 1529. Minimum Suffix Flips

```python
class Solution:
    def minFlips(self, target: str) -> int:
        i = 0
        while i < len(target) and target[i] == '0':
            i += 1
        s = target[i:]
        i = 0
        res = 0
        while i < len(s):
            start = i 
            j = start
            while j < len(s) and s[j] == s[start]:
                j += 1
            res += 1
            i = j 
        return res
```

### 2419. Longest Subarray With Maximum Bitwise AND

```python
class Solution:
    def longestSubarray(self, nums: List[int]) -> int:
        mx = max(nums)
        i = 0
        res = 0
        n = len(nums)
        while i < n:
            if nums[i] == mx:
                j = i
                while j < n and nums[j] == mx:
                    j += 1
                res = max(res, j - i)
                i = j 
            else:
                i += 1
        return res
```

### 2645. Minimum Additions to Make Valid String

```python
class Solution:
    def addMinimum(self, word: str) -> int:
        i, n = 0, len(word)
        res = 0
        while i < n:
            j = i + 1
            while j < n and ord(word[j]) - ord(word[j - 1]) in [1, 2]:
                j += 1
            res += 3 - (j - i)
            i = j 
        return res
```

### 809. Expressive Words

```python
class Solution:
    def expressiveWords(self, s: str, words: List[str]) -> int:
        def listGroup(s):
            n = len(s)
            res = []
            i = 0
            while i < n:
                start = i 
                j = start 
                while j < n and s[j] == s[start]:
                    j += 1
                res.append((s[start], j - start))
                i = j
            return res 
        def check(word, origin):
            if [c for c, n in word] != [c for c, n in origin]:
                return False
            for a, b in zip(word, origin):
                if a[1] > b[1] or (a[1] < b[1] and b[1] < 3):
                    return False
            return True

        origin = listGroup(s)
        res = 0
        for word in words:
            if check(listGroup(word), origin):
                res += 1
        return res
```

### 1807. Evaluate the Bracket Pairs of a String

```python
class Solution:
    def evaluate(self, s: str, knowledge: List[List[str]]) -> str:
        d = defaultdict(lambda: '?')
        for key, val in knowledge:
            d[key] = val 
        res = ''
        i = 0
        while i < len(s):
            if s[i] == '(':
                j = i + 1
                while j < len(s) and s[j] != ')':
                    j += 1
                res += d[s[i + 1: j]]
                i = j + 1
            else:
                res += s[i]
                i += 1
        return res
```

### 830. Positions of Large Groups

```python
class Solution:
    def largeGroupPositions(self, s: str) -> List[List[int]]:
        i, res = 0, []
        while i < len(s):
            start = i 
            j = start
            while j < len(s) and s[j] == s[start]:
                j += 1
            if j - start >= 3:
                res.append([start, j - 1])
            i = j
        return res
```


### 2943. Maximize Area of Square Hole in Grid

```python
class Solution:
    def maximizeSquareHoleArea(self, n: int, m: int, hBars: List[int], vBars: List[int]) -> int:
        hBars.sort()
        vBars.sort()
        def check(nums):
            res, i, n = 0, 0, len(nums)
            while i < n:
                start = i
                while i < n - 1 and nums[i] + 1 == nums[i + 1]:
                    i += 1
                res = max(res, i - start + 1)
                i += 1
            return res
        h, v = check(hBars) + 1, check(vBars) + 1
        return min(h, v) ** 2
```

### 2981. Find Longest Special Substring That Occurs Thrice I

```python
class Solution:
    def maximumLength(self, s: str) -> int:
        def check(s):
            n = len(s)
            letter = s[0]
            for i in range(n):
                c[letter * (i + 1)] += n - i 
                
        n, c, i = len(s), Counter(), 0
        while i < n:
            start = i
            j = start 
            while j < n and s[j] == s[start]:
                j += 1
            check(s[start: j])
            i = j
        res = [len(k) for k, v in c.items() if v >= 3]
        return max(res) if res else -1
```

### 2982. Find Longest Special Substring That Occurs Thrice II

```python
class Solution:
    def maximumLength(self, s: str) -> int:
        def check(s):
            n = len(s)
            letter = s[0]
            for i in range(n - 1, -1, -1):
                c[letter * (i + 1)] += n - i 
                if c[letter * (i + 1)] >= 3:
                    break
                
        n, c, i = len(s), Counter(), 0
        while i < n:
            start = i
            j = start 
            while j < n and s[j] == s[start]:
                j += 1
            check(s[start: j])
            i = j
        res = [len(k) for k, v in c.items() if v >= 3]
        return max(res) if res else -1
```

### 2948. Make Lexicographically Smallest Array by Swapping Elements

```python 
class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        nums = sorted([(i, n) for i, n  in enumerate(nums)], key = lambda x: x[1])
        i, n = 0, len(nums)
        res = [0] * n 
        while i < n:
            j = i + 1
            while j < n and abs(nums[j][1] - nums[j - 1][1]) <= limit:
                j += 1
            idx_array = sorted([i for i, n in nums[i: j]])
            num = [n for i, n in nums[i: j]]
            for idx, v in zip(idx_array, num):
                res[idx] = v
            i = j 
        return res
```