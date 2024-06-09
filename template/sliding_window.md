## template

```python
def fn(arr):
    l, res = 0, 0
    for r, v in enumerate(arr):
        # may need some code
        while WINDOW_BROKEN:
            # handle l pointer of window
            l += 1
        res = max(res, r - l + 1)
    return res
```

## Question list

### 1 fixed

* [1456. Maximum Number of Vowels in a Substring of Given Length 1263](#1456-maximum-number-of-vowels-in-a-substring-of-given-length)
* [2269. Find the K-Beauty of a Number 1280](#2269-find-the-k-beauty-of-a-number)
* [1984. Minimum Difference Between Highest and Lowest of K Scores 1306](#1984-minimum-difference-between-highest-and-lowest-of-k-scores)
* [643. Maximum Average Subarray I 1350](#643-maximum-average-subarray-i)
* [1343. Number of Sub-arrays of Size K and Average Greater than or Equal to Threshold 1317](#1343-number-of-sub-arrays-of-size-k-and-average-greater-than-or-equal-to-threshold)
* [2090. K Radius Subarray Averages 1358](#2090-k-radius-subarray-averages)
* [2379. Minimum Recolors to Get K Consecutive Black Blocks 1360](#2379-minimum-recolors-to-get-k-consecutive-black-blocks)
* [1052. Grumpy Bookstore Owner 1418](#1052-grumpy-bookstore-owner)
* [2841. Maximum Sum of Almost Unique Subarray 1546](#2841-maximum-sum-of-almost-unique-subarray)
* [2461. Maximum Sum of Distinct Subarrays With Length K 1553](#2461-maximum-sum-of-distinct-subarrays-with-length-k)
* [1423. Maximum Points You Can Obtain from Cards 1574](#1423-maximum-points-you-can-obtain-from-cards)
* [2134. Minimum Swaps to Group All 1's Together II 1748](#2134-minimum-swaps-to-group-all-1s-together-ii)
* [2653. Sliding Subarray Beauty 1786](#2653-sliding-subarray-beauty)
* [438. Find All Anagrams in a String 1750](#438-find-all-anagrams-in-a-string)
* [567. Permutation in String 1750](#567-permutation-in-string)
2156. 查找给定哈希值的子串 2063
2953. 统计完全子字符串 2449
346. 数据流中的移动平均值（会员题）
1100. 长度为 K 的无重复字符子串（会员题）
1852. 每个子数组的数字种类数（会员题）
2067. 等计数子串的数量（会员题）
2107. 分享 K 个糖果后独特口味的数量（会员题）

### 2 non-fixed (longest or largest)

* [3. Longest Substring Without Repeating Characters 1500](#3-Longest-Substring-Without-Repeating-Characters)
* [1493. Longest Subarray of 1's After Deleting One Element 1423](#1493-longest-subarray-of-1s-after-deleting-one-element)
* [2730. Find the Longest Semi-Repetitive Substring 1502](#2730-find-the-longest-semi-repetitive-substring)
* [904. Fruit Into Baskets 1516](#904-fruit-into-baskets)
* [1695. Maximum Erasure Value 1529](#1695-maximum-erasure-value)
* [2958. Length of Longest Subarray With at Most K Frequency 1535](#2958-length-of-longest-subarray-with-at-most-k-frequency)
* [2024. Maximize the Confusion of an Exam 1643](#2024-maximize-the-confusion-of-an-exam)
* [1004. Max Consecutive Ones III 1656](#1004-max-consecutive-ones-iii)
1438. 绝对差不超过限制的最长连续子数组 1672
2401. 最长优雅子数组 1750
1658. 将 x 减到 0 的最小操作数 1817
1838. 最高频元素的频数 1876
2516. 每种字符至少取 K 个 1948
2831. 找出最长等值子数组 1976
2106. 摘水果 2062
1610. 可见点的最大数目 2147
2781. 最长合法子字符串的长度 2204
2968. 执行操作使频率分数最大 2444
395. 至少有 K 个重复字符的最长子串
1763. 最长的美好子字符串
159. 至多包含两个不同字符的最长子串（会员题）
340. 至多包含 K 个不同字符的最长子串（会员题）


* [209. Minimum Size Subarray Sum](#209-Minimum-Size-Subarray-Sum)
* [15. 3Sum](#15-3Sum)
* [16. 3Sum Closest](#16-3Sum-Closest)
* [259. 3Sum Smaller](#259-3Sum-Smaller)
* [18. 4Sum](#18-4Sum)
* [11. Container With Most Water](#11-Container-With-Most-Water)

* [159. Longest Substring with At Most Two Distinct Characters](#159-Longest-Substring-with-At-Most-Two-Distinct-Characters)
* [340. Longest Substring with At Most K Distinct Characters](#340-Longest-Substring-with-At-Most-K-Distinct-Characters)[same as 159]
* [487. Max Consecutive Ones II](#487-Max-Consecutive-Ones-II)
* [1004. Max Consecutive Ones III](#1004-Max-Consecutive-Ones-III)[same as 487]

### fixed

### 1456. Maximum Number of Vowels in a Substring of Given Length

```python
class Solution:
    def maxVowels(self, s: str, k: int) -> int:
        vowels = set('aeiou')
        res = cnt = l = 0
        for r, c in enumerate(s):
            if c in vowels:
                cnt += 1
            if r - l + 1 == k:
                res = max(res, cnt)
                if s[l] in vowels:
                    cnt -= 1
                l += 1
        return res
```

### 2269. Find the K-Beauty of a Number

```python
class Solution:
    def divisorSubstrings(self, num: int, k: int) -> int:
        s = str(num)
        n, res = len(s), 0
        for i in range(n - k + 1):
            ans = int(s[i: i + k])
            if ans and num % ans == 0:
                res += 1
        return res 
```

### 1984. Minimum Difference Between Highest and Lowest of K Scores

```python
class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        nums.sort()
        res, l = inf, 0
        for r, n in enumerate(nums):
            if r - l + 1 == k:
                res = min(res, nums[r] - nums[l])
                l += 1
        return res 
```

### 643. Maximum Average Subarray I

```python
class Solution:
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        res, l, total = -inf, 0, 0
        for r, n in enumerate(nums):
            total += n 
            if r - l + 1 == k:
                res = max(res, total / k)
                total -= nums[l]
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

### 2090. K Radius Subarray Averages

```python
class Solution:
    def getAverages(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        l, res = 0, [-1] * n
        total = sum(nums[: 2 * k])
        for r in range(k, n - k):
            total += nums[r + k]
            res[r] = total // (2 * k + 1)
            total -= nums[r - k]
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

### 1052. Grumpy Bookstore Owner

```python
class Solution:
    def maxSatisfied(self, customers: List[int], grumpy: List[int], minutes: int) -> int:
        total = 0
        for c, g in zip(customers, grumpy):
            if g == 0:
                total += c
        l, res, mx = 0, 0, 0
        for r, c in enumerate(customers):
            if grumpy[r] == 1:
                mx += customers[r]
            if r - l + 1 == minutes:
                res = max(res, mx)
                if grumpy[l] == 1:
                    mx -= customers[l]
                l += 1
        return total + res
```

### 2841. Maximum Sum of Almost Unique Subarray

```python
class Solution:
    def maxSum(self, nums: List[int], m: int, k: int) -> int:
        res, d, l = 0, Counter(), 0
        total = 0
        for r, n in enumerate(nums):
            d[n] += 1
            total += n
            if r - l + 1 == k:
                if len(d) >= m:
                    res = max(res, total)
                d[nums[l]] -= 1
                total -= nums[l]
                if d[nums[l]] == 0:
                    d.pop(nums[l])
                l += 1
        return res 
```

### 2461. Maximum Sum of Distinct Subarrays With Length K

```python
class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        d = Counter()
        l, res = 0, 0
        total = 0
        for r, n in enumerate(nums):
            total += n
            d[n] += 1
            if r - l + 1 == k:
                if len(d) == k:
                    res = max(res, total)
                d[nums[l]] -= 1
                if d[nums[l]] == 0:
                    d.pop(nums[l])
                total -= nums[l]
                l += 1
        return res
```

### 1423. Maximum Points You Can Obtain from Cards

- reverse 

```python
class Solution:
    def maxScore(self, cardPoints: List[int], k: int) -> int:
        n = len(cardPoints)
        window = n - k
        l, res, total = 0, inf, 0
        for r, c in enumerate(cardPoints):
            total += c
            if r - l + 1 == window:
                res = min(res, total)
                total -= cardPoints[l]
                l += 1
        return sum(cardPoints) - res if window else sum(cardPoints)
```

### 2134. Minimum Swaps to Group All 1's Together II

- fix window same as 1151

```python
class Solution:
    def minSwaps(self, nums: List[int]) -> int:
        # [0,1,0,1,1,0,0,0,1,0,1,1,0,0]
        # [1,1,0,0,1,1,1,0,0,1]
        # [1,1,1,0,0,1,0,1,1,0] [1,1,1,0,0,1,0,1,1,0]
        window = nums.count(1)
        nums += nums 
        l, res = 0, inf
        total = 0
        for r, n in enumerate(nums):
            if n == 1:
                total += 1
            if r - l + 1 == window:
                res = min(res, window - total)
                if nums[l] == 1:
                    total -= 1
                l += 1
        return res if res != inf else 0
```

### 2653. Sliding Subarray Beauty

```python
from sortedcontainers import SortedList
class Solution:
    def getSubarrayBeauty(self, nums: List[int], k: int, x: int) -> List[int]:
        sl = SortedList()
        l, res = 0, []
        for r, n in enumerate(nums):
            sl.add(n)
            if len(sl) > k:
                sl.remove(nums[l])
                l += 1
            if len(sl) == k:
                if sl[x - 1] < 0:
                    res.append(sl[x - 1])
                else:
                    res.append(0)
        return res
```

### 438. Find All Anagrams in a String

```python
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        pLength = len(p)
        d, p = Counter(), Counter(p)
        res, l = [], 0
        for r, c in enumerate(s):
            d[c] += 1
            if r - l + 1 == pLength:
                if d == p:
                    res.append(l)
                d[s[l]] -= 1
                l += 1
        return res
```

### 567. Permutation in String

```python
class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        d1, d2 = Counter(s1), Counter()
        l = 0
        for r, c in enumerate(s2):
            d2[c] += 1
            if r - l + 1 == len(s1):
                if d2 == d1:
                    return True
                d2[s2[l]] -= 1
                l += 1
        return False
```

### non-fixed


### 3. Longest Substring Without Repeating Characters

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        d = defaultdict(int)
        res, l = 0, 0
        for r, v in enumerate(s):
            d[v] += 1
            while d[v] > 1: # window broken
                d[s[l]] -= 1
                l += 1
            res = max(res, r - l + 1)
        return res
```

### 1493. Longest Subarray of 1's After Deleting One Element

```python
class Solution:
    def longestSubarray(self, nums: List[int]) -> int:
        l, res, count = 0, 0, 0
        for r, n in enumerate(nums):
            if n == 0:
                count += 1
            while count > 1:
                if nums[l] == 0:
                    count -= 1
                l += 1
            res = max(res, r - l)
        return res
```

### 2730. Find the Longest Semi-Repetitive Substring

```python
class Solution:
    def longestSemiRepetitiveSubstring(self, s: str) -> int:
        def check(i, j):
            res = s[i: j + 1]
            l, count = 0, 0
            for r in range(1, len(res)):
                if res[r] == res[l]:
                    count += 1
                l += 1
            return count <= 1

        n, res = len(s), 0
        for i in range(n):
            for j in range(i, n):
                if check(i, j):
                    res = max(res, j - i + 1)
        return res
```


### 904. Fruit Into Baskets

```python
class Solution:
    def totalFruit(self, fruits: List[int]) -> int:
        d = defaultdict(int)
        l, res = 0, 0
        for r, c in enumerate(fruits):
            d[c] += 1
            while len(d) > 2:
                d[fruits[l]] -= 1
                if d[fruits[l]] == 0:
                    d.pop(fruits[l])
                l += 1
            res = max(res, r - l + 1)
        return res
```

### 1695. Maximum Erasure Value

```python
class Solution:
    def maximumUniqueSubarray(self, nums: List[int]) -> int:
        l, res = 0, 0
        d, total = defaultdict(int), 0
        for r, n in enumerate(nums):
            d[n] += 1
            total += n
            while d[n] > 1:
                d[nums[l]] -= 1
                total -= nums[l]
                l += 1
            res = max(res, total)
        return res
```

### 2958. Length of Longest Subarray With at Most K Frequency

```python
class Solution:
    def maxSubarrayLength(self, nums: List[int], k: int) -> int:
        # nums = [1,2,3,1,2,3,1,2], k = 2
        d = defaultdict(int)
        res, l = 0, 0
        for r, n in enumerate(nums):
            d[n] += 1
            while d[n] > k:
                d[nums[l]] -= 1
                l += 1
            res = max(res, r - l + 1)
        return res
```


### 2024. Maximize the Confusion of an Exam

```python
class Solution:
    def maxConsecutiveAnswers(self, answerKey: str, k: int) -> int:
        d, l, res = Counter(), 0, 0
        for r, c in enumerate(answerKey):
            d[c] += 1
            while len(d) == 2 and min(d.values()) > k:
                d[answerKey[l]] -= 1
                if d[answerKey[l]] == 0:
                    d.pop(answerKey[l])
                l += 1
            res = max(res, r - l + 1)
        return res
```

### 1004. Max Consecutive Ones III

```python
class Solution:
    def longestOnes(self, nums: List[int], k: int) -> int:
        count = 0
        l, res = 0, 0
        for r, n in enumerate(nums):
            if n == 0:
                count += 1
            while count > k:
                if nums[l] == 0:
                    count -= 1
                l += 1
            res = max(res, r - l + 1)
        return res
```


##############################################################
### 209. Minimum Size Subarray Sum

```python
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        res, l, ans = inf, 0, 0
        for r, v in enumerate(nums):
            ans += v
            while ans >= target: # check res inside window
                res = min(res, r - l + 1)
                ans -= nums[l]
                l += 1
        return res if res != inf else 0
```

### 713. Subarray Product Less Than K

```python
class Solution:
    def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
        if k <= 1:
            return 0
        res, l, ans = 0, 0, 1
        for r, v in enumerate(nums):
            ans *= v
            while ans >= k:
                ans //= nums[l]
                l += 1
            res += r - l + 1
        return res
```

### 159. Longest Substring with At Most Two Distinct Characters

```python
class Solution:
    def lengthOfLongestSubstringTwoDistinct(self, s: str) -> int:
        d = defaultdict(int)
        res, l = 0, 0
        for r, c in enumerate(s):
            d[c] += 1
            while len(d) > 2:
                d[s[l]] -= 1
                if d[s[l]] == 0:
                    d.pop(s[l])
                l += 1
            res = max(res, r - l  + 1)
        return res
```

### 340. Longest Substring with At Most K Distinct Characters

```python
class Solution:
    def lengthOfLongestSubstringKDistinct(self, s: str, k: int) -> int:
        d = defaultdict(int)
        res, l = 0, 0
        for r, c in enumerate(s):
            d[c] += 1
            while len(d) > k:
                d[s[l]] -= 1
                if d[s[l]] == 0:
                    d.pop(s[l])
                l += 1
            res = max(res, r - l + 1)
        return res
```


### 487. Max Consecutive Ones II

```python
class Solution:
    def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
        count = 0
        l, res = 0, 0
        for r, n in enumerate(nums):
            if n == 0:
                count += 1
            while count > 1:
                if nums[l] == 0:
                    count -= 1
                l += 1
            res = max(res, r - l + 1)
        return res
```

### 2260. Minimum Consecutive Cards to Pick Up

```python
class Solution:
    def minimumCardPickup(self, cards: List[int]) -> int:
        d, res, l = defaultdict(int), inf, 0
        for r, c in enumerate(cards):
            d[c] += 1
            while d[c] > 1:
                res = min(res, r - l + 1)
                d[cards[l]] -= 1
                l += 1
        return res if res != inf else -1
```

### 930. Binary Subarrays With Sum

```python
class Solution:
    def numSubarraysWithSum(self, nums: List[int], goal: int) -> int:
        res, total, l = 0, 0, 0
        prefix = defaultdict(int)
        prefix[0] = 1
        for r, n in enumerate(nums):
            total += n
            if total - goal in prefix:
                res += prefix[total - goal] 
            prefix[total] += 1
        return res
```

### 1876. Substrings of Size Three with Distinct Characters

```python
class Solution:
    def numSubarraysWithSum(self, nums: List[int], goal: int) -> int:
        res, total, l = 0, 0, 0
        prefix = defaultdict(int)
        prefix[0] = 1
        for r, n in enumerate(nums):
            total += n
            if total - goal in prefix:
                res += prefix[total - goal] 
            prefix[total] += 1
        return res
```


### 1208. Get Equal Substrings Within Budget

```python
class Solution:
    def equalSubstring(self, s: str, t: str, maxCost: int) -> int:
        cost, l, res = 0, 0, 0
        for r, c in enumerate(s):
            cost += abs(ord(c) - ord(t[r]))
            while cost > maxCost:
                cost -= abs(ord(s[l]) - ord(t[l]))
                l += 1
            res = max(res, r - l + 1)
        return res
```

### 76. Minimum Window Substring

```python
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        d, d_t = Counter(), Counter(t)
        res, l = s + '#', 0
        for r, v in enumerate(s):
            d[v] += 1
            while d >= d_t:
                res = min([res, s[l: r + 1]], key = len)
                d[s[l]] -= 1
                l += 1
        return res if res != s + '#' else ''
```

```python
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        d_s, d_t = Counter(), Counter(t)
        l, res = 0, s + '$'
        count, d_t_length = 0, len(d_t)
        for r, c in enumerate(s):
            d_s[c] += 1
            if d_s[c] == d_t[c]:
                count += 1
            while count == d_t_length:
                res = min(res, s[l: r + 1], key = len)
                d_s[s[l]] -= 1
                if d_s[s[l]] < d_t[s[l]]:
                    count -= 1
                l += 1
        return res if res != s + '$' else ''
```


### 1852. Distinct Numbers in Each Subarray

```python
class Solution:
    def distinctNumbers(self, nums: List[int], k: int) -> List[int]:
        d, l, res = defaultdict(int), 0, []
        for r, n in enumerate(nums):
            d[n] += 1
            if r - l + 1 == k:
                res.append(len(d))
                d[nums[l]] -= 1
                if d[nums[l]] == 0:
                    d.pop(nums[l])
                l += 1
        return res
```

### 2302. Count Subarrays With Score Less Than K

```python
class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        l, res = 0, 0
        total = 0
        for r, n in enumerate(nums):
            total += n
            while total * (r - l + 1) >= k:
                total -= nums[l]
                l += 1
            res += r - l + 1
        return res
```

### 480. Sliding Window Median

```python
from sortedcontainers import SortedList
class Solution:
    def medianSlidingWindow(self, nums: List[int], k: int) -> List[float]:
        sl = SortedList()
        l, res = 0, []
        for r, n in enumerate(nums):
            sl.add(n)
            if r - l + 1 == k:
                L = len(sl)
                if (r - l + 1) % 2 == 1:
                    res.append(sl[L // 2])
                else:
                    res.append((sl[L // 2 - 1] + sl[L // 2]) / 2)
                sl.remove(nums[l])
                l += 1
        return res
```

### 632. Smallest Range Covering Elements from K Lists

- sort + counter + sliding window

```python
class Solution:
    def smallestRange(self, nums: List[List[int]]) -> List[int]:
        n = len(nums)
        d = Counter()
        arr = []
        for i, num in enumerate(nums):
            for j, x in enumerate(num):
                arr.append((x, i))
        arr.sort()
        
        l, res = 0, [-inf, inf]
        for r, (num, i) in enumerate(arr):
            d[i] += 1
            while len(d) == n:
                if arr[r][0] - arr[l][0] < res[1] - res[0]:
                    res[0], res[1] = arr[l][0], arr[r][0]
                d[arr[l][1]] -= 1
                if d[arr[l][1]] == 0:
                    d.pop(arr[l][1])
                l += 1
        return res
```

### 2090. K Radius Subarray Averages

```python
class Solution:
    def getAverages(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        l, res = 0, [-1] * n
        total = sum(nums[: 2 * k])
        for r in range(k, n - k):
            total += nums[r + k]
            res[r] = total // (2 * k + 1)
            total -= nums[r - k]
        return res
```

### 2743. Count Substrings Without Repeating Character

```python
class Solution:
    def numberOfSpecialSubstrings(self, s: str) -> int:
        d = Counter()
        l, res = 0, 0
        for r, c in enumerate(s):
            d[c] += 1
            while d[c] > 1:
                d[s[l]] -= 1
                l += 1
            res += r - l + 1
        return res
```

### 1658. Minimum Operations to Reduce X to Zero

```python
class Solution:
    def minOperations(self, nums: List[int], x: int) -> int:
        n = len(nums)
        total = sum(nums)
        target = total - x
        l, res, ans = 0, -inf, 0
        if target == 0:
            return n
        for r, y in enumerate(nums):
            ans += y
            while ans > target and l < r:
                ans -= nums[l]
                l += 1
            if ans == target:
                res = max(res, r - l + 1)
        return n - res if res != -inf else -1
```







### 1839. Longest Substring Of All Vowels in Order


```python
class Solution:
    def longestBeautifulSubstring(self, word: str) -> int:    
        d = set()
        l, res = 0, 0
        for r, c in enumerate(word):
            d.add(c)
            if r > 0 and word[r] < word[r - 1]:
                # for i in range(l, r):
                #     d[word[i]] -= 1
                #     if d[word[i]] == 0:
                #         d.pop(word[i])
                l = r
                d = set([c])
            if len(d) == 5:
                res = max(res, r - l + 1)
        return res
```

### 718. Maximum Length of Repeated Subarray

```python
class Solution:
    def findLength(self, nums1: List[int], nums2: List[int]) -> int:
        R, C = len(nums1) + 1, len(nums2) + 1
        dp = [[0] * C for r in range(R)]
        res = 0
        for r in range(1, R):
            for c in range(1, C):
                if nums1[r - 1] == nums2[c - 1]:
                    dp[r][c] = dp[r - 1][c - 1] + 1
                    res = max(res, dp[r][c])
        return res
```

### sliding window + counter with variable window

### 2537. Count the Number of Good Subarrays

```python
class Solution:
    def countGood(self, nums: List[int], k: int) -> int:
        cnt = Counter()
        ans = left = pairs = 0
        for x in nums:
            cnt[x] += 1  # 移入右端点
            if cnt[x] > 1:
                pairs += cnt[x] - 1
            while pairs - (cnt[nums[left]] - 1) >= k:
                if cnt[nums[left]] > 1:
                    pairs -= cnt[nums[left]] - 1
                cnt[nums[left]] -= 1  # 移出左端点
                left += 1
            if pairs >= k:
                ans += left + 1
        return ans
```

```python
class Solution:
    def countGood(self, nums: List[int], k: int) -> int:
        d = Counter()
        res = l = pairs = 0
        for n in nums:
            pairs += d[n]
            d[n] += 1
            while pairs - d[nums[l]] + 1 >= k:
                d[nums[l]] -= 1
                pairs -= d[nums[l]]
                l += 1
            if pairs >= k:
                res += l + 1
        return res
```

### 2962. Count Subarrays Where Max Element Appears at Least K Times

- same as 2537: variable window

```python
class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        mx, l, times, res = max(nums), 0, 0, 0
        for n in nums:
            if n == mx:
                times += 1
            cnt = 0 if nums[l] != mx else 1
            while times - cnt >= k:
                times -= cnt 
                l += 1
                cnt = 0 if nums[l] != mx else 1
            if times >= k:
                res += l + 1
        return res
```

### 1151. Minimum Swaps to Group All 1's Together

- fix window

```python
class Solution:
    def minSwaps(self, data: List[int]) -> int:
        window = data.count(1)
        zeros = 0
        l, res, n = 0, inf, len(data)
        for r in range(n):
            if data[r] == 0:
                zeros += 1
            if r - l + 1 == window:
                res = min(res, zeros)
                if data[l] == 0:
                    zeros -= 1
                l += 1
        return res if res != inf else 0
```


### 2107. Number of Unique Flavors After Sharing K Candies

```python
class Solution:
    def shareCandies(self, candies: List[int], k: int) -> int:
        res = -inf
        d = Counter(candies)
        l, n = 0, len(candies)
        for r in range(n):
            d[candies[r]] -= 1
            if d[candies[r]] == 0:
                d.pop(candies[r])
            if r - l + 1 == k:
                res = max(res, len(d))
                d[candies[l]] += 1
                l += 1
        return res if res != -inf else len(set(candies))
```

### 1031. Maximum Sum of Two Non-Overlapping Subarrays

- 2 sum: prefix sum

```python
class Solution:
    def maxSumTwoNoOverlap(self, nums: List[int], firstLen: int, secondLen: int) -> int:
        s = list(accumulate(nums, initial = 0))
        def f(a, b):
            maxSumA = 0
            for i in range(a + b, len(s)):
                maxSumA = max(maxSumA, s[i - b] - s[i - b - a])
                self.res = max(self.res, maxSumA + s[i] - s[i - b])
        self.res = 0
        f(firstLen, secondLen)
        f(secondLen, firstLen)
        return self.res
```

### 2555. Maximize Win From Two Segments

```python
class Solution:
    def maximizeWin(self, prizePositions: List[int], k: int) -> int:
        pre = [0] * (len(prizePositions) + 1)
        res = l = 0
        for r, p in enumerate(prizePositions):
            while p - prizePositions[l] > k:
                l += 1
            res = max(res, r - l + 1 + pre[l])
            pre[r + 1] = max(pre[r], r - l + 1)
        return res
```

### 2516. Take K of Each Character From Left and Right

```python
class Solution:
    def takeCharacters(self, s: str, k: int) -> int:
        d = Counter(s)
        for key in 'abc':
            d[key] -= k 
            if d[key] < 0:
                return -1

        l, res, cnt = 0, 0, Counter()
        for r, c in enumerate(s):
            cnt[c] += 1
            while cnt[c] > d[c]:
                cnt[s[l]] -= 1
                l += 1
            res = max(res, r - l + 1)
        return len(s) - res
```

### 1918. Kth Smallest Subarray Sum

- binary search + sliding window

```python
class Solution:
    def kthSmallestSubarraySum(self, nums: List[int], k: int) -> int:
        def check(threshold):
            count = 0
            l, val = 0, 0
            for r, n in enumerate(nums):
                val += n 
                while val > threshold:
                    val -= nums[l]
                    l += 1
                count += r - l + 1
            return count >= k
            
        l, r, res = min(nums), sum(nums), 0
        while l <= r:
            m = l + (r - l) // 2
            if check(m):
                res = m
                r = m - 1
            else:
                l = m + 1
        return res
```

### 1888. Minimum Number of Flips to Make the Binary String Alternating

- 2 * s pattern + sliding window

```python
class Solution:
    def minFlips(self, s: str) -> int:
        s = s * 2
        n = len(s)
        A, B, m = '', '', n // 2
        for i in range(n):
            A += '0' if i % 2 == 0 else '1'
            B += '1' if i % 2 == 0 else '0'

        def check(A):
            l, res, count = 0, inf, 0
            for r, c in enumerate(s):
                if A[r] != c:
                    count += 1
                if r - l + 1 > m:
                    if A[l] != s[l]:
                        count -= 1
                    l += 1
                if r - l + 1 == m:
                    res = min(res, count)
            return res 
        return min(check(A), check(B))
```

### 1248. Count Number of Nice Subarrays

- 2 while sliding window

```python
class Solution:
    def numberOfSubarrays(self, nums: List[int], k: int) -> int:
        odd = 0
        for n in nums:
            if n % 2:
                odd += 1
        if odd < k:
            return 0
        
        l, res, count, even = 0, 0, 0, 0
        for r, n in enumerate(nums):
            if n % 2:
                count += 1
            while count > k:
                if nums[l] % 2:
                    count -= 1
                l += 1
                even = 0
            
            while count == k and nums[l] % 2 == 0:
                even += 1
                l += 1
            if count == k:
                res += even + 1
                
        return res
```

### 1838. Frequency of the Most Frequent Element

- greedy + sliding window

```python
class Solution:
    def maxFrequency(self, nums: List[int], k: int) -> int:
        nums.sort()
        l, res, total = 0, 0, 0
        for r, n in enumerate(nums):
            total += n 
            while n * (r - l + 1) > total + k:
                total -= nums[l]
                l += 1
            res = max(res, r - l + 1)
        return res
```

### 1234. Replace the Substring for Balanced String

- reverse check

```python
class Solution:
    def balancedString(self, s: str) -> int:
        def check():
            return all(v <= n // 4 for v in d.values())
        d, n = Counter(s), len(s)
        if check():
            return 0
        
        l, res = 0, n
        for r, c in enumerate(s):
            d[c] -= 1
            while check():
                res = min(res, r - l + 1)
                d[s[l]] += 1
                l += 1
        return res
```

### 1477. Find Two Non-overlapping Sub-arrays Each With Target Sum

- dp + sliding window

```python
class Solution:
    def minSumOfLengths(self, arr: List[int], target: int) -> int:
        pre = [inf] * (len(arr) + 1)
        res, l, total = inf, 0, 0
        for r, n in enumerate(arr):
            total += n
            while total > target:
                total -= arr[l]
                l += 1
            if total == target:
                res = min(res, r - l + 1 + pre[l])
                pre[r + 1] = min(pre[r], r - l + 1)
            else:
                pre[r + 1] = pre[r]
        return res if res != inf else -1
```

### 1871. Jump Game VII

- bfs

```python
class Solution:
    def canReach(self, s: str, minJump: int, maxJump: int) -> bool:
        q, furthest, n = deque([0]), 0, len(s)
        while q:
            i = q.popleft()
            start = max(i + minJump, furthest + 1)
            for j in range(start, min(i + maxJump + 1, n)):
                if s[j] == '0':
                    q.append(j)
                    if j == n - 1:
                        return True
            furthest = i + maxJump
        return False
```

### 727. Minimum Window Subsequence

```python
class Solution:
    def minWindow(self, s1: str, s2: str) -> str:
        i, j = 0, 0
        l, r, mn = 0, 0, inf
        while i < len(s1):
            if s1[i] == s2[j]:
                j += 1
            if j == len(s2):
                r = i
                j -= 1
                while j >= 0:
                    if s1[i] == s2[j]:
                        j -= 1
                    i -= 1
                i += 1
                if r - i + 1 < mn:
                    l = i
                    mn = r - l + 1
                j = 0
            i += 1
        return "" if mn == inf else s1[l: l + mn]
```

### 2799. Count Complete Subarrays in an Array

```python
class Solution:
    def countCompleteSubarrays(self, nums: List[int]) -> int:
        L = len(set(nums))
        l, res = 0, 0
        c, n = Counter(), len(nums)
        for r, v in enumerate(nums):
            c[v] += 1
            while len(c) == L:
                res += n - r 
                c[nums[l]] -= 1
                if c[nums[l]] == 0:
                    c.pop(nums[l])
                l += 1
        return res
```

### 992. Subarrays with K Different Integers

```python
class Solution:
    def subarraysWithKDistinct(self, nums: List[int], k: int) -> int:
        def atMost(k):
            l, res = 0, 0
            n, c = len(nums), Counter()
            r = 0
            for r, v in enumerate(nums):
                c[v] += 1
                while len(c) > k:
                    c[nums[l]] -= 1
                    if c[nums[l]] == 0:
                        c.pop(nums[l])
                    l += 1
                res += r - l + 1
            return res
        return atMost(k) - atMost(k - 1)
```

### 

```python
class Solution:
    def maxTotalFruits(self, fruits: List[List[int]], startPos: int, k: int) -> int:
        n = 10 ** 6
        prefix = [0] * n
        for p, c in fruits:
            prefix[p + 1] = c
        prefix = list(accumulate(prefix))
        startPos += 1
        res = 0
        for i in range(startPos, startPos + k + 1):
            right = i 
            left = min(startPos - (k - (i - startPos) * 2), startPos)
            res = max(res, prefix[right] - prefix[left - 1])
            # print(res, left, right)
        for i in range(startPos, max(0, startPos - k - 1), -1):
            left = i 
            right = max(startPos + (k - (startPos - i) * 2), startPos)
            res = max(res, prefix[right] - prefix[left - 1])
        return res
```

### 2106. Maximum Fruits Harvested After at Most K Steps

```python
class Solution:
    def maxTotalFruits(self, fruits: List[List[int]], startPos: int, k: int) -> int:
        n = max(a for a, b in fruits)
        prefix = [0] * (n + k + startPos + 2)
        for p, c in fruits:
            prefix[p + 1] = c
        prefix = list(accumulate(prefix))
        startPos += 1
        res = 0
        for i in range(0, k + 1):
            a = max(min(startPos - k + 2 * i, startPos), 0)
            b = startPos + i
            c = max(startPos - i, 0)
            d = max(startPos, startPos + k - 2 * i)
            if a >= 1 or c >= 1:
                res = max(res, prefix[b] - prefix[a - 1], prefix[d] - prefix[c - 1])
            if a == 0:
                res = max(res, prefix[b])
            if c == 0:
                res = max(res, prefix[d])
        return res
```

### 2398. Maximum Number of Robots Within Budget

```python
from sortedcontainers import SortedList
class Solution:
    def maximumRobots(self, chargeTimes: List[int], runningCosts: List[int], budget: int) -> int:
        def check(threshold):
            n = len(chargeTimes)
            l, res, sl, total = 0, 0, SortedList(), 0
            for r, c in enumerate(chargeTimes):
                sl.add(c)
                total += runningCosts[r]
                if r - l + 1 == threshold:
                    if sl[-1] + threshold * total <= budget:
                        return True
                    sl.remove(chargeTimes[l])
                    total -= runningCosts[l]
                    l += 1
            return False

        l, r, res = 1, len(chargeTimes), 0
        while l <= r:
            m = l + (r - l) // 2
            if check(m):
                res = m 
                l = m + 1
            else:
                r = m - 1
        return res
```

### 2398. Maximum Number of Robots Within Budget

```python
from sortedcontainers import SortedList
class Solution:
    def maximumRobots(self, chargeTimes: List[int], runningCosts: List[int], budget: int) -> int:
        n = len(chargeTimes)
        l, res, sl, total = 0, 0, SortedList(), 0
        for r, c in enumerate(chargeTimes):
            sl.add(c)
            total += runningCosts[r]
            L = r - l + 1
            while sl and sl[-1] + L * total > budget:
                sl.remove(chargeTimes[l])
                total -= runningCosts[l]
                l += 1
            res = max(res, r - l + 1)
        return res

# monotonic queue

class Solution:
    def maximumRobots(self, chargeTimes: List[int], runningCosts: List[int], budget: int) -> int:
        res = s = l = 0
        q = deque()
        for r, (t, c) in enumerate(zip(chargeTimes, runningCosts)):
            while q and t >= chargeTimes[q[-1]]:
                q.pop()
            q.append(r)
            s += c
            while q and chargeTimes[q[0]] + (r - l  + 1) * s > budget:
                if q[0] == l:
                    q.popleft()
                s -= runningCosts[l]
                l += 1
            res = max(res, r - l + 1)
        return res
```

### 3034. Number of Subarrays That Match a Pattern I

```python
class Solution:
    def countMatchingSubarrays(self, nums: List[int], pattern: List[int]) -> int:
        def check(arr):
            i, j = 0, 0
            while j < len(pattern):
                if pattern[j] == 1 and arr[i] >= arr[i + 1]:
                    return False
                elif pattern[j] == 0 and arr[i] != arr[i + 1]:
                    return False
                elif pattern[j] == -1 and arr[i] <= arr[i + 1]:
                    return False
                j += 1
                i += 1
            return True
        n, m = len(nums), len(pattern)
        res = 0
        for i in range(n - m):
            if check(nums[i: i + m + 1]):
                res += 1
        return res
```

### 3090. Maximum Length Substring With Two Occurrences

```python
class Solution:
    def maximumLengthSubstring(self, s: str) -> int:
        res = 0
        l = 0
        d = defaultdict(int)
        for r, c in enumerate(s):
            d[c] += 1
            while d[c] > 2:
                d[s[l]] -= 1
                l += 1
            res = max(res, r - l + 1)
        return res 
```

