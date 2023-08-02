### 2733. Neither Minimum nor Maximum

```python
class Solution:
    def findNonMinOrMax(self, nums: List[int]) -> int:
        minNum, maxNum, res = min(nums), max(nums), []
        for n in nums:
            if n not in [minNum, maxNum]:
                return n
        return -1
```

### 1086. 前五科的均分

给你一个不同学生的分数列表 items，其中 items[i] = [IDi, scorei] 表示 IDi 的学生的一科分数，你需要计算每个学生 最高的五科 成绩的 平均分。

返回答案 result 以数对数组形式给出，其中 result[j] = [IDj, topFiveAveragej] 表示 IDj 的学生和他 最高的五科 成绩的 平均分。result 需要按 IDj  递增的 顺序排列 。

学生 最高的五科 成绩的 平均分 的计算方法是将最高的五科分数相加，然后用 整数除法 除以 5 。
 
示例 1：

输入：items = [[1,91],[1,92],[2,93],[2,97],[1,60],[2,77],[1,65],[1,87],[1,100],[2,100],[2,76]]
输出：[[1,87],[2,88]]
解释：
ID = 1 的学生分数为 91、92、60、65、87 和 100 。前五科的平均分 (100 + 92 + 91 + 87 + 65) / 5 = 87
ID = 2 的学生分数为 93、97、77、100 和 76 。前五科的平均分 (100 + 97 + 93 + 77 + 76) / 5 = 88.6，但是由于使用整数除法，结果转换为 88

```python
class Solution:
    def highFive(self, items: List[List[int]]) -> List[List[int]]:
        d = defaultdict(list)
        for id, score in items:
            d[id].append(score)

        res = []
        for id, scores in d.items():
            res.append([id, sum(sorted(scores, reverse = True)[: 5]) // 5])
        res.sort()
        return res
```

### 1502. Can Make Arithmetic Progression From Sequence

```python
class Solution:
    def canMakeArithmeticProgression(self, arr: List[int]) -> bool:
        s = set()
        arr.sort()
        for i in range(1, len(arr)):
            s.add(arr[i] - arr[i - 1])
        return True if len(s) == 1 else False
```

### 1133. 最大唯一数

```python
class Solution:
    def canMakeArithmeticProgression(self, arr: List[int]) -> bool:
        s = set()
        arr.sort()
        for i in range(1, len(arr)):
            s.add(arr[i] - arr[i - 1])
        return True if len(s) == 1 else False
```

### 1196. 最多可以买到的苹果数量

```python
class Solution:
    def maxNumberOfApples(self, weight: List[int]) -> int:
        weight.sort()
        res, count = 0, 0
        for w in weight:
            if w + res <= 5000:
                res += w
                count += 1
            else:
                break
        return count
```

### 1065. 字符串的索引对

```python
class Solution:
    def indexPairs(self, text: str, words: List[str]) -> List[List[int]]:
        res, n = [], len(text)
        for word in words:
            word_length = len(word)
            for i in range(n):
                if word == text[i: i + word_length]:
                    res.append([i, i + word_length - 1])
        res.sort()
        return res
```

### 2706. Buy Two Chocolates

```python
class Solution:
    def buyChoco(self, prices: List[int], money: int) -> int:
        prices.sort()
        if money >= sum(prices[: 2]):
            return money - sum(prices[: 2])
        return money
```

### 1099. 小于 K 的两数之和

```python
class Solution:
    def twoSumLessThanK(self, nums: List[int], k: int) -> int:
        l, r, res = 0, len(nums) - 1, -1
        nums.sort()
        while l < r:
            two = nums[l] + nums[r]
            if two < k:
                res = max(res, two)
            if two >= k:
                r -= 1
            else:
                l += 1
        return res
```

### 252. 会议室

```python
class Solution:
    def canAttendMeetings(self, intervals: List[List[int]]) -> bool:
        intervals.sort()
        n = len(intervals)
        for i in range(1, n):
            if intervals[i][0] < intervals[i - 1][1]:
                return False
        return True
```

### 1984. Minimum Difference Between Highest and Lowest of K Scores

```python
class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        if k == 1:
            return 0
        nums.sort()
        res = min([nums[i] - nums[i - k + 1] for i in range(k - 1, len(nums))])
        return res
```

### 1874. 两个数组的最小乘积和

```python
class Solution:
    def minProductSum(self, nums1: List[int], nums2: List[int]) -> int:
        nums1.sort()
        nums2.sort(reverse = True)
        res = 0
        for n1, n2 in zip(nums1, nums2):
            res += n1 * n2
        return res
```

### 1637. Widest Vertical Area Between Two Points Containing No Points

```python
class Solution:
    def maxWidthOfVerticalArea(self, points: List[List[int]]) -> int:
        res = []
        for x, y in points:
            res.append(x)
        res.sort()
        dist = [res[i] - res[i - 1] for i in range(1, len(res))]
        return max(dist)
```

### 1630. Arithmetic Subarrays

```python
class Solution:
    def checkArithmeticSubarrays(self, nums: List[int], l: List[int], r: List[int]) -> List[bool]:
        def isArithmetic(nums):
            nums.sort()
            nums = [nums[i] - nums[i - 1] for i in range(1, len(nums))]
            if len(set(nums)) == 1:
                return True
            else:
                return False
        res = []
        for start, end in zip(l, r):
            if isArithmetic(nums[start: end + 1]):
                res.append(True)
            else:
                res.append(False)
        return res
```

### 2268. 最少按键次数

```python
class Solution:
    def minimumKeypresses(self, s: str) -> int:
        ans = [v for k, v in Counter(s).items()]
        ans.sort(reverse = True)
        res = sum(ans[: 9]) + sum(ans[9: 18]) * 2 + sum(ans[18: ]) * 3
        return res
```

### 2733. Neither Minimum nor Maximum

```python
class Solution:
    def findNonMinOrMax(self, nums: List[int]) -> int:
        minNum, maxNum, res = min(nums), max(nums), []
        for n in nums:
            if n not in [minNum, maxNum]:
                return n
        return -1
```

### 1086. 前五科的均分

给你一个不同学生的分数列表 items，其中 items[i] = [IDi, scorei] 表示 IDi 的学生的一科分数，你需要计算每个学生 最高的五科 成绩的 平均分。

返回答案 result 以数对数组形式给出，其中 result[j] = [IDj, topFiveAveragej] 表示 IDj 的学生和他 最高的五科 成绩的 平均分。result 需要按 IDj  递增的 顺序排列 。

学生 最高的五科 成绩的 平均分 的计算方法是将最高的五科分数相加，然后用 整数除法 除以 5 。
 
示例 1：

输入：items = [[1,91],[1,92],[2,93],[2,97],[1,60],[2,77],[1,65],[1,87],[1,100],[2,100],[2,76]]
输出：[[1,87],[2,88]]
解释：
ID = 1 的学生分数为 91、92、60、65、87 和 100 。前五科的平均分 (100 + 92 + 91 + 87 + 65) / 5 = 87
ID = 2 的学生分数为 93、97、77、100 和 76 。前五科的平均分 (100 + 97 + 93 + 77 + 76) / 5 = 88.6，但是由于使用整数除法，结果转换为 88

```python
class Solution:
    def highFive(self, items: List[List[int]]) -> List[List[int]]:
        d = defaultdict(list)
        for id, score in items:
            d[id].append(score)

        res = []
        for id, scores in d.items():
            res.append([id, sum(sorted(scores, reverse = True)[: 5]) // 5])
        res.sort()
        return res
```

### 1502. Can Make Arithmetic Progression From Sequence

```python
class Solution:
    def canMakeArithmeticProgression(self, arr: List[int]) -> bool:
        s = set()
        arr.sort()
        for i in range(1, len(arr)):
            s.add(arr[i] - arr[i - 1])
        return True if len(s) == 1 else False
```

### 1133. 最大唯一数

```python
class Solution:
    def canMakeArithmeticProgression(self, arr: List[int]) -> bool:
        s = set()
        arr.sort()
        for i in range(1, len(arr)):
            s.add(arr[i] - arr[i - 1])
        return True if len(s) == 1 else False
```

### 1196. 最多可以买到的苹果数量

```python
class Solution:
    def maxNumberOfApples(self, weight: List[int]) -> int:
        weight.sort()
        res, count = 0, 0
        for w in weight:
            if w + res <= 5000:
                res += w
                count += 1
            else:
                break
        return count
```

### 1065. 字符串的索引对

```python
class Solution:
    def indexPairs(self, text: str, words: List[str]) -> List[List[int]]:
        res, n = [], len(text)
        for word in words:
            word_length = len(word)
            for i in range(n):
                if word == text[i: i + word_length]:
                    res.append([i, i + word_length - 1])
        res.sort()
        return res
```

### 2706. Buy Two Chocolates

```python
class Solution:
    def buyChoco(self, prices: List[int], money: int) -> int:
        prices.sort()
        if money >= sum(prices[: 2]):
            return money - sum(prices[: 2])
        return money
```

### 1099. 小于 K 的两数之和

```python
class Solution:
    def twoSumLessThanK(self, nums: List[int], k: int) -> int:
        l, r, res = 0, len(nums) - 1, -1
        nums.sort()
        while l < r:
            two = nums[l] + nums[r]
            if two < k:
                res = max(res, two)
            if two >= k:
                r -= 1
            else:
                l += 1
        return res
```

### 252. 会议室

```python
class Solution:
    def canAttendMeetings(self, intervals: List[List[int]]) -> bool:
        intervals.sort()
        n = len(intervals)
        for i in range(1, n):
            if intervals[i][0] < intervals[i - 1][1]:
                return False
        return True
```

### 1984. Minimum Difference Between Highest and Lowest of K Scores

```python
class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        if k == 1:
            return 0
        nums.sort()
        res = min([nums[i] - nums[i - k + 1] for i in range(k - 1, len(nums))])
        return res
```

### 1874. 两个数组的最小乘积和

```python
class Solution:
    def minProductSum(self, nums1: List[int], nums2: List[int]) -> int:
        nums1.sort()
        nums2.sort(reverse = True)
        res = 0
        for n1, n2 in zip(nums1, nums2):
            res += n1 * n2
        return res
```

### 1637. Widest Vertical Area Between Two Points Containing No Points

```python
class Solution:
    def maxWidthOfVerticalArea(self, points: List[List[int]]) -> int:
        res = []
        for x, y in points:
            res.append(x)
        res.sort()
        dist = [res[i] - res[i - 1] for i in range(1, len(res))]
        return max(dist)
```

### 1630. Arithmetic Subarrays

```python
class Solution:
    def checkArithmeticSubarrays(self, nums: List[int], l: List[int], r: List[int]) -> List[bool]:
        def isArithmetic(nums):
            nums.sort()
            nums = [nums[i] - nums[i - 1] for i in range(1, len(nums))]
            if len(set(nums)) == 1:
                return True
            else:
                return False
        res = []
        for start, end in zip(l, r):
            if isArithmetic(nums[start: end + 1]):
                res.append(True)
            else:
                res.append(False)
        return res
```
### 2517. Maximum Tastiness of Candy Basket

```python
class Solution:
    def maximumTastiness(self, price: List[int], k: int) -> int:
        def isFind(x):
            prev, count, i = price[0], 1, 1
            while i < n:
                if price[i] - prev >= x:
                    prev, count = price[i], count + 1
                i += 1
                if count >= k:
                    return True
            return False
        
        price.sort()
        n = len(price)
        l, r = 1, 10 ** 9
        while l <= r:
            m = l + (r - l) // 2
            if isFind(m):
                l = m + 1
            else:
                r = m - 1
        return r
```

### 698. Partition to K Equal Sum Subsets

same as 1723, 2305

```python
class Solution:
    def canPartitionKSubsets(self, nums: List[int], k: int) -> bool:
        if sum(nums) // k != sum(nums) / k:
            return False
        nums.sort(reverse = True)
        parts = [sum(nums) // k] * k
        n, k = len(nums), len(parts)
        def dfs(idx):
            if idx == n:
                return not sum(parts)
            flag = 0
            for i in range(k):
                if parts[i] == sum(nums) // k:
                    if flag == 1: continue
                    flag = 1
                if parts[i] >= nums[idx]:
                    parts[i] -= nums[idx]
                    if dfs(idx + 1):
                        return True
                    parts[i] += nums[idx]
        return dfs(0)
```

### 1723. Find Minimum Time to Finish All Jobs

```python
class Solution:
    def minimumTimeRequired(self, jobs: List[int], k: int) -> int:
        jobs.sort(reverse = True)
        plan, n = [0] * k, len(jobs)
        def dfs(curJobs, limit): # cookies
            if curJobs == n:
                return True
            flag = 0 # trim for first have jobs from no jobs of anyone
            for i in range(k): # children
                if plan[i] + jobs[curJobs] > limit:
                    continue
                if plan[i] == 0:
                    if flag == 1: continue
                    flag = 1
                plan[i] += jobs[curJobs]
                if dfs(curJobs + 1, limit):
                    return True
                plan[i] -= jobs[curJobs]
            return False
        l, r = max(jobs), sum(jobs)
        while l < r:
            for i in range(k):
                plan[i] = 0
            m = l + (r - l) // 2
            if dfs(0, m):
                r = m
            else:
                l = m + 1
        return l
```

### 2305. Fair Distribution of Cookies

```python
class Solution:
    def distributeCookies(self, cookies: List[int], k: int) -> int:
        cookies.sort(reverse = True)
        plan, n = [0] * k, len(cookies)
        def dfs(curCookies, limit): # cookies
            if curCookies == n:
                return True
            for i in range(k): # children
                if plan[i] + cookies[curCookies] > limit:
                    continue
                plan[i] += cookies[curCookies]
                if dfs(curCookies + 1, limit):
                    return True
                plan[i] -= cookies[curCookies]
            return False
        l, r = max(cookies), sum(cookies)
        while l < r:
            for i in range(k):
                plan[i] = 0
            m = l + (r - l) // 2
            if dfs(0, m):
                r = m
            else:
                l = m + 1
        return l
```