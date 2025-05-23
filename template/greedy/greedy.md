## greedy

## 1 min-max

* [1833. Maximum Ice Cream Bars](#1833-maximum-ice-cream-bars)
* [3074. Apple Redistribution into Boxes](#3074-apple-redistribution-into-boxes)
* [2279. Maximum Bags With Full Capacity of Rocks](#2279-maximum-bags-with-full-capacity-of-rocks)
* [1005. Maximize Sum Of Array After K Negations](#1005-maximize-sum-of-array-after-k-negations)
* [1481. Least Number of Unique Integers after K Removals](#1481-least-number-of-unique-integers-after-k-removals)
* [1403. Minimum Subsequence in Non-Increasing Order](#1403-minimum-subsequence-in-non-increasing-order)
* [3010. Divide an Array Into Subarrays With Minimum Cost I](#3010-divide-an-array-into-subarrays-with-minimum-cost-i)
* [1338. Reduce Array Size to The Half](#1338-reduce-array-size-to-the-half)
* [1262. Greatest Sum Divisible by Three](#1262-greatest-sum-divisible-by-three)
* [948. Bag of Tokens](#948-bag-of-tokens)
* [1775. Equal Sum Arrays With Minimum Number of Operations](#1775-equal-sum-arrays-with-minimum-number-of-operations)

## prefix sum

* [1788. Maximize the Beauty of the Garden](#1788-maximize-the-beauty-of-the-garden)
* [1727. Largest Submatrix With Rearrangements](#1727-largest-submatrix-with-rearrangements)
* [2087. Minimum Cost Homecoming of a Robot in a Grid](#2087-minimum-cost-homecoming-of-a-robot-in-a-grid)
* [1526. Minimum Number of Increments on Subarrays to Form a Target Array](#1526-minimum-number-of-increments-on-subarrays-to-form-a-target-array)
* [1996. The Number of Weak Characters in the Game](#1996-the-number-of-weak-characters-in-the-game)

### 1788. Maximize the Beauty of the Garden

```python
class Solution:
    def maximumBeauty(self, flowers: List[int]) -> int:
        n = len(flowers)
        pre = [0] * (n + 1)
        right = {}
        for i, v in enumerate(flowers):
            pre[i + 1] = pre[i] + max(v, 0)
            right[v] = i  
    
        ans = -inf
        for i, v in enumerate(flowers):
            if right[v] > i:
                ans = max(ans, v * 2 + pre[right[v]] - pre[i + 1])
                right[v] = i
        return ans
```

### 1727. Largest Submatrix With Rearrangements

```python
class Solution:
    def largestSubmatrix(self, matrix: List[List[int]]) -> int:
        R, C = len(matrix), len(matrix[0])
        for r in range(1, R):
            for c in range(C):
                if matrix[r][c]:
                    matrix[r][c] += matrix[r - 1][c]
        
        def check(nums):
            n, res = len(nums), 0
            for i, v in enumerate(nums):
                res = max(res, v * (n - i))
            return res 

        res = 0
        for row in matrix:
            row = sorted(row)
            res = max(res, check(row))
        return res
```

### 2087. Minimum Cost Homecoming of a Robot in a Grid

```python
class Solution:
    def minCost(self, startPos: List[int], homePos: List[int], rowCosts: List[int], colCosts: List[int]) -> int:
        sx, sy = startPos
        hx, hy = homePos
        colSum = sum(colCosts[sy + 1: hy + 1]) if sy <= hy else sum(colCosts[hy: sy])
        rowSum = sum(rowCosts[sx + 1: hx + 1]) if sx <= hx else sum(rowCosts[hx: sx])
        return colSum + rowSum
```

### 1526. Minimum Number of Increments on Subarrays to Form a Target Array

```python
class Solution:
    def minNumberOperations(self, target: List[int]) -> int:
        n = len(target)
        f = [0 for _ in range(n)]
        f[0] = target[0]
        for i in range(1, n):
            if target[i] <= target[i - 1]:
                f[i] = f[i - 1]
            else:
                f[i] = f[i - 1] + target[i] - target[i - 1]
        return f[-1]
```

### 1996. The Number of Weak Characters in the Game

```python
class Solution:
    def numberOfWeakCharacters(self, properties: List[List[int]]) -> int:
        properties.sort(key=lambda p: (-p[0], p[1]))
        res = max_def = 0
        for _, d in properties:
            if d < max_def: 
                res += 1
            else: 
                max_def = d
        return res
```

### 2178. Maximum Split of Positive Even Integers

```python
class Solution:
    def maximumEvenSplit(self, finalSum: int) -> List[int]:
        if finalSum % 2:
            return []
        res = []
        step = 2
        while finalSum:
            if step <= finalSum:
                finalSum -= step
                res.append(step)
                step += 2
            else:
                res[-1] += finalSum
                break
        return res
```

### 2567. Minimum Score by Changing Two Elements

```python
class Solution:
    def minimizeSum(self, nums: List[int]) -> int:
        nums.sort()
        a, b, c = nums[-1] - nums[2], nums[-2] - nums[1], nums[-3] - nums[0]
        return min(a, b, c)
```

### 1509. Minimum Difference Between Largest and Smallest Value in Three Moves

```python
class Solution:
    def minDifference(self, nums: List[int]) -> int:
        if len(nums) <= 4:
            return 0
        nums.sort()
        a = nums[-1] - nums[3]
        b = nums[-4] - nums[0]
        c = nums[-3] - nums[1]
        d = nums[-2] - nums[2]
        return min(a, b, c, d)
```

### 1833. Maximum Ice Cream Bars

```python
class Solution:
    def maxIceCream(self, costs: List[int], coins: int) -> int:
        freq = [0] * (10 ** 5 + 1)
        for c in costs:
            freq[c] += 1
        cnt = 0
        for i in range(1, 10 ** 5 + 1):
            if coins >= i:
                mn = min(freq[i], coins // i)
                cnt += mn
                coins -= i * mn
            else:
                break
        return cnt
```

### 3074. Apple Redistribution into Boxes

```python
class Solution:
    def minimumBoxes(self, apple: List[int], capacity: List[int]) -> int:
        total = sum(apple)
        capacity.sort(reverse = True)
        cnt = 0
        for i, c in enumerate(capacity):
            cnt += c 
            if cnt >= total:
                return i + 1
```

### 2279. Maximum Bags With Full Capacity of Rocks

```python
class Solution:
    def maximumBags(self, capacity: List[int], rocks: List[int], additionalRocks: int) -> int:
        diff = [c - r for c, r in zip(capacity, rocks)]
        diff.sort()
        pre = list(accumulate(diff))
        for i, n in enumerate(pre):
            if n > additionalRocks:
                return i 
        return len(rocks)
```

### 1005. Maximize Sum Of Array After K Negations

```python
class Solution:
    def largestSumAfterKNegations(self, nums: List[int], k: int) -> int:
        heapify(nums)
        while nums[0] < 0 and k > 0: # add k > 0
            heappush(nums, -heappop(nums))
            k -= 1
        if k%2 == 1:
            nums[0] = -nums[0]
        return sum(nums)
```

### 1481. Least Number of Unique Integers after K Removals

```python
class Solution:
    def findLeastNumOfUniqueInts(self, arr: List[int], k: int) -> int:
        c = Counter(arr)
        pq = []
        for key, v in c.items():
            heappush(pq, (v, key))

        while pq:
            if k - pq[0][0] >= 0:
                v, key = heappop(pq)
                k -= v
            else:
                break
        return len(pq)
```

### 1262. Greatest Sum Divisible by Three

```python
class Solution:
    def maxSumDivThree(self, nums: List[int]) -> int:
        s = sum(nums)
        if s % 3 == 0:
            return s 
        a1 = sorted([n for n in nums if n % 3 == 1])
        a2 = sorted([n for n in nums if n % 3 == 2])
        if s % 3 == 2:
            a1, a2 = a2, a1 
        res = s - a1[0] if a1 else 0 
        if len(a2) > 1:
            res = max(res, s - a2[0] - a2[1])
        return res 
```

### 948. Bag of Tokens

```python
class Solution:
    def bagOfTokensScore(self, tokens: List[int], power: int) -> int:
        tokens.sort()
        n = len(tokens)
        l, r = 0, n - 1
        score, res = 0, 0
        while l <= r:
            if power >= tokens[l]:
                score += 1
                power -= tokens[l]
                l += 1
                res = max(res, score)
            else:
                if score == 0:
                    break
                if score >= 1:
                    power += tokens[r]
                    score -= 1
                    r -= 1
        return res
```

### 1775. Equal Sum Arrays With Minimum Number of Operations

```python
class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        n1, n2 = len(nums1), len(nums2)
        mn_nums1, mx_nums1 = n1, n1 * 6
        mn_nums2, mx_nums2 = n2, n2 * 6
        if mn_nums1 > mx_nums2 or mn_nums2 > mx_nums1:
            return -1
        d = sum(nums2) - sum(nums1)
        if d < 0:
            d = -d 
            nums1, nums2 = nums2, nums1
        c = Counter([6 - x for x in nums1]) + Counter([x - 1 for x in nums2]) 
        res = 0
        for i in range(5, 0, -1):
            if i * c[i] >= d:
                return res + ceil(d / i)
            res += c[i]
            d -= i * c[i]
```

* [55. Jump Game](#55-jump-game)

### 55. Jump Game

```python
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        n = len(nums)
        furthest = nums[0]
        for i in range(1, n):
            if furthest >= i:
                furthest = max(furthest, i + nums[i])
        return furthest >= n - 1
```

### 45. Jump Game II

```python
class Solution:
    def jump(self, nums: List[int]) -> int:
        n = len(nums)
        furthest, prev, res = 0, 0, 0
        for i in range(n - 1):
            if furthest >= i:
                furthest = max(furthest, i + nums[i])
                if i == prev:
                    prev = furthest
                    res += 1
        return res
```

### 455. Assign Cookies

```python
class Solution:
    def findContentChildren(self, g: List[int], s: List[int]) -> int:
        i, j, res = 0, 0, 0
        g, s = sorted(g), sorted(s)
        while i < len(g) and j < len(s):
            if s[j] >= g[i]:
                res += 1
                i += 1
                j += 1
            else:
                j += 1
        return res
```

### 860. Lemonade Change

```python
class Solution:
    def lemonadeChange(self, bills: List[int]) -> bool:
        d = {5: 0, 10: 0}
        for bill in bills:
            if bill == 5:
                d[bill] += 1
            elif bill == 10:
                d[bill] += 1
                d[5] -= 1
            elif bill == 20:
                if d[10] > 0:
                    d[10] -= 1
                    d[5] -= 1
                else:
                    d[5] -= 3
            if d[5] < 0:
                return False
        return True
```

## sorting

### 179. Largest Number

```python
def isSwap(s1, s2):
            return int(s1 + s2) < int(s2 + s1)
        origin = nums[::]
        nums = [str(n) for n in nums]
        n = len(nums)
        for i in range(n):
            for j in range(i + 1, n):
                if isSwap(nums[i], nums[j]):
                    nums[i], nums[j] = nums[j], nums[i]
        return ''.join(nums) if sum(origin) != 0 else '0'
```

## intervals

## regret

### 767. Reorganize String

```python
class Solution:
    def reorganizeString(self, s: str) -> str:
        c = Counter(s)
        mx = max(c.values())
        if mx > len(s) // 2 + 1 and len(s) % 2:
            return ''
        if mx > len(s) // 2 and len(s) % 2 == 0:
            return ''
        res = ''
        pq = []
        for k, v in c.items():
            heappush(pq, (-v, k))
        while pq:
            v, k = heappop(pq)
            res += k
            if pq:
                v2, k2 = heappop(pq)
                res += k2 
                v2 = -v2
                if v2 - 1 != 0:
                    heappush(pq, (-(v2 - 1), k2))
            v = -v
            if v - 1 != 0:
                heappush(pq, (-(v - 1), k))
        return res
```

### 945. Minimum Increment to Make Array Unique

```python
class Solution:
    def minIncrementForUnique(self, nums: List[int]) -> int:
        nums.sort()
        res = 0
        for i in range(1, len(nums)):
            if nums[i] <= nums[i - 1]:
                res += nums[i - 1] - nums[i] + 1
                nums[i] = nums[i - 1] + 1
        return res
```

### 1090. Largest Values From Labels

```python
class Solution:
    def largestValsFromLabels(self, values: List[int], labels: List[int], numWanted: int, useLimit: int) -> int:
        d = defaultdict(list)
        for v, l in zip(values, labels):
            d[l].append(v)
        res = []
        for k, v in d.items():
            v.sort(reverse = True)
            res += v[:useLimit]
        
        res.sort(reverse = True)
        return sum(res[: numWanted])
```

## greedy stack: no duplicate and with sequence

- use greedy ideas with a stack to solve the min max problems with sequence

- 316. Remove Duplicate Letters
- 1081. Smallest Subsequence of Distinct Characters (same as 316)
- 402. Remove K Digits
- 321. Create Maximum Number

### 316. Remove Duplicate Letters

```python
class Solution:
    def removeDuplicateLetters(self, s: str) -> str:
        # acbd | abcd
        left = Counter(s)
        res = []
        res_set = set()
        for c in s:
            left[c] -= 1
            if c in res_set:
                continue
            while res and c < res[-1] and left[res[-1]]:
                res_set.remove(res.pop())
            res.append(c)
            res_set.add(c)
        return ''.join(res)
```

### 1081. Smallest Subsequence of Distinct Characters (same as 316)

```python
class Solution:
    def smallestSubsequence(self, s: str) -> str:
        res, left, res_set = [], Counter(s), set()
        for c in s:
            left[c] -= 1
            if c in res_set:
                continue
            while res and c < res[-1] and left[res[-1]]:
                res_set.remove(res.pop())
            res.append(c)
            res_set.add(c)
        return ''.join(res)
```

### 402. Remove K Digits

```python
class Solution:
    def removeKdigits(self, num: str, k: int) -> str:
        stack = []
        remain = len(num) - k
        for digit in num:
            while k and stack and stack[-1] > digit:
                stack.pop()
                k -= 1
            stack.append(digit)
        return ''.join(stack[:remain]).lstrip('0') or '0'
```

### 321. Create Maximum Number

```python
class Solution:
    def maxNumber(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
        def divide(nums, k):
            stack = []
            drop = len(nums) - k
            for num in nums:
                while drop and stack and stack[-1] < num:
                    stack.pop()
                    drop -= 1
                stack.append(num)
            return stack[:k]

        def merge(A, B):
            ans = []
            while A or B:
                bigger = A if A > B else B
                ans.append(bigger.pop(0))
            return ans
        
        res = [0] * k
        for i in range(k + 1):
            if i <= len(nums1) and k-i <= len(nums2):
                res = max(res, merge(divide(nums1, i), divide(nums2, k-i)))
        return res
```

### 861. Score After Flipping Matrix

```python
class Solution:
    def matrixScore(self, grid: List[List[int]]) -> int:
        def flipRow(r):
            for c in range(C):
                grid[r][c] = 1 - grid[r][c]

        def flipCol(c):
            for r in range(R):
                grid[r][c] = 1 - grid[r][c]

        def checkBinary(nums):
            return int(''.join([str(n) for n in nums]), 2)

        R, C = len(grid), len(grid[0])
        for r in range(R):
            if grid[r][0] == 0:
                flipRow(r)
        for c in range(1, C):
            if sum(grid[r][c] for r in range(R)) * 2 < R:
                flipCol(c)
        return sum(checkBinary(row) for row in grid)
```

### 2126. Destroying Asteroids

```python
class Solution:
    def asteroidsDestroyed(self, mass: int, asteroids: List[int]) -> bool:
        asteroids.sort()
        for a in asteroids:
            if mass < a:
                return False
            mass += a 
        return True
```

### 984. String Without AAA or BBB

```python
class Solution:
    def strWithout3a3b(self, a: int, b: int) -> str:
        res = ''
        while a or b:
            if len(res) >= 2 and res[-1] == res[-2]:
                writeA = res[-1] == 'b'
            else:
                writeA = a >= b
            if writeA:
                a -= 1
                res += 'a'
            else:
                b -= 1
                res += 'b'
        return res
```

### 991. Broken Calculator

```python
class Solution:
    def brokenCalc(self, x: int, y: int) -> int:
        res = 0
        while x < y:
            if y % 2 == 1:
                y += 1
            else:
                y //= 2
            res += 1
        return res + (x - y)
```

### 987. Vertical Order Traversal of a Binary Tree

```python
class Solution:
    def verticalTraversal(self, root: Optional[TreeNode]) -> List[List[int]]:
        d = defaultdict(list)
        q = deque([(root, 0, 0)])
        while q:
            node, row, val = q.popleft()
            d[val].append((row, node.val))
            if node.left:
                q.append((node.left, row + 1, val - 1))
            if node.right:
                q.append((node.right, row + 1, val + 1))
        res = []
        for k in sorted(d.keys()):
            res.append([item[1] for item in sorted(d[k])])
        return res
```

### 467. Unique Substrings in Wraparound String

```python
class Solution:
    def findSubstringInWraproundString(self, s: str) -> int:
        i, res = 0, 0
        d = Counter()
        while i < len(s):
            start = i 
            d[ord(s[start])] = max(d[ord(s[start])], 1)
            j = start + 1
            while j < len(s) and (ord(s[j]) - ord(s[j - 1]) == 1 or ord(s[j]) - ord(s[j - 1]) == -25):
                j += 1
                d[ord(s[j - 1])] = max(d[ord(s[j - 1])], j - start)
            i = j 
        return sum(d.values())
```

### 1665. Minimum Initial Energy to Finish Tasks

```python
'''math + greedy'''
class Solution:
    def minimumEffort(self, tasks: List[List[int]]) -> int:
        # [3, 7], [5, 8]
        # 1: [a1, m1], 2: [a2, m2]
        # T12 = max(m1, a1 + m2), T21 = max(m2, a2 + m1)
        # if m1 - a1 > m2 - a2 => a1 + m2 < a2 + m1
        # if m1 > m2:
        #           T21 = a2 + m1 > a1 + m2
        #                 a2 + m1 > m1
        #           => T21 > T12
        # if m1 < m2:
        #           T12 = a1 + m2 < a2 + m1
        #                 a1 + m2 > m2
        #           => T12 < T21
        tasks.sort(key = lambda x: x[0] - x[1])
        res, left = 0, 0
        for a, m in tasks:
            if m > left:
                res += m - left 
                left = m 
            left -= a 
        return res
```

```python
'''binary search + greedy'''
class Solution:
    def minimumEffort(self, tasks: List[List[int]]) -> int:
        # [3, 7], [5, 8]
        # 1: [a1, m1], 2: [a2, m2]
        # T12 = max(m1, a1 + m2), T21 = max(m2, a2 + m1)
        # if m1 - a1 > m2 - a2 => a1 + m2 < a2 + m1
        # if m1 > m2:
        #           T21 = a2 + m1 > a1 + m2
        #                 a2 + m1 > m1
        #           => T21 > T12
        # if m1 < m2:
        #           T12 = a1 + m2 < a2 + m1
        #                 a1 + m2 > m2
        #           => T12 < T21
        def check(threshold):
            for a, m in tasks:
                if threshold >= m:
                    threshold -= a 
                else:
                    return False
            return True
        tasks.sort(key = lambda x: x[0] - x[1])
        l, r, res = 0, 10 ** 9, 0
        while l <= r:
            m = l + (r - l) // 2
            if check(m):
                res = m 
                r = m - 1
            else:
                l = m + 1
        return res
```

### 1630. Arithmetic Subarrays

```python
class Solution:
    def checkArithmeticSubarrays(self, nums: List[int], l: List[int], r: List[int]) -> List[bool]:
        def check(l, r):
            res = sorted(nums[l: r + 1])
            res = [res[i] - res[i - 1] for i in range(1, len(res))]
            return len(set(res)) == 1
        
        res = []
        for a, b in zip(l, r):
            if check(a, b):
                res.append(True)
            else:
                res.append(False)
        return res
```

### 1167. Minimum Cost to Connect Sticks

```python
class Solution:
    def connectSticks(self, sticks: List[int]) -> int:
        heapify(sticks)
        res = 0
        while len(sticks) > 1:
            a, b = heappop(sticks), heappop(sticks)
            res += a + b
            heappush(sticks, (a + b))
        return res
```

### 969. Pancake Sorting

```python
class Solution:
    def pancakeSort(self, arr: List[int]) -> List[int]:
        n = len(arr)
        res = []
        for i in range(n):
            mx = max(arr[: n - i])
            idx = arr.index(mx)
            arr = arr[:idx + 1][::-1] + arr[idx + 1: n - i]
            arr = arr[::-1]
            res.extend([idx + 1, n - i])
        return res
```

### 870. Advantage Shuffle

```python
class Solution:
    def advantageCount(self, nums1: List[int], nums2: List[int]) -> List[int]:
        nums1.sort()
        nums2 = [(n, i) for i, n in enumerate(nums2)]
        nums2.sort()
        i, j = 0, 0
        n = len(nums1)
        res = [-1] * n 
        visited = [False] * n 
        while i < n and j < n:
            if nums1[i] > nums2[j][0]:
                res[nums2[j][1]] = nums1[i]
                visited[i] = True
                i += 1
                j += 1
            else:
                i += 1
        k = 0
        ans = []
        for i, n in enumerate(visited):
            if n == False:
                ans.append(i)
        for i, n in enumerate(res):
            if n == -1:
                res[i] = nums1[ans[k]]
                k += 1
        return res
```

### 2139. Minimum Moves to Reach Target Score

```python
class Solution:
    def minMoves(self, target: int, maxDoubles: int) -> int:
        res = 0
        while target != 1:
            if maxDoubles > 0:
                if target % 2 == 0:
                    target //= 2
                    maxDoubles -= 1
                else:
                    target -= 1
                res += 1
            else:
                res += target - 1
                target = 1
        return res
```

### 2216. Minimum Deletions to Make Array Beautiful

```python
class Solution:
    def minDeletion(self, nums: List[int]) -> int:
        # [1,1,2,2,3,3]
        # [1,1,1,2,3,3]
        # [1,2,3,3]
        n = len(nums)
        pairs = 0
        i = 0
        while i < n:
            if i + 1 < n and nums[i + 1] != nums[i]:
                pairs += 1
                i += 1
            i += 1
        return n - pairs * 2
```

## greedy to mid point

### 462. Minimum Moves to Equal Array Elements II

- O(n ^ 2)

```python
class Solution:
    def minMoves2(self, nums: List[int]) -> int:
        res = inf
        n = len(nums)
        for i in range(n):
            ans = 0
            for j in range(n):
                if i != j:
                    ans += abs(nums[i] - nums[j])
            res = min(res, ans)
        return res if n > 1 else 0
```

- O(n)

```python
class Solution:
    def minMoves2(self, nums: List[int]) -> int:
        # [1, 2, 3]
        # [1, 2, 9, 10]
        n = len(nums)
        nums.sort()
        if n % 2 == 1:
            mid = n // 2
            return sum(abs(v - nums[mid]) for v in nums)
        else:
            mid1, mid2 = n // 2 - 1, n // 2
            res1 = sum(abs(v - nums[mid1]) for v in nums)
            res2 = sum(abs(v - nums[mid2]) for v in nums)
            return min(res1, res2)
```

- optimized code: only need to use mid

```python
class Solution:
    def minMoves2(self, nums: List[int]) -> int:
        # [1, 2, 3]
        # [1, 2, 6, 10]
        # (b - a) + (c - b) + (d - b) = b - a + c - b + d - b = c - a + d - b
        # (c - a) + (c - b) + (d - c) = c - a + c - b + d - c = c - a + d - b
        n = len(nums)
        nums.sort()
        mid = n // 2
        return sum(abs(v - nums[mid]) for v in nums)
```

### 453. Minimum Moves to Equal Array Elements

```python
class Solution:
    def minMoves(self, nums: List[int]) -> int:
        mn = min(nums)
        res = 0
        for n in nums:
            res += n - mn
        return res

        # k = ((min(nums) + k) * n - sum) / (n - 1)
        # k(n - 1) = mn * n +  - sum =  - k => k = sum - mn * n 
```

### 2033. Minimum Operations to Make a Uni-Value Grid

```python
class Solution:
    def minOperations(self, grid: List[List[int]], x: int) -> int:
        nums = []
        for item in grid:
            nums.extend(item)

        nums.sort()
        mid = nums[len(nums) // 2]
        res = 0
        for n in nums:
            if abs(mid - n) % x:
                return -1
            res += abs(mid - n) // x
        return res 
```

### 2448. Minimum Cost to Make Array Equal

```python
class Solution:
    def minCost(self, nums: List[int], cost: List[int]) -> int:
        # nums = [1,3,5,2]
        # cost = [2,3,1,14]
        res = sorted(zip(nums, cost))
        total = sum(cost)
        mid = total // 2 + 1
        count = 0
        for i, (n, c) in enumerate(res):
            count += c 
            if mid <= count:
                idx = i 
                break
        ans = 0
        for i, (n, c) in enumerate(res):
            ans += abs(n - res[idx][0]) * c 
        return ans
```

### 1403. Minimum Subsequence in Non-Increasing Order

```python
class Solution:
    def minSubsequence(self, nums: List[int]) -> List[int]:
        total = sum(nums)
        nums.sort(reverse = True)
        prefix = list(accumulate(nums))
        for i, n in enumerate(prefix):
            if n > total - n:
                return nums[:i + 1]
```

### 3010. Divide an Array Into Subarrays With Minimum Cost I

```python
class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        res = inf 
        n = len(nums)
        for i in range(1, n):
            for j in range(i + 1, n):
                res = min(res, nums[0] + nums[i] + nums[j])
        return res
```

### 1338. Reduce Array Size to The Half

```python
class Solution:
    def minSetSize(self, arr: List[int]) -> int:
        c = Counter(arr)
        n = len(arr)
        res, ans = 0, 0
        for v in sorted(c.values(), reverse = True):
            res += v
            ans += 1
            if res * 2 >= n:
                break 
        return ans
```

### 3075. Maximize Happiness of Selected Children

```python
class Solution:
    def maximumHappinessSum(self, happiness: List[int], k: int) -> int:
        happiness.sort(reverse = True)
        res = 0
        minus = 0
        for i, h in enumerate(happiness):
            res += max(h - minus, 0)
            minus += 1
            if i == k - 1:
                break
        return res
```

### 2554. Maximum Number of Integers to Choose From a Range I

```python
class Solution:
    def maxCount(self, banned: List[int], n: int, maxSum: int) -> int:
        banned = set(banned)
        res, count = 0, 0
        for i in range(1, n + 1):
            if i not in banned and res + i <= maxSum:
                res += i
                count += 1
            elif res + i > maxSum:
                break
        return count
```

### 2126. Destroying Asteroids

```python
class Solution:
    def asteroidsDestroyed(self, mass: int, asteroids: List[int]) -> bool:
        asteroids.sort()
        for a in asteroids:
            if mass < a:
                return False
            mass += a 
        return True
```

### 2587. Rearrange Array to Maximize Prefix Score

```python
class Solution:
    def maxScore(self, nums: List[int]) -> int:
        prefix = list(accumulate(sorted(nums, reverse = True)))
        return sum(n > 0 for n in prefix)
```

### 1846. Maximum Element After Decreasing and Rearranging

```python
class Solution:
    def maximumElementAfterDecrementingAndRearranging(self, arr: List[int]) -> int:
        res = 1
        arr.sort()
        for n in arr:
            if n >= res:
                res += 1
        return res - 1
```

### 976. Largest Perimeter Triangle

```python
class Solution:
    def largestPerimeter(self, nums: List[int]) -> int:
        nums.sort(reverse = True)
        for i in range(len(nums) - 2):
            if nums[i + 2] + nums[i + 1] > nums[i]:
                return sum(nums[i: i + 3])
        return 0
```

### 2028. Find Missing Observations

```python
class Solution:
    def missingRolls(self, rolls: List[int], mean: int, n: int) -> List[int]:
        m = len(rolls)
        total = (m + n) * mean 
        left = total - sum(rolls)
        res = []
        if n * 1 <= left <= n * 6:
            d, mod = divmod(left, n)
            for i in range(n):
                res.append(d)
            if mod:
                for i in range(mod):
                    res[i] += 1
        return res
```

### 2498. Frog Jump II

```python
class Solution:
    def maxJump(self, stones: List[int]) -> int:
        first, last = stones[0], stones[-1]
        middle = stones[1:-1]
        forward = [first] + middle[::2] + [last ]
        backward = [first] + middle[1::2] + [last]
        dist1 = max(forward[i] - forward[i - 1] for i in range(1, len(forward)))
        dist2 = max(backward[i] - backward[i - 1] for i in range(1, len(backward)))
        return max(dist1, dist2)
```

### 517. Super Washing Machines

```python
class Solution:
    def findMinMoves(self, machines: List[int]) -> int:
        total = sum(machines)
        n = len(machines)
        if total % n:
            return -1
        average = total // n 
        res, need = 0, 0
        for m in machines:
            need += m - average
            res = max(res, abs(need), m - average)
        return res 
```

### 2371. Minimize Maximum Value in a Grid

```python
class Solution:
    def minScore(self, grid: List[List[int]]) -> List[List[int]]:
        R, C = len(grid), len(grid[0])
        nums = sorted([(grid[r][c], r, c) for r in range(R) for c in range(C)])
        row, col = [0] * R, [0] * C 
        for num, r, c in nums:
            grid[r][c] = max(row[r], col[c]) + 1
            row[r] = col[c] = grid[r][c]
        return grid
```

- interval

* [452. Minimum Number of Arrows to Burst Balloons](#452-minimum-number-of-arrows-to-burst-balloons)
* [435. Non-overlapping Intervals](#435-non-overlapping-intervals)
* [646. Maximum Length of Pair Chain](#646-maximum-length-of-pair-chain)
* [1272. Remove Interval](#1272-remove-interval)

### 57. Insert Interval

- use count to store one result
- use start, end to record one merged interval

```python
class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        intervals.append(newInterval)
        events = []
        for s, e in intervals:
            events.append((s, -1))
            events.append((e, 1))
        events.sort()

        count, res = 0, []
        start, end = inf, -inf
        for point, sign in events:
            if sign < 0:
                count += 1
                start = min(start, point)
            else:
                count -= 1
                end = max(end, point)
            if count == 0:
                res.append([start, end])
                start, end = inf, -inf
        return res
```

- better O(n) solution

```python
class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        s, e = newInterval 
        res = []
        insert = False 
        for start, end in intervals:
            if e < start:
                if not insert:
                    res.append([s, e])
                    insert = True
                res.append([start, end])
            elif end < s:
                res.append([start, end])
            else:
                s, e = min(s, start), max(e, end)
        if not insert:
            res.append([s, e])
        return res 
```


### 56. Merge Intervals

```python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        n = len(intervals)
        intervals.sort()
        start, end = intervals[0]
        res = []
        for i in range(1, n):
            s, e = intervals[i]
            if s > end:
                res.append([start, end])
                start = s 
            end = max(end, e)
        res.append([start, end])
        return res 
```

### 452. Minimum Number of Arrows to Burst Balloons

- greedy at end
- use prev end

```python
class Solution:
    def findMinArrowShots(self, points: List[List[int]]) -> int:
        points.sort(key = lambda x: x[1])
        prev, count = -inf, 0
        for s, e in points:
            if s > prev:
                count += 1
                prev = e
        return count
```

### 435. Non-overlapping Intervals

- greedy at end
- use prev end

```python
class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        intervals.sort(key = lambda x: x[1])
        prev, count = -inf, 0
        for s, e in intervals:
            if s >= prev:
                count += 1
                prev = e
        return len(intervals) - count
```

### 646. Maximum Length of Pair Chain

```python
class Solution:
    def findLongestChain(self, intervals: List[List[int]]) -> int:
        intervals.sort(key = lambda x: x[1])
        prev, count = -inf, 0
        for s, e in intervals:
            if s > prev:
                count += 1
                prev = e
        return count
```

### 1272. Remove Interval

```python
class Solution:
    def removeInterval(self, intervals: List[List[int]], toBeRemoved: List[int]) -> List[List[int]]:
        start, end = toBeRemoved
        res = []
        for s, e in intervals:
            if s >= end or e <= start:
                res.append([s, e])
            else:
                if s < start:
                    res.append([s, start])
                if e > end:
                    res.append([end, e])
        return res
```

### 665. Non-decreasing Array

```python
class Solution:
    def checkPossibility(self, nums: List[int]) -> bool:
        n = len(nums)
        cnt = 0
        for i in range(1, n):
            if nums[i] < nums[i - 1]:
                cnt += 1
                if i == 1 or nums[i] >= nums[i - 2]:
                    nums[i - 1] = nums[i]
                else:
                    nums[i] = nums[i - 1]
        return cnt <= 1
```

### 1717. Maximum Score From Removing Substrings

```python 
class Solution:
    def maximumGain(self, s: str, x: int, y: int) -> int:
        def remove_pair(pair, val):
            stack = []
            res = 0
            for c in self.s:
                if c == pair[1] and stack and stack[-1] == pair[0]:
                    stack.pop()
                    res += val 
                else:
                    stack.append(c)
            self.s = ''.join(stack)
            return res 
        pair = 'ab' if x > y else 'ba'
        self.s = s 
        res = 0
        res += remove_pair(pair, max(x, y))
        res += remove_pair(pair[::-1], min(x, y))
        return res 
```

### 2193. Minimum Number of Moves to Make Palindrome

```python 
class Solution:
    def minMovesToMakePalindrome(self, s: str) -> int:
        s = list(s)
        res = 0
        while s:
            l = s.index(s[-1])
            if l == len(s) - 1:
                res += l // 2
            else:
                s.pop(l)
                res += l
            s.pop()
        return res 
```