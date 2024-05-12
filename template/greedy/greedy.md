# greedy

## simulation

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