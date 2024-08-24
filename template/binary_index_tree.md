## template 1: 

```python
class BIT:
    def __init__(self, nums):
        self.t = [0] + nums 
        self.n = len(self.t)
        for i in range(1, self.n):
            p = i + self.lowbit(i)
            if p < self.n:
                self.t[p] += self.t[i] 

    def lowbit(self, i):
        return i & -i 

    def add(self, i, k):
        while i < self.n:
            self.t[i] += k 
            i += self.lowbit(i)

    def update(self, i, k):
        cur_val = self.query(i, i)
        val = k - cur_val
        self.add(i + 1, val)

    def sum(self, i):
        res = 0
        while i > 0:
            res += self.t[i]
            i -= self.lowbit(i) 
        return res 

    def query(self, l, r):
        return self.sum(r + 1) - self.sum(l)
```
```
# for sum:

# sum(7) = self.tree[7] + self.tree[6] + self.tree[4] 
# 4 = 100, flip last bit, becomes 000, jump out of loop

# for add:
# add(4, 10) = self.tree[4] + 10, self.tree[8] + 10, when 16 > length of 15
# jump out of the loop

# for init:
# start with one, accumulate parent: p = i + (i & -i)
# remember: sum: 764, add: 48 16
```

## questions:

### 1 1d Contiguous

* [307. Range Sum Query - Mutable](#307-range-sum-query---mutable) 1800
* [1649. Create Sorted Array through Instructions](#1649-create-sorted-array-through-instructions) 2208
* [3187. Peaks in Array]()

### 2 1d Contiguous with map

* [2179. Count Good Triplets in an Array](#2179-count-good-triplets-in-an-array) 2272
* [2659. Make Array Empty](#2659-make-array-empty) 2282

### 3 1d Discrete

* [3072. Distribute Elements Into Two Arrays II](#3072-distribute-elements-into-two-arrays-ii) 2053

### 4 2d Contiguous

* [308. Range Sum Query 2D - Mutable](#308-range-sum-query-2d---mutable) 1900

### 3


### 307. Range Sum Query - Mutable

```python
class BIT:
    def __init__(self, nums):
        self.t = [0] + nums 
        self.n = len(self.t)
        for i in range(1, self.n):
            p = i + self.lowbit(i)
            if p < self.n:
                self.t[p] += self.t[i] 

    def lowbit(self, i):
        return i & -i

    def add(self, i, k):
        while i < self.n:
            self.t[i] += k
            i += self.lowbit(i)

    def update(self, i, k):
        cur_val = self.query(i, i)
        val = k - cur_val
        self.add(i + 1, val)

    def sum(self, i):
        res = 0
        while i > 0:
            res += self.t[i]
            i -= self.lowbit(i)
        return res

    def query(self, l, r):
        return self.sum(r + 1) - self.sum(l)

class NumArray:

    def __init__(self, nums: List[int]):
        self.t = BIT(nums)

    def update(self, index: int, val: int) -> None:
        self.t.update(index, val)

    def sumRange(self, left: int, right: int) -> int:
        return self.t.query(left, right)
```

### 308. Range Sum Query 2D - Mutable

```python
class BIT:
    def __init__(self, nums):
        self.t = [0] + nums 
        self.n = len(self.t)
        for i in range(1, self.n):
            p = i + self.lowbit(i)
            if p < self.n:
                self.t[p] += self.t[i] 

    def lowbit(self, i):
        return i & -i 

    def add(self, i, k):
        while i < self.n:
            self.t[i] += k 
            i += self.lowbit(i)

    def update(self, i, k):
        cur_val = self.query(i, i)
        val = k - cur_val
        self.add(i + 1, val)

    def sum(self, i):
        res = 0
        while i > 0:
            res += self.t[i]
            i -= self.lowbit(i) 
        return res 

    def query(self, l, r):
        return self.sum(r + 1) - self.sum(l)

class NumMatrix:

    def __init__(self, matrix: List[List[int]]):
        self.t = BIT(sum(matrix, []))
        self.R, self.C = len(matrix), len(matrix[0])

    def getIndex(self, r, c):
        return r * self.C + c 

    def update(self, row: int, col: int, val: int) -> None:
        i = self.getIndex(row, col)
        self.t.update(i, val)

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        res = 0
        for row in range(row1, row2 + 1):
            res += self.t.query(self.getIndex(row, col1), self.getIndex(row, col2))
        return res
```

### 1649. Create Sorted Array through Instructions

- sorted list

```python
from sortedcontainers import SortedList
class Solution:
    def createSortedArray(self, instructions: List[int]) -> int:
        mod = 10 ** 9 + 7
        sl, res = SortedList(), 0
        for i in instructions:
            n = len(sl)
            l, r = sl.bisect_left(i), n - sl.bisect_right(i)
            res += min(l, r)
            sl.add(i)
        return res % mod
```

```python
class BIT:
    __slot__ = 'tree'
    
    def __init__(self, n):
        self.tree = [0] * n 
        self.n = len(self.tree)

    def lowbit(self, i):
        return i & -i 

    def add(self, i, k):
        while i < self.n:
            self.tree[i] += k
            i += self.lowbit(i)

    def sum(self, i):
        res = 0
        while i > 0:
            res += self.tree[i]
            i -= self.lowbit(i)
        return res

class Solution:
    def createSortedArray(self, instructions: List[int]) -> int:
        n = 10 ** 5
        t = BIT(n + 1)
        res, mod = 0, 10 ** 9 + 7
        for i in instructions:
            l, r = t.sum(i - 1), t.sum(n) - t.sum(i)
            t.add(i, 1)
            res += min(l, r)
        return res % mod
```

### 3187. Peaks in Array

```python
class BIT:
    def __init__(self, nums):
        self.t = [0] + nums 
        self.n = len(self.t)

    def lowbit(self, i):
        return i & -i

    def add(self, i, k):
        while i < self.n:
            self.t[i] += k
            i += self.lowbit(i)

    def update(self, i, k):
        cur_val = self.query(i, i)
        val = k - cur_val
        self.add(i + 1, val)

    def sum(self, i):
        res = 0
        while i > 0:
            res += self.t[i]
            i -= self.lowbit(i)
        return res

    def query(self, l, r):
        if r < l:
            return 0
        return self.sum(r) - self.sum(l - 1)

class Solution:
    def countOfPeaks(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        bit = BIT([0] * (n - 2))
        def add(i, val):
            if nums[i - 1] < nums[i] and nums[i] > nums[i + 1]:
                bit.add(i, val)
        for i in range(1, n - 1):
            add(i, 1)
        res = []
        for sign, a, b in queries:
            if sign == 2:
                for i in range(max(a - 1, 1), min(a + 2, n - 1)):
                    add(i, -1)
                nums[a] = b 
                for i in range(max(a - 1, 1), min(a + 2, n - 1)):
                    add(i, 1)
            else:
                val = bit.query(a + 1, b - 1)
                res.append(val)
        return res 
```

### 3072. Distribute Elements Into Two Arrays II

- sorted list

```python
from sortedcontainers import SortedList
class Solution:
    def resultArray(self, nums: List[int]) -> List[int]:
        # [1, 2, 3, 4]
        def greaterCount(arr, val):
            n = len(arr)
            i = arr.bisect_left(val + 1)
            return n - i
                
        sl1, sl2 = SortedList([nums[0]]), SortedList([nums[1]])
        arr1, arr2 = [nums[0]], [nums[1]]
        for n in nums[2:]:
            if greaterCount(sl1, n) > greaterCount(sl2, n) or (greaterCount(sl1, n) == greaterCount(sl2, n) and (len(sl1) <= len(sl2))):
                arr1.append(n)
                sl1.add(n)
            else:
                arr2.append(n)
                sl2.add(n)
        return arr1 + arr2
```

```python
class BIT:
    __slot__ = 't'

    def __init__(self, n): # init BIT
        self.t = [0] * n
        self.n = len(self.t)

    def lowbit(self, i):
        return i & -i

    def add(self, i, k): # add k to index i
        while i < self.n:
            self.t[i] += k
            i += self.lowbit(i) # add last set bit

    def sum(self, i): # sum from index 1 to i
        res = 0
        while i > 0:
            res += self.t[i]
            i -= (i & -i) # minus the last set bit
        return res

class Solution:
    def resultArray(self, nums: List[int]) -> List[int]:
        arr1, arr2 = [nums[0]], [nums[1]]
        sorted_arr = sorted(set(nums))
        n = len(sorted_arr)
        t1, t2 = BIT(n + 1), BIT(n + 1)
        t1.add(bisect_left(sorted_arr, nums[0]) + 1, 1)
        t2.add(bisect_left(sorted_arr, nums[1]) + 1, 1)
        for v in nums[2:]:
            idx = bisect_left(sorted_arr, v) + 1
            gc1 = len(arr1) - t1.sum(idx)
            gc2 = len(arr2) - t2.sum(idx)
            if gc1 > gc2 or gc1 == gc2 and len(arr1) <= len(arr2):
                arr1.append(v)
                t1.add(idx, 1)
            else:
                arr2.append(v)
                t2.add(idx, 1)
        return arr1 + arr2 
```

### 2179. Count Good Triplets in an Array

```python
from sortedcontainers import SortedList
class Solution:
    def goodTriplets(self, nums1: List[int], nums2: List[int]) -> int:
        d = {v: i for i, v in enumerate(nums1)}
        sl = SortedList()
        n = len(nums1)
        res = 0
        for i in range(1, n - 1):
            sl.add(d[nums2[i - 1]])
            x = d[nums2[i]]
            less = sl.bisect_left(x)
            res += less * (n - 1 - x - (i - less))
        return res 
```

```python
class BIT:
    def __init__(self, nums): # init BIT
        self.tree = [0] + nums

    def add(self, i, k): 
        while i < len(self.tree):
            self.tree[i] += k
            i += (i & -i) 

    def sum(self, i): 
        res = 0
        while i > 0:
            res += self.tree[i]
            i -= (i & -i)
        return res

class Solution:
    def goodTriplets(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        bit1, bit2 = BIT([0] * n), BIT([0] * n)
        d = {v: i for i, v in enumerate(nums1)}
        prefix, suffix = [0] * n, [0] * n
        for i in range(1, n - 1):
            bit1.add(d[nums2[i - 1]] + 1, 1)
            s = bit1.sum(d[nums2[i]])
            prefix[i] = s
        for i in range(n - 2, 0, -1):
            bit2.add(d[nums2[i + 1]] + 1, 1)
            s = bit2.sum(n) - bit2.sum(d[nums2[i]] + 1)
            suffix[i] = s
        res = 0
        for a, b in zip(prefix, suffix):
            res += a * b
        return res
```

### 2659. Make Array Empty

```python
class BIT:
    def __init__(self, nums): 
        self.tree = [0] + nums

    def add(self, i, k): 
        while i < len(self.tree):
            self.tree[i] += k
            i += (i & -i) 

    def sum(self, i): 
        res = 0
        while i > 0:
            res += self.tree[i]
            i -= (i & -i) 
        return res

    def query(self, l, r):
        return self.sum(r) - self.sum(l - 1)

class Solution:
    def countOperationsToEmptyArray(self, nums: List[int]) -> int:
        n = len(nums)
        bit = BIT([0] * n)
        idx_array = sorted([(v, i + 1) for i, v in enumerate(nums)])
        last_pos = 1
        res = n
        for v, pos in idx_array:
            if pos >= last_pos:
                res += pos - last_pos - bit.query(last_pos, pos)
            else:
                res += (n - last_pos - bit.query(last_pos, n)) + (pos - bit.query(1, pos))
            bit.add(pos, 1)
            last_pos = pos
        return res
```