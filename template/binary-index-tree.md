## template 1: 

```python
class BIT:
    def __init__(self, nums): # init BIT
        self.tree = [0] + nums
        for i in range(1, len(self.tree)):
            p = i + (i & -i) # index to parent
            if p < len(self.tree):
                self.tree[p] += self.tree[i]

    def add(self, i, k): # add k to index i
        while i < len(self.tree):
            self.tree[i] += k
            i += (i & -i) # add last set bit

    def update(self, i, n):
        k = n - self.query(i, i)
        while i < len(self.tree):
            self.tree[i] += k
            i += (i & -i) # add last set bit

    def sum(self, i): # sum from index 1 to i
        res = 0
        while i > 0:
            res += self.tree[i]
            i -= (i & -i) # minus the last set bit
        return res

    def query(self, l, r): # sum from l to r
        return self.sum(r) - self.sum(l - 1)

# for sum:

# sum(7) = self.tree[7] + self.tree[6] + self.tree[4] 
# 4 = 100, flip last bit, becomes 000, jump out of loop

# for add:
# add(4, 10) = self.tree[4] + 10, self.tree[8] + 10, when 16 > length of 15
# jump out of the loop

# for init:
# start with one, accumulate parent: p = i + (i & -i)
```

## questions:

### 1 Contigous
* [307. Range Sum Query - Mutable](#307-range-sum-query---mutable)
* [308. Range Sum Query 2D - Mutable](#308-range-sum-query-2d---mutable)

### 2 Discrete
* [3072. Distribute Elements Into Two Arrays II](#3072-distribute-elements-into-two-arrays-ii)
* [2179. Count Good Triplets in an Array](#2179-count-good-triplets-in-an-array)
* [2659. Make Array Empty](#2659-make-array-empty)

### 307. Range Sum Query - Mutable

```python
class BIT:
    def __init__(self, nums):
        self.tree = [0] + nums
        self.n = len(self.tree)
        for i in range(1, self.n):
            p = i + self.lowbit(i)
            if p < self.n:
                self.tree[p] += self.tree[i]

    def lowbit(self, i):
        return i & -i

    def add(self, i, k): 
        while i < self.n:
            self.tree[i] += k
            i += self.lowbit(i)

    def update(self, i, k): 
        cur_val = self.query(i, i)
        val = k - cur_val
        self.add(i + 1, val)

    def sum(self, i):
        res = 0
        while i > 0:
            res += self.tree[i]
            i -= self.lowbit(i)
        return res

    def query(self, l, r):
        return self.sum(r + 1) - self.sum(l)

class NumArray:

    def __init__(self, nums: List[int]):
        self.tree = BIT(nums)

    def update(self, index: int, val: int) -> None:
        self.tree.update(index, val)

    def sumRange(self, left: int, right: int) -> int:
        return self.tree.query(left, right)


# Your NumArray object will be instantiated and called as such:
# obj = NumArray(nums)
# obj.update(index,val)
# param_2 = obj.sumRange(left,right)
```

### 308. Range Sum Query 2D - Mutable

```python
class BIT:
    def __init__(self, nums):
        self.tree = [0] + nums
        self.n = len(self.tree)
        for i in range(1, self.n):
            p = i + self.lowbit(i)
            if p < self.n:
                self.tree[p] += self.tree[i]

    def lowbit(self, i):
        return i & -i

    def add(self, i, k): 
        while i < self.n:
            self.tree[i] += k
            i += self.lowbit(i)

    def update(self, i, k): 
        cur_val = self.query(i, i)
        val = k - cur_val
        self.add(i + 1, val)

    def sum(self, i):
        res = 0
        while i > 0:
            res += self.tree[i]
            i -= self.lowbit(i)
        return res

    def query(self, l, r):
        return self.sum(r + 1) - self.sum(l)

class NumMatrix:

    def __init__(self, matrix: List[List[int]]):
        self.tree = BIT(sum(matrix, []))
        self.R, self.C = len(matrix), len(matrix[0])

    def update(self, row: int, col: int, val: int) -> None:
        i = row * self.C + col 
        self.tree.update(i, val)

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        res = 0
        for row in range(row1, row2 + 1):
            res += self.tree.query(row * self.C + col1, row * self.C + col2)
        return res 


# Your NumMatrix object will be instantiated and called as such:
# obj = NumMatrix(matrix)
# obj.update(row,col,val)
# param_2 = obj.sumRegion(row1,col1,row2,col2)
```

### 3072. Distribute Elements Into Two Arrays II

```python
from sortedcontainers import SortedList
class Solution:
    def resultArray(self, nums: List[int]) -> List[int]:
        def greaterCount(arr, val):
            n = len(arr)
            i = arr.bisect_left(val + 1)
            return n - i
                
        sl1, sl2 = SortedList([nums[0]]), SortedList([nums[1]])
        arr1, arr2 = [nums[0]], [nums[1]]
        for n in nums[2:]:
            if greaterCount(sl1, n) > greaterCount(sl2, n):
                arr1.append(n)
                sl1.add(n)
            elif greaterCount(sl1, n) < greaterCount(sl2, n):
                arr2.append(n)
                sl2.add(n)
            elif greaterCount(sl1, n) == greaterCount(sl2, n):
                if len(arr1) < len(arr2):
                    arr1.append(n)
                    sl1.add(n)
                elif len(arr1) > len(arr2):
                    arr2.append(n)
                    sl2.add(n)
                else:
                    arr1.append(n)
                    sl1.add(n)
        return arr1 + arr2
```

### 2179. Count Good Triplets in an Array

```python
class BIT:
    def __init__(self, nums): # init BIT
        self.tree = [0] + nums
        for i in range(1, len(self.tree)):
            p = i + (i & -i) # index to parent
            if p < len(self.tree):
                self.tree[p] += self.tree[i]

    def add(self, i, k): # add k to index i
        while i < len(self.tree):
            self.tree[i] += k
            i += (i & -i) # add last set bit

    def sum(self, i): # sum from index 1 to i
        res = 0
        while i > 0:
            res += self.tree[i]
            i -= (i & -i) # minus the last set bit
        return res

class Solution:
    def goodTriplets(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        bit1, bit2 = BIT([0] * n), BIT([0] * n)
        d = {v: i for i, v in enumerate(nums1)} # map
        prefix, suffix = [0] * n, [0] * n
        for i in range(1, n - 1):
            bit1.add(d[nums2[i - 1]] + 1, 1)
            s = bit1.sum(d[nums2[i]] + 1)
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
    def __init__(self, nums): # init BIT
        self.tree = [0] + nums
        for i in range(1, len(self.tree)):
            p = i + (i & -i) # index to parent
            if p < len(self.tree):
                self.tree[p] += self.tree[i]

    def add(self, i, k): # add k to index i
        while i < len(self.tree):
            self.tree[i] += k
            i += (i & -i) # add last set bit

    def sum(self, i): # sum from index 1 to i
        res = 0
        while i > 0:
            res += self.tree[i]
            i -= (i & -i) # minus the last set bit
        return res

    def query(self, l, r):
        return self.sum(r) - self.sum(l - 1)
# https://www.youtube.com/watch?v=13RA9fH-Dq4
class Solution:
    def countOperationsToEmptyArray(self, nums: List[int]) -> int:
        n = len(nums)
        bit = BIT([1] * n)
        idx_array = sorted([(v, i + 1) for i, v in enumerate(nums)])
        last_pos = -1
        res = n
        for v, pos in idx_array:
            if last_pos == -1:
                res += bit.query(1, pos - 1)
            elif pos > last_pos:
                res += bit.query(last_pos + 1, pos - 1)
            elif pos < last_pos:
                res += bit.query(1, pos - 1) + bit.query(last_pos + 1, n)
            bit.add(pos, -1)
            last_pos = pos
        return res
```