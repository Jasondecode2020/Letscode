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

- 307. Range Sum Query - Mutable
- 308. Range Sum Query 2D - Mutable
- 2179. Count Good Triplets in an Array
- 2659. Make Array Empty

### 307. Range Sum Query - Mutable

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

class NumArray:

    def __init__(self, nums: List[int]):
        self.tree = BIT(nums)

    def update(self, index: int, val: int) -> None:
        cur_val = self.sumRange(index, index)
        diff = val - cur_val
        self.tree.add(index + 1, diff)

    def sumRange(self, left: int, right: int) -> int:
        return self.tree.sum(right + 1) - self.tree.sum(left)
```

### 308. Range Sum Query 2D - Mutable

```python
class BIT:
    def __init__(self, nums): # init BIT
        self.tree = [0] + nums
        for i in range(1, len(self.tree)):
            p = i + (i & -i)
            if p < len(self.tree):
                self.tree[p] += self.tree[i]

    def add(self, i, k): # add k to index i
        while i < len(self.tree):
            self.tree[i] += k
            i += (i & -i)

    def sum(self, i): # sum from index 1 to i
        res = 0
        while i > 0:
            res += self.tree[i]
            i -= (i & -i)
        return res

class NumMatrix:

    def __init__(self, matrix: List[List[int]]):
        self.tree = BIT(sum(matrix, []))
        self.R, self.C = len(matrix), len(matrix[0])

    def update(self, row: int, col: int, val: int) -> None:
        index = row * self.C + col
        cur_val = self.sumRegion(row, col, row, col)
        diff = val - cur_val
        self.tree.add(index + 1, diff)

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        res = 0
        for r in range(row1, row2 + 1):
            left, right = r * self.C + col1, r * self.C + col2
            res += (self.tree.sum(right + 1) - self.tree.sum(left))
        return res

# Your NumMatrix object will be instantiated and called as such:
# obj = NumMatrix(matrix)
# obj.update(row,col,val)
# param_2 = obj.sumRegion(row1,col1,row2,col2)
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