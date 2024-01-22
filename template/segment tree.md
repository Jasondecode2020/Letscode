## template 1:

```python
class SegmentTree: 
    def __init__(self, nums: List[int]):
        self.n = len(nums)
        self.seg = [0] * (self.n * 4) # 4 * n to keep array big enough
        # self.seg = [0] * (2 << self.n.bit_length()) # reduce space and time
        self.build(nums, 0, 0, self.n - 1) # index from 0 to n - 1, root: 0

    def build(self, nums, o, l, r): # o start from 0, root node, l, r inclusive
        if l == r:
            self.seg[o] = nums[l]
            return
        m = l + (r - l) // 2
        self.build(nums, o * 2 + 1, l, m) # postorder traversal
        self.build(nums, o * 2 + 2, m + 1, r)
        self.seg[o] = self.seg[o * 2 + 1] + self.seg[o * 2 + 2]

    def update(self, index, val, o, l, r): # at position index update to val
        if l == r:
            self.seg[o] = val
            return
        m = l + (r - l) // 2
        if index <= m:
            self.update(index, val, o * 2 + 1, l, m)
        else:
            self.update(index, val, o * 2 + 2, m + 1, r)
        self.seg[o] = self.seg[o * 2 + 1] + self.seg[o * 2 + 2]

    def query(self, L, R, o, l, r): # query from L to R inclusive
        if L == l and R == r:
            return self.seg[o]
        m = l + (r - l) // 2
        if R <= m:
            return self.query(L, R, o * 2 + 1, l, m)
        if L > m:
            return self.query(L, R, o * 2 + 2, m + 1, r)
        return self.query(L, m, o * 2 + 1, l, m) + self.query(m + 1, R, o * 2 + 2, m + 1, r)
```

### 307. Range Sum Query - Mutable

```python
class SegmentTree: 
    def __init__(self, nums: List[int]):
        self.n = len(nums)
        self.seg = [0] * (self.n * 4) # 4 * n to keep array big enough
        self.build(nums, 0, 0, self.n - 1) # index from 0 to n - 1, root: 0

    def build(self, nums, o, l, r): # o start from 0, root node, l, r inclusive
        if l == r:
            self.seg[o] = nums[l]
            return
        m = l + (r - l) // 2
        self.build(nums, o * 2 + 1, l, m)
        self.build(nums, o * 2 + 2, m + 1, r)
        self.seg[o] = self.seg[o * 2 + 1] + self.seg[o * 2 + 2]

    def update(self, index, val, o, l, r): # index update to val
        if l == r:
            self.seg[o] = val
            return
        m = l + (r - l) // 2
        if index <= m:
            self.update(index, val, o * 2 + 1, l, m)
        else:
            self.update(index, val, o * 2 + 2, m + 1, r)
        self.seg[o] = self.seg[o * 2 + 1] + self.seg[o * 2 + 2]

    def query(self, L, R, o, l, r): # query from L to R inclusive
        if L == l and R == r:
            return self.seg[o]
        m = l + (r - l) // 2
        if R <= m:
            return self.query(L, R, o * 2 + 1, l, m)
        if L > m:
            return self.query(L, R, o * 2 + 2, m + 1, r)
        return self.query(L, m, o * 2 + 1, l, m) + self.query(m + 1, R, o * 2 + 2, m + 1, r)

class NumArray:
    def __init__(self, nums: List[int]):
        self.st = SegmentTree(nums)

    def update(self, index: int, val: int) -> None:
        self.st.update(index, val, 0, 0, self.st.n - 1)

    def sumRange(self, left: int, right: int) -> int:
        return self.st.query(left, right, 0, 0, self.st.n - 1)
```

### 308. Range Sum Query 2D - Mutable

```python
class SegmentTree: 
    def __init__(self, nums: List[int]):
        self.n = len(nums)
        self.seg = [0] * (self.n * 4) # 4 * n to keep array big enough
        # self.seg = [0] * (2 << self.n.bit_length()) # reduce space and time
        self.build(nums, 0, 0, self.n - 1) # index from 0 to n - 1, root: 0

    def build(self, nums, o, l, r): # o start from 0, root node, l, r inclusive
        if l == r:
            self.seg[o] = nums[l]
            return
        m = l + (r - l) // 2
        self.build(nums, o * 2 + 1, l, m) # postorder traversal
        self.build(nums, o * 2 + 2, m + 1, r)
        self.seg[o] = self.seg[o * 2 + 1] + self.seg[o * 2 + 2]

    def update(self, index, val, o, l, r): # at position index update to val
        if l == r:
            self.seg[o] = val
            return
        m = l + (r - l) // 2
        if index <= m:
            self.update(index, val, o * 2 + 1, l, m)
        else:
            self.update(index, val, o * 2 + 2, m + 1, r)
        self.seg[o] = self.seg[o * 2 + 1] + self.seg[o * 2 + 2]

    def query(self, L, R, o, l, r): # query from L to R inclusive
        if L == l and R == r:
            return self.seg[o]
        m = l + (r - l) // 2
        if R <= m:
            return self.query(L, R, o * 2 + 1, l, m)
        if L > m:
            return self.query(L, R, o * 2 + 2, m + 1, r)
        return self.query(L, m, o * 2 + 1, l, m) + self.query(m + 1, R, o * 2 + 2, m + 1, r)

class NumMatrix:

    def __init__(self, matrix: List[List[int]]):
        self.st = SegmentTree(sum(matrix, []))
        self.R, self.C = len(matrix), len(matrix[0])

    def update(self, row: int, col: int, val: int) -> None:
        index = row * self.C + col
        self.st.update(index, val, 0, 0, self.st.n - 1)

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        res = 0
        for r in range(row1, row2 + 1):
            left, right = r * self.C + col1, r * self.C + col2
            res += self.st.query(left, right, 0, 0, self.st.n - 1)
        return res
```