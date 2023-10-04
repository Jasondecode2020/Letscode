307. Range Sum Query - Mutable

```python
class NumArray:

    def __init__(self, nums: List[int]):
        # prepare padding of nums
        self.nums = nums
        self.n = 2 ** math.ceil(math.log2(len(nums))) if len(nums) > 1 else 2 # padding
        self.nums += [0] * (self.n - len(nums))
        # build segment tree
        self.tree = [0] * self.n * 2
        for i in range(self.n):
            self.tree[self.n + i] = self.nums[i] 
        for i in range(self.n - 1, 0, -1):
            self.tree[i] = self.tree[2 * i] + self.tree[2 * i + 1]

    def update(self, index: int, val: int) -> None:
        self.tree[self.n + index] = val
        j = (self.n + index) // 2
        while j > 0:
            self.tree[j] = self.tree[2 * j] + self.tree[2 * j + 1]
            j //= 2

    def sumRange(self, left: int, right: int) -> int:
        def f(node, node_low, node_high, query_low, query_high, tree):
            if node_low >= query_low and node_high <= query_high:
                return tree[node]
            if node_low > query_high or node_high < query_low:
                return 0
            last_in_left = (node_low + node_high) // 2 
            return f(2 * node, node_low, last_in_left, query_low, query_high, tree) + \
                    f(2 * node + 1, last_in_left + 1, node_high, query_low, query_high, tree)
        return f(1, 0, self.n - 1, left, right, self.tree)
```