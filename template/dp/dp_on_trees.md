## tree dp: 7 problems

## template:

- 1245. Tree Diameter (dfs with parent)
- 2246. Longest Path With Different Adjacent Characters (dfs no parent)

### 1 Tree Diameter Questions

- 543. Diameter of Binary Tree
- 124. Binary Tree Maximum Path Sum
- 1245. Tree Diameter (dfs with parent)
- 2246. Longest Path With Different Adjacent Characters (dfs no parent)
- 1522. Diameter of N-Ary Tree

### 2 Tree combination problems

- 894. All Possible Full Binary Trees
- 1339. Maximum Product of Splitted Binary Tree

### 3 Tree maximum independent set

- 337. House Robber III

### 4 reroot dp

### 543. Diameter of Binary Tree

```python
class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        self.res = 0
        def dfs(node):
            if not node:
                return 0
            l, r = dfs(node.left), dfs(node.right)
            self.res = max(self.res, l + r)
            return max(l, r) + 1
        dfs(root)
        return self.res
```


### 124. Binary Tree Maximum Path Sum

```python
class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        self.res = root.val
        def dfs(root):
            if not root:
                return 0
            leftMax = max(dfs(root.left), 0)
            rightMax = max(dfs(root.right), 0)
            self.res = max(self.res, root.val + leftMax + rightMax)
            return root.val + max(leftMax, rightMax)
        dfs(root)
        return self.res
```

### 1245. Tree Diameter

```python
class Solution:
    def treeDiameter(self, edges: List[List[int]]) -> int:
        g = defaultdict(list)
        for u, v in edges:
            g[u].append(v)
            g[v].append(u)

        self.res = 0
        def dfs(x, parent):
            x_len = 0
            for y in g[x]:
                if y == parent:
                    continue
                y_len = dfs(y, x) + 1
                self.res = max(self.res, x_len + y_len)
                x_len = max(x_len, y_len)
            return x_len
        dfs(0, -1)
        return self.res
```

### 2246. Longest Path With Different Adjacent Characters

```python
class Solution:
    def longestPath(self, parent: List[int], s: str) -> int:
        g = defaultdict(list)
        n = len(parent)
        for i in range(1, n):
            g[parent[i]].append(i)

        self.res = 0
        def dfs(x):
            x_len = 0
            for y in g[x]:
                y_len = dfs(y) + 1
                if s[y] != s[x]:
                    self.res = max(self.res, x_len + y_len)
                    x_len = max(x_len, y_len)
            return x_len
        dfs(0)
        return self.res + 1
```

### 1522. Diameter of N-Ary Tree

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children if children is not None else []
"""

class Solution:
    def diameter(self, root: 'Node') -> int:
        """
        :type root: 'Node'
        :rtype: int
        """
        self.res = 0
        def dfs(node):
            if not node:
                return 0
            x_len = 0
            for child in node.children:
                y_len = dfs(child) + 1
                self.res = max(self.res, x_len + y_len)
                x_len = max(x_len, y_len)
            return x_len
        dfs(root)
        return self.res
```

### 894. All Possible Full Binary Trees

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    @cache
    def allPossibleFBT(self, n: int) -> List[Optional[TreeNode]]:
        if n == 1:
            return [TreeNode()]
        res = []
        for i in range(1, n, 2):
            for l in self.allPossibleFBT(i):
                for r in self.allPossibleFBT(n - i - 1):
                    root = TreeNode(0, l, r)
                    res.append(root)
        return res
```

### 1339. Maximum Product of Splitted Binary Tree

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxProduct(self, root: Optional[TreeNode]) -> int:
        def postorderSum(node):
            if node:
                postorderSum(node.left)
                postorderSum(node.right)
                self.total += node.val
        self.total = 0
        postorderSum(root)

        self.res, mod = 0, 10 ** 9 + 7
        def postorderMax(node):
            if not node:
                return 0
            l, r = postorderMax(node.left), postorderMax(node.right)
            curSum = node.val + l + r
            self.res = max(self.res, curSum * (self.total - curSum))
            return curSum
        postorderMax(root)
        return self.res % mod
```

### 337. House Robber III

```python
class Solution:
    def rob(self, root: Optional[TreeNode]) -> int:
        def dfs(root): #[robRoot, notRobRoot]
            if not root:
                return 0, 0
            rob_l, not_rob_l = dfs(root.left)
            rob_r, not_rob_r = dfs(root.right)
            robRoot = root.val + not_rob_l + not_rob_r
            notRobRoot = max(rob_l, not_rob_l) + max(rob_r, not_rob_r)
            return robRoot, notRobRoot
        return max(dfs(root))
```

### 298. Binary Tree Longest Consecutive Sequence

```python
class Solution:
    def longestConsecutive(self, root: Optional[TreeNode]) -> int:
        self.res = 1
        def dfs(node):
            if not node:
                return 1, -inf
            left_length, left_val = dfs(node.left)
            right_length, right_val = dfs(node.right)
            if node.val + 1 == left_val:
                left_length += 1
                self.res = max(self.res, left_length)
            elif node.val + 1 == right_val:
                right_length += 1
                self.res = max(self.res, right_length)
            else:
                left_length, right_length = 1, 1
            return max(left_length, right_length), node.val
        dfs(root)
        return self.res
```

### 1372. Longest ZigZag Path in a Binary Tree

```python
class Solution:
    def longestZigZag(self, root: Optional[TreeNode]) -> int:
        def dfs(node, is_left, depth):
            if not node:
                return depth
            if is_left:
                return max(dfs(node.left, True, 1), dfs(node.right, False, depth + 1))
            return max(dfs(node.right, False, 1), dfs(node.left, True, depth + 1))
        return dfs(root, True, 0) - 1
```

### 834. Sum of Distances in Tree

```python
class Solution:
    def sumOfDistancesInTree(self, n: int, edges: List[List[int]]) -> List[int]:
        g = defaultdict(list)
        for u, v in edges:
            g[u].append(v)
            g[v].append(u)

        res, subTreeSize = [0] * n, [1] * n 
        def dfs(x, parent, depth):
            res[0] += depth 
            for y in g[x]:
                if y != parent:
                    dfs(y, x, depth + 1)
                    subTreeSize[x] += subTreeSize[y]
        dfs(0, -1, 0)

        def reRoot(x, parent):
            for y in g[x]:
                if y != parent:
                    res[y] = res[x] + n - 2 * subTreeSize[y]
                    reRoot(y, x)
        reRoot(0, -1)
        return res
```

### 2581. Count Number of Possible Root Nodes

```python
class Solution:
    def rootCount(self, edges: List[List[int]], guesses: List[List[int]], k: int) -> int:
        g = defaultdict(list)
        for u, v in edges:
            g[u].append(v)
            g[v].append(u)

        self.cnt = 0
        guesses = set([(u, v) for u, v in guesses])
        def dfs(x, parent):
            for y in g[x]:
                if y != parent:
                    if (x, y) in guesses:
                        self.cnt += 1
                    dfs(y, x)
        dfs(0, -1)
        
        self.res = 0
        def reRoot(x, parent, cnt):
            if cnt >= k:
                self.res += 1
            for y in g[x]:
                if y != parent:
                    reRoot(y, x, cnt - ((x, y) in guesses) + ((y, x) in guesses))
        reRoot(0, -1, self.cnt)
        return self.res
```

### 2858. Minimum Edge Reversals So Every Node Is Reachable

```python
class Solution:
    def minEdgeReversals(self, n: int, edges: List[List[int]]) -> List[int]:
        g = defaultdict(list)
        for u, v in edges:
            g[u].append(v)
            g[v].append(u)

        s = set([(u, v) for u, v in edges])
        self.cnt = 0
        def dfs(x, parent):
            for y in g[x]:
                if y != parent:
                    if (x, y) not in s:
                        self.cnt += 1
                    dfs(y, x)
        dfs(0, -1)
        
        res = [0] * n 
        def reRoot(x, parent, cnt):
            res[x] = cnt 
            for y in g[x]:
                if y != parent:
                    reRoot(y, x, cnt + ((x, y) in s) - ((y, x) in s))
        reRoot(0, -1, self.cnt)
        return res
```