## Diameter

* [543. Diameter of Binary Tree](#543-diameter-of-binary-tree)
* [124. Binary Tree Maximum Path Sum](#124-binary-tree-maximum-path-sum)
* [2385. Amount of Time for Binary Tree to Be Infected](#2385-amount-of-time-for-binary-tree-to-be-infected)
* [687. Longest Univalue Path](#687-longest-univalue-path)
* [549. Binary Tree Longest Consecutive Sequence II]()
### 543. Diameter of Binary Tree

```python
class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        def dfs(node):
            if not node:
                return -1
            l, r = dfs(node.left), dfs(node.right)
            self.res = max(self.res, l + r + 2)
            return max(l, r) + 1
        self.res = 0
        dfs(root)
        return self.res 
```

### 124. Binary Tree Maximum Path Sum

```python
class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        def dfs(node):
            if not node:
                return 0
            l, r = max(dfs(node.left), 0), max(dfs(node.right), 0)
            self.res = max(self.res, node.val + l + r)
            return node.val + max(l, r)
        self.res = -inf
        dfs(root)
        return self.res 
```

### 2385. Amount of Time for Binary Tree to Be Infected

```python
class Solution:
    def amountOfTime(self, root: Optional[TreeNode], start: int) -> int:
        g = defaultdict(list)
        def dfs(node):
            if node:
                if node.left:
                    g[node.val].append(node.left.val)
                    g[node.left.val].append(node.val)
                    dfs(node.left)
                if node.right:
                    g[node.val].append(node.right.val)
                    g[node.right.val].append(node.val)
                    dfs(node.right)

        dfs(root)
        visited = set([start])
        q = deque([(start, 0)])
        while q:
            node, dist = q.popleft()
            for nei in g[node]:
                if nei not in visited:
                    visited.add(nei)
                    q.append((nei, dist + 1))
        return dist

```

### 687. Longest Univalue Path

```python
class Solution:
    def longestUnivaluePath(self, root: Optional[TreeNode]) -> int:
        def dfs(node):
            if not node:
                return -1
            l_len = dfs(node.left) + 1
            r_len = dfs(node.right) + 1
            if node.left and node.left.val != node.val:
                l_len = 0
            if node.right and node.right.val != node.val:
                r_len = 0
            self.res = max(self.res, l_len + r_len)
            return max(l_len, r_len)
        self.res = 0
        dfs(root)
        return self.res
```

### 549. Binary Tree Longest Consecutive Sequence II

```python
class Solution:
    def longestConsecutive(self, root: Optional[TreeNode]) -> int:
        def dfs(root):
            if not root: return 0, 0
            left, right = dfs(root.left), dfs(root.right)
            inc = dec = 1
            if root.left:
                if root.left.val == root.val + 1:
                    inc += left[0]
                elif root.left.val == root.val - 1:
                    dec += left[1]
            if root.right:
                if root.right.val == root.val + 1:
                    inc = max(inc, right[0] + 1)
                elif root.right.val == root.val - 1:
                    dec = max(dec, right[1] + 1)
            
            self.res = max(self.res, inc + dec - 1)
            return inc, dec
        
        self.res = 1
        dfs(root)
        return self.res
```