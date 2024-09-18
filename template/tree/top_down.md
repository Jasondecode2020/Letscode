
## 2 top down DFS(16)

* [104. Maximum Depth of Binary Tree](#104-maximum-depth-of-binary-tree)
* [111. Minimum Depth of Binary Tree](#111-minimum-depth-of-binary-tree)
* [112. Path Sum](#112-path-sum)
* [129. Sum Root to Leaf Numbers](#129-sum-root-to-leaf-numbers)
* [199. Binary Tree Right Side View](#199-binary-tree-right-side-view)

* [1448. Count Good Nodes in Binary Tree](#1448-count-good-nodes-in-binary-tree)
* [1315. Sum of Nodes with Even-Valued Grandparent](#1315-sum-of-nodes-with-even-valued-grandparent)
* [988. Smallest String Starting From Leaf](#988-smallest-string-starting-from-leaf)
* [1026. Maximum Difference Between Node and Ancestor](#1026-maximum-difference-between-node-and-ancestor)
* [1022. Sum of Root To Leaf Binary Numbers](#1022-sum-of-root-to-leaf-binary-numbers)

* [623. Add One Row to Tree](#623-add-one-row-to-tree)
* [1372. Longest ZigZag Path in a Binary Tree](#1372-longest-zigzag-path-in-a-binary-tree)
* [1457. Pseudo-Palindromic Paths in a Binary Tree](#1457-pseudo-palindromic-paths-in-a-binary-tree)
* [2689. Extract Kth Character From The Rope Tree](#2689-extract-kth-character-from-the-rope-tree)
* [971. Flip Binary Tree To Match Preorder Traversal](#971-flip-binary-tree-to-match-preorder-traversal)

* [1430. Check If a String Is a Valid Sequence from Root to Leaves Path in a Binary Tree](#1430-check-if-a-string-is-a-valid-sequence-from-root-to-leaves-path-in-a-binary-tree)
* [545. Boundary of Binary Tree](#545-boundary-of-binary-tree)


### 104. Maximum Depth of Binary Tree

```python
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        def dfs(node):
            if not node:
                return 0
            return max(dfs(node.left), dfs(node.right)) + 1
        return dfs(root)
```

### 111. Minimum Depth of Binary Tree

```python
class Solution:
    def minDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        q = deque([(root, 1)])
        while q:
            node, depth = q.popleft()
            if not node.left and not node.right:
                return depth
            if node.left:
                q.append((node.left, depth + 1))
            if node.right:
                q.append((node.right, depth + 1))
```

### 112. Path Sum

```python
class Solution:
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        def dfs(node, presum):
            if not node:
                return False
            if not node.left and not node.right:
                return presum + node.val == targetSum
            return dfs(node.left, presum + node.val) or dfs(node.right, presum + node.val)
            
        return dfs(root, 0)
```

### 129. Sum Root to Leaf Numbers

```python
class Solution:
    def sumNumbers(self, root: Optional[TreeNode]) -> int:
        def check(res):
            for i, a in enumerate(res):
                n = len(a)
                res[i] = sum(a[j] * 10 ** (n - j - 1) for j in range(n))
            return sum(res)
            
        self.res = []
        def dfs(node, path):
            if node:
                if not node.left and not node.right:
                    self.res.append(path + [node.val])
                    return
                dfs(node.left, path + [node.val])
                dfs(node.right, path + [node.val])
        dfs(root, [])
        return check(self.res)
```

### 199. Binary Tree Right Side View

```python
class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []
        res, q = [], deque([root])
        while q:
            n = len(q)
            for i in range(n):
                node = q.popleft()
                if i == n - 1:
                    res.append(node.val)
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
        return res
```

### 1448. Count Good Nodes in Binary Tree

```python
class Solution:
    def goodNodes(self, root: TreeNode, mx=-inf) -> int:
        if not root:
            return 0
        l = self.goodNodes(root.left, max(mx, root.val))
        r = self.goodNodes(root.right, max(mx, root.val))
        return l + r + (root.val >= mx)
```

### 1315. Sum of Nodes with Even-Valued Grandparent

```python
class Solution:
    def sumEvenGrandparent(self, root: TreeNode) -> int:
        def dfs(node):
            if node:
                if node.val % 2 == 0:
                    if node.left:
                        if node.left.left:
                            self.res += node.left.left.val 
                        if node.left.right:
                            self.res += node.left.right.val 
                    if node.right:
                        if node.right.left:
                            self.res += node.right.left.val
                        if node.right.right:
                            self.res += node.right.right.val
                dfs(node.left)
                dfs(node.right)
        self.res = 0
        dfs(root)
        return self.res 
```

### 988. Smallest String Starting From Leaf

```python
class Solution:
    def smallestFromLeaf(self, root: Optional[TreeNode]) -> str:
        letters = ascii_lowercase
        def check(res):
            for i, a in enumerate(res):
                res[i] = ''.join([letters[i] for i in a])
            return sorted(res)[0]
            
        self.res = []
        def backtrack(node, path):
            if node:
                if not node.left and not node.right:
                    self.res.append([node.val] + path)
                    return
                backtrack(node.left, [node.val] + path)
                backtrack(node.right, [node.val] + path)
        backtrack(root, [])
        return check(self.res)
```

### 1026. Maximum Difference Between Node and Ancestor

```python
class Solution:
    def maxAncestorDiff(self, root: Optional[TreeNode]) -> int:
        def dfs(node):
            if not node:
                return -inf, inf
            mx = mn = node.val
            max_l, min_l = dfs(node.left)
            max_r, min_r = dfs(node.right)
            mx = max(mx, max_l, max_r)
            mn = min(mn, min_l, min_r)
            self.res = max(self.res, mx - node.val, node.val - mn)
            return mx, mn
            
        self.res = 0
        dfs(root)
        return self.res
```

### 1022. Sum of Root To Leaf Binary Numbers

```python
class Solution:
    def sumRootToLeaf(self, root: Optional[TreeNode]) -> int:
        def check(a):
            return int(''.join([str(i) for i in a]), 2)

        res = []
        def dfs(node, path, presum):
            if not node:
                return []
            if not node.left and not node.right:
                res.append(path + [node.val])
            dfs(node.right, path + [node.val], presum + node.val)
            dfs(node.left, path + [node.val], presum + node.val)
            return res
        arr = dfs(root, [], 0)
        ans = 0
        for a in arr:
            ans += check(a)
        return ans
```

### 623. Add One Row to Tree

```python
class Solution:
    def addOneRow(self, root: Optional[TreeNode], val: int, depth: int) -> Optional[TreeNode]:
        if depth == 1:
            return TreeNode(val, root, None)
        level = [root]
        for i in range(2, depth):
            temp = []
            for node in level:
                if node.left:
                    temp.append(node.left)
                if node.right:
                    temp.append(node.right)
            level = temp
        for node in level:
            node.left = TreeNode(val, node.left, None)
            node.right = TreeNode(val, None, node.right)
        return root
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

### 1457. Pseudo-Palindromic Paths in a Binary Tree

```python
class Solution:
    def pseudoPalindromicPaths (self, root: Optional[TreeNode]) -> int:
        p = [0] * 10
        def dfs(node):
            if not node:
                return 0
            p[node.val] ^= 1
            if node.left == node.right:
                res = 1 if sum(p) <= 1 else 0
            else:
                res = dfs(node.left) + dfs(node.right)
            p[node.val] ^= 1
            return res 
        return dfs(root)
```

### 2689. Extract Kth Character From The Rope Tree

```python
class Solution:
    def getKthCharacter(self, root: Optional[object], k: int) -> str:
        """
        :type root: Optional[RopeTreeNode]
        """
        self.res = ''
        def dfs(node):
            if node:
                if not node.left and not node.right:
                    self.res += node.val
                dfs(node.left)
                dfs(node.right)
        dfs(root)
        return self.res[k - 1]
```

### 298. Binary Tree Longest Consecutive Sequence

```python
class Solution:
    def longestConsecutive(self, root: Optional[TreeNode]) -> int:
        def dfs(node):
            if not node:
                return 0
            l, r = dfs(node.left), dfs(node.right)
            ans = 1
            if node.left and node.left.val == node.val + 1:
                ans = max(ans, l + 1)
            if node.right and node.right.val == node.val + 1:
                ans = max(ans, r + 1)
            self.res = max(self.res, ans)
            return ans 
        self.res = 0
        dfs(root)
        return self.res
```

### 971. Flip Binary Tree To Match Preorder Traversal

```python
class Solution:
    def flipMatchVoyage(self, root: Optional[TreeNode], voyage: List[int]) -> List[int]:
        stack = [root] 
        i = 0
        res = []
        while stack:
            node = stack.pop()
            if node.val != voyage[i]:
                return [-1]
            i += 1
            if node.left and node.left.val != voyage[i]:
                node.left, node.right = node.right, node.left 
                res.append(node.val)
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
        return res 
```

### 1430. Check If a String Is a Valid Sequence from Root to Leaves Path in a Binary Tree

```python
class Solution:
    def isValidSequence(self, root: Optional[TreeNode], arr: List[int]) -> bool:
        n = len(arr)
        def dfs(node, i):
            if i >= n or not node or node.val != arr[i]:
                return False
            elif not node.left and not node.right and i == n - 1:
                return True
            return dfs(node.left, i + 1) or dfs(node.right, i + 1)
        return dfs(root, 0)
```

### 545. Boundary of Binary Tree

```python
class Solution:
    def boundaryOfBinaryTree(self, root: Optional[TreeNode]) -> List[int]:
        if root and not root.left and not root.right:
            return [root.val]
        left, bottom, right = [], [], []
        def dfsLeft(node):
            if node and (node.left or node.right):
                left.append(node.val)
                if node.left:
                    dfsLeft(node.left)
                else:
                    dfsLeft(node.right)
        
        def dfsRight(node):
            if node and (node.left or node.right):
                right.append(node.val)
                if node.right:
                    dfsRight(node.right)
                else:
                    dfsRight(node.left)

        def dfsBottom(node):
            if node:
                if not node.left and not node.right:
                    bottom.append(node.val)
                dfsBottom(node.left)
                dfsBottom(node.right)

        dfsLeft(root.left)
        dfsBottom(root)
        dfsRight(root.right)
        return [root.val] + left + bottom + right[::-1]
```
