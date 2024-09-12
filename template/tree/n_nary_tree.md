## n-ary tree

* [589. N-ary Tree Preorder Traversal](#589-n-ary-tree-preorder-traversal)
* [590. N-ary Tree Postorder Traversal](#590-n-ary-tree-postorder-traversal)
* [559. Maximum Depth of N-ary Tree](#559-maximum-depth-of-n-ary-tree)
* [429. N-ary Tree Level Order Traversal](#429-n-ary-tree-level-order-traversal)
* [427. Construct Quad Tree](#427-construct-quad-tree)
* [558. Logical OR of Two Binary Grids Represented as Quad-Trees](#558-logical-or-of-two-binary-grids-represented-as-quad-trees)
* [1490. Clone N-ary Tree](#1490-clone-n-ary-tree)
* [1522. Diameter of N-Ary Tree](#1522-diameter-of-n-ary-tree)
* [1506. Find Root of N-Ary Tree](#1506-find-root-of-n-ary-tree)

### 589. N-ary Tree Preorder Traversal

```python
class Solution:
    def preorder(self, root: 'Node') -> List[int]:
        def dfs(node, res):
            if node:
                res.append(node.val)
                for child in node.children:
                    dfs(child, res)
            return res 
        return dfs(root, [])
```

```python
class Solution:
    def preorder(self, root: 'Node') -> List[int]:
        if not root:
            return []
        stack = [root]
        res = []
        while stack:
            node = stack.pop()
            res.append(node.val)
            for child in node.children[::-1]:
                stack.append(child)
        return res 
```

### 590. N-ary Tree Postorder Traversal

```python
class Solution:
    def postorder(self, root: 'Node') -> List[int]:
        def dfs(node, res):
            if node:
                for child in node.children:
                    dfs(child, res)
                res.append(node.val)
            return res 
        return dfs(root, [])
```

### 559. Maximum Depth of N-ary Tree

```python
class Solution:
    def maxDepth(self, root: 'Node') -> int:
        def dfs(node):
            if not node:
                return 0
            res = 0
            for child in node.children:
                res = max(res, dfs(child))
            return res + 1
        return dfs(root)
```

### 429. N-ary Tree Level Order Traversal

```python
class Solution:
    def levelOrder(self, root: 'Node') -> List[List[int]]:
        if not root:
            return []
        q = deque([root])
        res = []
        while q:
            level = []
            for i in range(len(q)):
                node = q.popleft()
                level.append(node.val)
                for child in node.children:
                    q.append(child)
            res.append(level)
        return res
```



### 427. Construct Quad Tree

```python
class Solution:
    def construct(self, grid: List[List[int]]) -> 'Node':
        def check(n, r, c):
            allSame = True 
            for i in range(n):
                for j in range(n):
                    if grid[r][c] != grid[r + i][c + j]:
                        allSame = False 
                        break
            return allSame

        def dfs(n, r, c):
            if check(n, r, c):
                return Node(grid[r][c], True)
            n //= 2 
            topLeft = dfs(n, r, c)
            topRight = dfs(n, r, c + n)
            bottomLeft = dfs(n, r + n, c)
            bottomRight = dfs(n, r + n, c + n)
            return Node(1, False, topLeft, topRight, bottomLeft, bottomRight)
        return dfs(len(grid), 0, 0)
```

### 558. Logical OR of Two Binary Grids Represented as Quad-Trees

```python
class Solution:
    def intersect(self, quadTree1: 'Node', quadTree2: 'Node') -> 'Node':
        if quadTree1.isLeaf:
            return Node(True, True) if quadTree1.val else quadTree2
        if quadTree2.isLeaf:
            return Node(True, True) if quadTree2.val else quadTree1
        topLeft = self.intersect(quadTree1.topLeft, quadTree2.topLeft)
        topRight = self.intersect(quadTree1.topRight, quadTree2.topRight)
        bottomLeft = self.intersect(quadTree1.bottomLeft, quadTree2.bottomLeft)
        bottomRight = self.intersect(quadTree1.bottomRight, quadTree2.bottomRight)
        if topLeft.isLeaf and topRight.isLeaf and bottomLeft.isLeaf and bottomRight.isLeaf and topLeft.val == topRight.val == bottomLeft.val == bottomRight.val:
            return Node(topLeft.val, True)
        return Node(False, False, topLeft, topRight, bottomLeft, bottomRight)
```

### 1490. Clone N-ary Tree

```python
class Solution:
    def cloneTree(self, root: 'Node') -> 'Node':
        def dfs(node):
            if not node:
                return None 
            cur = Node(node.val)
            for child in node.children:
                cur.children.append(dfs(child))
            return cur
        return dfs(root)
```

### 1506. Find Root of N-Ary Tree

```python
class Solution:
    def findRoot(self, tree: List['Node']) -> 'Node':
        child_set = set()
        for t in tree:
            for child in t.children:
                child_set.add(child)
        for t in tree:
            if t not in child_set:
                return t 
```

```python
class Solution:
    def findRoot(self, tree: List['Node']) -> 'Node':
        res = 0
        for t in tree:
            res ^= t.val
            for child in t.children:
                res ^= child.val 
        for t in tree:
            if t.val == res:
                return t
```

### 1522. Diameter of N-Ary Tree

```python
class Solution:
    def diameter(self, root: 'Node') -> int:
        """
        :type root: 'Node'
        :rtype: int
        """
        self.res = 0
        def dfs(node):
            first = 0
            for child in node.children:
                second = dfs(child) + 1
                self.res = max(self.res, first + second)
                first = max(first, second)
            return first
        dfs(root)
        return self.res
```

1516. 移动 N 叉树的子树（会员题）
428. 序列化和反序列化 N 叉树（会员题）