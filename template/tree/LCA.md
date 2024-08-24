## Lowest Common Ancester of BT or BST(8)

* [235. Lowest Common Ancestor of a Binary Search Tree](#235-lowest-common-ancestor-of-a-binary-search-tree)
* [236. Lowest Common Ancestor of a Binary Tree](#236-lowest-common-ancestor-of-a-binary-tree)
* [1257. Smallest Common Region](#1257-smallest-common-region)
* [1644. Lowest Common Ancestor of a Binary Tree II](#1644-lowest-common-ancestor-of-a-binary-tree-ii)
* [1650. Lowest Common Ancestor of a Binary Tree III](#1650-lowest-common-ancestor-of-a-binary-tree-iii)
* [1676. Lowest Common Ancestor of a Binary Tree IV](#1676-lowest-common-ancestor-of-a-binary-tree-iv)
* [865. Smallest Subtree with all the Deepest Nodes](#865-smallest-subtree-with-all-the-deepest-nodes)
* [1123. Lowest Common Ancestor of Deepest Leaves](#1123-lowest-common-ancestor-of-deepest-leaves)

### 235. Lowest Common Ancestor of a Binary Search Tree

```python
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if root.val > max(p.val, q.val):
            return self.lowestCommonAncestor(root.left, p, q)
        elif root.val < min(p.val, q.val):
            return self.lowestCommonAncestor(root.right, p, q)
        return root
```

### 236. Lowest Common Ancestor of a Binary Tree

```python
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        def dfs(node):
            if node in [p, q]:
                return node 
            if not node:
                return None 
            left, right = dfs(node.left), dfs(node.right)
            if left and right:
                return node 
            if left:
                return left 
            if right:
                return right 
        return dfs(root)
```

### 1257. Smallest Common Region

```python
class Solution:
    def findSmallestRegion(self, regions: List[List[str]], region1: str, region2: str) -> str:
        p = defaultdict(str)
        for region in regions:
            for r in region[1:]:
                p[r] = region[0]

        s = set([region1])
        while p[region1]:
            s.add(p[region1])
            region1 = p[region1]
            
        while region2:
            if region2 in s:
                return region2
            region2 = p[region2]
```

### 1644. Lowest Common Ancestor of a Binary Tree II

```python
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        def dfs(node):
            if node in [p, q]:
                return node 
            if not node:
                return None 
            left, right = dfs(node.left), dfs(node.right)
            if left and right:
                return node 
            if left:
                return left 
            if right:
                return right 
                
        def dfs2(node, res):
            if node:
                if node in [p, q]:
                    res.append(node)
                dfs2(node.left, res)
                dfs2(node.right, res)
            return len(res) == 2
        return dfs(root) if dfs2(root, []) else None
```

### 1650. Lowest Common Ancestor of a Binary Tree III

```python
class Solution:
    def lowestCommonAncestor(self, p: 'Node', q: 'Node') -> 'Node':
        pSet = set([p])
        while p.parent:
            pSet.add(p.parent)
            p = p.parent

        while q:
            if q in pSet:
                return q
            q = q.parent
```

### 1676. Lowest Common Ancestor of a Binary Tree IV

```python
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', nodes: 'List[TreeNode]') -> 'TreeNode':
        def dfs(node, p, q):
            if node in [p, q]:
                return node 
            if not node:
                return None 
            left, right = dfs(node.left, p, q), dfs(node.right, p, q)
            if left and right:
                return node 
            if left or right:
                return left or right 
        res = nodes[0]
        for i in range(1, len(nodes)):
            res = dfs(root, res, nodes[i])
        return res
```

### 865. Smallest Subtree with all the Deepest Nodes

```python
class Solution:
    def subtreeWithAllDeepest(self, root: TreeNode) -> TreeNode:
        def dfs(node, p, q):
            if node in [p, q]:
                return node 
            if not node:
                return None 
            left, right = dfs(node.left, p, q), dfs(node.right, p, q)
            if left and right:
                return node 
            if left or right:
                return left or right 
        
        def bfs():
            q = [root]
            ans = []
            while len(q) > 0:
                ans = q.copy()
                level = []
                for i in range(len(q)):
                    node = q.pop()
                    if node.left:
                        level.append(node.left)
                    if node.right:
                        level.append(node.right)
                q = level
            return ans
        
        res = bfs()
        ans = res[0]
        for i in range(1, len(res)):
            ans = dfs(root, ans, res[i])
        return ans
```

### 1123. Lowest Common Ancestor of Deepest Leaves

```python
class Solution:
    def lcaDeepestLeaves(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        def dfs(node, p, q):
            if node in [p, q]:
                return node 
            if not node:
                return None 
            left, right = dfs(node.left, p, q), dfs(node.right, p, q)
            if left and right:
                return node 
            if left or right:
                return left or right 
        
        def bfs():
            q = [root]
            ans = []
            while len(q) > 0:
                ans = q.copy()
                level = []
                for i in range(len(q)):
                    node = q.pop()
                    if node.left:
                        level.append(node.left)
                    if node.right:
                        level.append(node.right)
                q = level
            return ans
        
        res = bfs()
        ans = res[0]
        for i in range(1, len(res)):
            ans = dfs(root, ans, res[i])
        return ans
```