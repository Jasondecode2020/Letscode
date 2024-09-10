## Lowest Common Ancester of BT or BST(10)

* [235. Lowest Common Ancestor of a Binary Search Tree](#235-lowest-common-ancestor-of-a-binary-search-tree)
* [236. Lowest Common Ancestor of a Binary Tree](#236-lowest-common-ancestor-of-a-binary-tree)
* [1644. Lowest Common Ancestor of a Binary Tree II](#1644-lowest-common-ancestor-of-a-binary-tree-ii)
* [865. Smallest Subtree with all the Deepest Nodes](#865-smallest-subtree-with-all-the-deepest-nodes)
* [1123. Lowest Common Ancestor of Deepest Leaves](#1123-lowest-common-ancestor-of-deepest-leaves)

* [1676. Lowest Common Ancestor of a Binary Tree IV](#1676-lowest-common-ancestor-of-a-binary-tree-iv)
* [1257. Smallest Common Region](#1257-smallest-common-region) 1654
* [1650. Lowest Common Ancestor of a Binary Tree III](#1650-lowest-common-ancestor-of-a-binary-tree-iii)
* [1740. Find Distance in a Binary Tree](#1740-find-distance-in-a-binary-tree)
* [2096. Step-By-Step Directions From a Binary Tree Node to Another](#2096-step-by-step-directions-from-a-binary-tree-node-to-another)

### 235. Lowest Common Ancestor of a Binary Search Tree

```python
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        mn, mx = min(p.val, q.val), max(p.val, q.val)
        def dfs(node):
            if node.val < mn:
                return dfs(node.right)
            if node.val > mx:
                return dfs(node.left)
            return node 
        return dfs(root)
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
            left = dfs(node.left)
            right = dfs(node.right)
            if left and right:
                return node 
            return left or right
        return dfs(root)
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
            return left or right 
        
        def bfs():
            q = deque([root])
            while q:
                level = []
                res = list(q)
                for i in range(len(q)):
                    node = q.popleft()
                    if node.left:
                        q.append(node.left)
                    if node.right:
                        q.append(node.right)
            return res 
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
            return left or right 
        
        def bfs():
            q = deque([root])
            while q:
                res = list(q)
                for _ in range(len(q)):
                    node = q.popleft()
                    if node.left:
                        q.append(node.left)
                    if node.right:
                        q.append(node.right)
            return res 

        res = bfs()
        ans = res[0]
        for i in range(1, len(res)):
            ans = dfs(root, ans, res[i])
        return ans
```

### 1257. Smallest Common Region

```python
class Solution:
    def findSmallestRegion(self, regions: List[List[str]], region1: str, region2: str) -> str:
        parent = defaultdict(str)
        for region in regions:
            for r in region[1:]:
                parent[r] = region[0]

        s1 = set([region1])
        while parent[region1]:
            s1.add(parent[region1])
            region1 = parent[region1]

        while region2:
            if region2 in s1:
                return region2 
            region2 = parent[region2]
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
            return left or right 

        def dfs2(node, res):
            if node:
                if node in [p, q]:
                    res.append(node)
                dfs2(node.left, res)
                dfs2(node.right, res)
            return len(res) == 2
        return dfs(root) if dfs2(root, []) else None
```

### 1676. Lowest Common Ancestor of a Binary Tree IV

```python
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', nodes: 'List[TreeNode]') -> 'TreeNode':
        def dfs(node, p, q):
            if not node or node in [p, q]:
                return node 
            left, right = dfs(node.left, p, q), dfs(node.right, p, q)
            if left and right:
                return node 
            return left or right 
        res = nodes[0]
        for i in range(1, len(nodes)):
            res = dfs(root, res, nodes[i])
        return res 
```

### 1740. Find Distance in a Binary Tree

```python
class Solution:
    def findDistance(self, root: Optional[TreeNode], p: int, q: int) -> int:
        def dfs(node):
            if not node:
                return None 
            if node.val in [p, q]:
                return node 
            left, right = dfs(node.left), dfs(node.right)
            if left and right:
                return node 
            return left or right 

        root = dfs(root)
        queue = deque([(root, 0)])
        res = 0
        while queue:
            node, depth = queue.popleft()
            if node.val in [p, q]:
                res += depth 
            if node.left:
                queue.append((node.left, depth + 1))
            if node.right:
                queue.append((node.right, depth + 1))
        return res
```

### 2096. Step-By-Step Directions From a Binary Tree Node to Another

```python
class Solution:
    def getDirections(self, root: Optional[TreeNode], startValue: int, destValue: int) -> str:
        def dfs(node):
            if not node:
                return None
            if node.val in [startValue, destValue]:
                return node 
            left, right = dfs(node.left), dfs(node.right)
            if left and right:
                return node 
            return left or right 
        root = dfs(root)
        q = deque([(root, '')])
        res1, res2 = '', ''
        while q:
            node, direction = q.popleft()
            if node.val == startValue:
                res1 = len(direction) * 'U'
            if node.val == destValue:
                res2 = direction
            if node.left:
                q.append((node.left, direction + 'L'))
            if node.right:
                q.append((node.right, direction + 'R'))
        return res1 + res2 
```