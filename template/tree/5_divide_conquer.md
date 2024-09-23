## D & Q (4)

* [538. Convert BST to Greater Tree](#538-convert-bst-to-greater-tree)
* [1038. Binary Search Tree to Greater Sum Tree](#1038-binary-search-tree-to-greater-sum-tree)
* [865. Smallest Subtree with all the Deepest Nodes](#865-smallest-subtree-with-all-the-deepest-nodes)
* [1080. Insufficient Nodes in Root to Leaf Paths](#1080-insufficient-nodes-in-root-to-leaf-paths)

### 538. Convert BST to Greater Tree

```python
class Solution:
    def convertBST(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        self.res = 0
        def dfs(node):
            if node:
                dfs(node.right)
                self.res += node.val 
                node.val = self.res
                dfs(node.left)
        dfs(root)
        return root
```

### 1038. Binary Search Tree to Greater Sum Tree

```python
class Solution:
    def bstToGst(self, root: TreeNode) -> TreeNode:
        self.res = 0
        def dfs(node):
            if node:
                dfs(node.right)
                self.res += node.val
                node.val = self.res
                dfs(node.left) 
        dfs(root)
        return root
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

### 1080. Insufficient Nodes in Root to Leaf Paths

```python
class Solution:
    def sufficientSubset(self, root: Optional[TreeNode], limit: int) -> Optional[TreeNode]:
        limit -= root.val 
        if root.left is root.right:
            return None if limit > 0 else root 
        if root.left:
            root.left = self.sufficientSubset(root.left, limit)
        if root.right:
            root.right = self.sufficientSubset(root.right, limit)
        return root if root.left or root.right else None
```