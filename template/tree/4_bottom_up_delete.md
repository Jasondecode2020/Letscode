
## bottom up DFS deletion

* [814. Binary Tree Pruning](#814-binary-tree-pruning)
* [1325. Delete Leaves With a Given Value](#1325-delete-leaves-with-a-given-value)
* [1110. Delete Nodes And Return Forest](#)
### 814. Binary Tree Pruning

```python
class Solution:
    def pruneTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root:
            return None 
        root.left = self.pruneTree(root.left)
        root.right = self.pruneTree(root.right)
        if not root.left and not root.right and root.val == 0:
            return None
        return root
```

### 1325. Delete Leaves With a Given Value

```python
class Solution:
    def removeLeafNodes(self, root: Optional[TreeNode], target: int) -> Optional[TreeNode]:
        if not root:
            return None
        root.left = self.removeLeafNodes(root.left, target)
        root.right = self.removeLeafNodes(root.right, target)
        if not root.left and not root.right and root.val == target:
            return None
        return root
```

### 1110. Delete Nodes And Return Forest

```python
class Solution:
    def delNodes(self, root: Optional[TreeNode], to_delete: List[int]) -> List[TreeNode]:
        res = []
        s = set(to_delete)
        def dfs(node):
            if not node:
                return None
            node.left = dfs(node.left)
            node.right = dfs(node.right)
            if node.val not in s:
                return node
            if node.left:
                res.append(node.left)
            if node.right:
                res.append(node.right)
            return None
            
        if dfs(root):
            res.append(root)
        return res
```