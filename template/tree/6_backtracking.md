* [257. Binary Tree Paths](#257-binary-tree-paths)
* [113. Path Sum II](#113-path-sum-ii)
* [437. Path Sum III](#437-path-sum-iii)

### 257. Binary Tree Paths

```python
class Solution:
    def binaryTreePaths(self, root: Optional[TreeNode]) -> List[str]:
        def check(res):
            for i, a in enumerate(res):
                a = [str(c) for c in a]
                res[i] = '->'.join(a)
            return res
            
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

### 113. Path Sum II

```python
class Solution:
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> List[List[int]]:
        res = []
        def dfs(node, path, presum):
            if not node:
                return []
            if not node.left and not node.right and presum + node.val == targetSum:
                res.append(path + [node.val])
            dfs(node.right, path + [node.val], presum + node.val)
            dfs(node.left, path + [node.val], presum + node.val)
            return res
        return dfs(root, [], 0)
```

### 437. Path Sum III

```python
class Solution:
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> int:
        self.res = 0
        def dfs(node, presum):
            if not node:
                return
            if presum + node.val == targetSum:
                self.res += 1
            dfs(node.right, presum + node.val)
            dfs(node.left, presum + node.val)
            
        def preorder(node):
            if node:
                dfs(node, 0)
                preorder(node.left)
                preorder(node.right)
        preorder(root)
        return self.res
```

```python
class Solution:
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> int:
        self.res = 0
        prefix = defaultdict(int)
        prefix[0] = 1
        def dfs(node, presum):
            if not node:
                return
            curr = presum + node.val
            self.res += prefix[curr - targetSum]
            prefix[curr] += 1
            dfs(node.right, presum + node.val)
            dfs(node.left, presum + node.val)
            prefix[curr] -= 1
        dfs(root, 0)
        return self.res
```
