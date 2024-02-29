### 1120. Maximum Average Subtree

```python
class Solution:
    def maximumAverageSubtree(self, root: Optional[TreeNode]) -> float:
        def dfs(node):
            if not node:
                return 0, 0
            left, leftVal = dfs(node.left)
            right, rightVal = dfs(node.right)
            self.res = max(self.res, (leftVal + rightVal + node.val) / (left + right + 1))
            return left + right + 1, leftVal + rightVal + node.val 
        self.res = 0
        dfs(root)
        return self.res
```