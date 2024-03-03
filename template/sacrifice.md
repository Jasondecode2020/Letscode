### 979. Distribute Coins in Binary Tree

```python
class Solution:
    def distributeCoins(self, root: Optional[TreeNode]) -> int:
        self.res = 0
        def dfs(node):
            if not node:
                return 0, 0
            coins_l, nodes_l = dfs(node.left)
            coins_r, nodes_r = dfs(node.right)
            coins = coins_l + coins_r + node.val
            nodes = nodes_l + nodes_r + 1
            self.res += abs(coins - nodes)
            return coins, nodes 
        dfs(root)
        return self.res
```