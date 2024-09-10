
## bottom up DFS

* [111. Minimum Depth of Binary Tree](#111-minimum-depth-of-binary-tree)
* [951. Flip Equivalent Binary Trees](#951-flip-equivalent-binary-trees)
* [965. Univalued Binary Tree](#965-univalued-binary-tree)
* [663. Equal Tree Partition](#663-equal-tree-partition)
* [298. Binary Tree Longest Consecutive Sequence](#298-binary-tree-longest-consecutive-sequence)

* [1973. Count Nodes Equal to Sum of Descendants](#1973-count-nodes-equal-to-sum-of-descendants)


### 111. Minimum Depth of Binary Tree

```python
class Solution:
    def minDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        if root.left and not root.right:
            return self.minDepth(root.left) + 1
        if not root.left and root.right:
            return self.minDepth(root.right) + 1
        return min(self.minDepth(root.left), self.minDepth(root.right)) + 1
```

### 965. Univalued Binary Tree

```python
class Solution:
    def isUnivalTree(self, root: Optional[TreeNode]) -> bool:
        s = set()
        def dfs(node):
            if node:
                val = node.val
                s.add(val)
                dfs(node.left)
                dfs(node.right)
        dfs(root)
        return len(s) == 1
```

### 100. Same Tree

```python
class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        def dfs(p, q):
            if not p and not q:
                return True
            if (p and not q) or (q and not p) or p.val != q.val:
                return False
            return dfs(p.left, q.left) and dfs(p.right, q.right)
        return dfs(p, q)
```

### 101. Symmetric Tree

```python
class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
      def symmetric(p, q):
        if not p and not q:
          return True
        if p and not q or q and not p or p.val != q.val:
          return False
        return symmetric(p.left, q.right) and symmetric(p.right, q.left)
      return symmetric(root.left, root.right)
```

### 951. Flip Equivalent Binary Trees

```python
class Solution:
    def flipEquiv(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> bool:
        def dfs(r1, r2):
            if not r1 and not r2:
                return True
            if (r1 and not r2) or (r2 and not r1) or (r1.val != r2.val):
                return False
            return (dfs(r1.left, r2.left) and dfs(r1.right, r2.right)) or (dfs(r1.left, r2.right) and dfs(r1.right, r2.left))
        return dfs(root1, root2) 
```

### 1379. Find a Corresponding Node of a Binary Tree in a Clone of That Tree

```python
class Solution:
    def getTargetCopy(self, original: TreeNode, cloned: TreeNode, target: TreeNode) -> TreeNode:
        self.res = None
        def dfs(node, cloned):
            if not node:
                return None
            if node == target:
                self.res = cloned
            dfs(node.left, cloned.left)
            dfs(node.right, cloned.right)
        dfs(original, cloned)
        return self.res
```

### 110. Balanced Binary Tree

```python
class Solution:
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        def dfs(node):
            if not node:
                return 0
            left = dfs(node.left)
            if left == -1:
                return -1
            right = dfs(node.right)
            if right == -1 or abs(left - right) > 1:
                return -1
            return max(left, right) + 1
        return dfs(root) != -1
```

### 226. Invert Binary Tree

```python
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        def dfs(node):
            if not node:
                return None
            node.left, node.right = dfs(node.right), dfs(node.left)
            return node 
        return dfs(root)
```

### 617. Merge Two Binary Trees

```python
class Solution:
    def mergeTrees(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root1 and not root2:
            return None
        if not root1:
            return root2
        if not root2:
            return root1
        root = TreeNode(root1.val + root2.val)
        root.left = self.mergeTrees(root1.left, root2.left)
        root.right = self.mergeTrees(root1.right, root2.right)
        return root
```

### 2331. Evaluate Boolean Binary Tree

```python
class Solution:
    def evaluateTree(self, root: Optional[TreeNode]) -> bool:
        def dfs(node):
            if not node:
                return
            left = dfs(node.left)
            right = dfs(node.right)
            if node.val == 2:
                return left or right
            if node.val == 3:
                return left and right
            if node.val == 0:
                return False
            if node.val == 1:
                return True
        return dfs(root)
```

### 508. Most Frequent Subtree Sum

```python
class Solution:
    def findFrequentTreeSum(self, root: Optional[TreeNode]) -> List[int]:
        res = []
        def dfs(node):
            if not node:
                return 0
            left = dfs(node.left)
            right = dfs(node.right)
            res.append(node.val + left + right)
            return node.val + left + right
        dfs(root)
        d = Counter(res)
        maxFreq = d.most_common()[0][1]
        return [k for k, v in d.items() if v == maxFreq]
```

### 563. Binary Tree Tilt

```python
class Solution:
    def findTilt(self, root: Optional[TreeNode]) -> int:
        self.res = 0
        def dfs(node):
            if not node:
                return 0
            left_sum = dfs(node.left)
            right_sum = dfs(node.right)
            self.res += abs(left_sum - right_sum)
            return left_sum + right_sum + node.val
        dfs(root)
        return self.res
```

### 606. Construct String from Binary Tree

```python
class Solution:
    def tree2str(self, root: Optional[TreeNode]) -> str:
        def dfs(node):
            if not node:
                return ''
            left = dfs(node.left)
            right = dfs(node.right)
            res = ''
            if left and not right:
                res = '(' + left + ')'
            if right and not left:
                res = '()' + '(' + right + ')'
            if left and right:
                res = '(' + left + ')' + '(' + right + ')'
            return str(node.val) + res
        return dfs(root)
```

### 2265. Count Nodes Equal to Average of Subtree

```python
class Solution:
    def averageOfSubtree(self, root: Optional[TreeNode]) -> int:
        self.res = 0
        def postorder(node):
            if not node:
                return 0, 0
            l_count, l_sum = postorder(node.left)
            r_count, r_sum = postorder(node.right)
            if node.val == (l_sum + r_sum + node.val) // (l_count + r_count + 1):
                self.res += 1
            return l_count + r_count + 1, l_sum + r_sum + node.val
        postorder(root)
        return self.res
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

### 1339. Maximum Product of Splitted Binary Tree

```python
class Solution:
    def maxProduct(self, root: Optional[TreeNode]) -> int:
        def postorderSum(node):
            if node:
                postorderSum(node.left)
                postorderSum(node.right)
                self.total += node.val
        self.total = 0
        postorderSum(root)

        self.res, mod = 0, 10 ** 9 + 7
        def postorderMax(node):
            if not node:
                return 0
            l, r = postorderMax(node.left), postorderMax(node.right)
            curSum = node.val + l + r
            self.res = max(self.res, curSum * (self.total - curSum))
            return curSum
        postorderMax(root)
        return self.res % mod
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

### 572. Subtree of Another Tree

```python
class Solution:
    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        def sameTree(node1, node2):
            if not node1 and not node2:
                return True
            if node1 and not node2 or node2 and not node1 or node1.val != node2.val:
                return False
            return sameTree(node1.left, node2.left) and sameTree(node1.right, node2.right)
        
        def dfs(node):
            if not node:
                return False
            if sameTree(node, subRoot):
                return True
            return dfs(node.left) or dfs(node.right)
        return dfs(root)
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

### 250. Count Univalue Subtrees

```python
class Solution:
    def countUnivalSubtrees(self, root: Optional[TreeNode]) -> int:
        def checkUni(node, s):
            if node:
                s.add(node.val)
                if len(s) == 2:
                    return False
                checkUni(node.left, s)
                checkUni(node.right, s)
            return len(s) == 1

        self.res = 0
        def dfs(node):
            if node:
                if checkUni(node, set()):
                    self.res += 1
                dfs(node.left)
                dfs(node.right)
        dfs(root)
        return self.res
```

### 1120. Maximum Average Subtree

```python
class Solution:
    def maximumAverageSubtree(self, root: Optional[TreeNode]) -> float:
        def dfs(node):
            if not node:
                return 0, 0
            leftNum, leftVal = dfs(node.left)
            rightNum, rightVal = dfs(node.right)
            self.res = max(self.res, (leftVal + rightVal + node.val) / (leftNum + rightNum + 1))
            return leftNum + rightNum + 1, leftVal + rightVal + node.val 
        self.res = 0
        dfs(root)
        return self.res
```

### 333. Largest BST Subtree

```python
class Solution:
    def largestBSTSubtree(self, root: Optional[TreeNode]) -> int:
        def valid(node, l, r):
            if not node:
                return True
            v = node.val
            self.count += 1
            if v <= l or v >= r:
                return False
            return valid(node.left, l, v) and valid(node.right, v, r)

        self.res = 0
        def dfs(node):
            if node:
                self.count = 0
                if valid(node, -inf, inf):
                    self.res = max(self.res, self.count)
                    return
                dfs(node.left)
                dfs(node.right)
        dfs(root)
        return self.res 
```

### 366. Find Leaves of Binary Tree

```python
class Solution:
    def findLeaves(self, root: Optional[TreeNode]) -> List[List[int]]:
        def dfs(root):
            if not root:
                return 0
            l, r = dfs(root.left), dfs(root.right)
            depth = max(l, r) + 1
            res[depth].append(root.val)
            return depth

        res = defaultdict(list)
        dfs(root)
        return [v for v in res.values()]
```

### 156. Binary Tree Upside Down

```python
class Solution:
    def upsideDownBinaryTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        def dfs(node):
            if not node.left:
                return node
            newRoot = dfs(node.left)
            node.left.left = node.right
            node.left.right = node
            node.left = node.right = None
            return newRoot 
        if not root:
            return None
        return dfs(root)
```

### 1145. Binary Tree Coloring Game

```python
class Solution:
    def btreeGameWinningMove(self, root: Optional[TreeNode], n: int, x: int) -> bool:
        def dfs(node):
            if not node:
                return 0
            ls = dfs(node.left)
            rs = dfs(node.right)
            if node.val == x:
                self.left_size, self.right_size = ls, rs 
            return ls + rs + 1
            
        self.left_size = self.right_size = 0
        dfs(root)
        return max(self.left_size, self.right_size, n - 1 - self.left_size - self.right_size) * 2 > n 
```

### 1530. Number of Good Leaf Nodes Pairs

```python
class Solution:
    def countPairs(self, root: TreeNode, distance: int) -> int:
        def dfs(node):
            if not node:
                return []
            if not node.left and not node.right:
                return [1]
            l, r = dfs(node.left), dfs(node.right)
            for a in l:
                for b in r:
                    if a + b <= distance:
                        self.res += 1
            return [n + 1 for n in l + r if n + 1 <= distance]
        self.res = 0
        dfs(root)
        return self.res
```

### 1973. Count Nodes Equal to Sum of Descendants

```python
class Solution:
    def equalToDescendants(self, root: Optional[TreeNode]) -> int:
        def dfs(node):
            if not node:
                return 0
            l, r = dfs(node.left), dfs(node.right)
            if l + r == node.val:
                self.res += 1
            return l + r + node.val
        self.res = 0
        dfs(root)
        return self.res 
```

### 663. Equal Tree Partition

```python
class Solution:
    def checkEqualTree(self, root: Optional[TreeNode]) -> bool:
        def dfs(node):
            if node:
                self.res += node.val
                dfs(node.left)
                dfs(node.right)
        self.res = 0
        dfs(root)

        def dfs2(node):
            if not node:
                return 0
            l, r = dfs2(node.left), dfs2(node.right)
            res = l + r + node.val
            if node != origin and res == self.res / 2:
                self.ans = True 
            return res
        origin = root 
        self.ans = False 
        dfs2(root)
        return self.ans
```

### 2792. Count Nodes That Are Great Enough

### 1612. Check If Two Expression Trees are Equivalent

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