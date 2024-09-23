## BST(20)

* [98. Validate Binary Search Tree](#98-validate-binary-search-tree)
* [230. Kth Smallest Element in a BST](#230-kth-smallest-element-in-a-bst)
* [501. Find Mode in Binary Search Tree](#501-find-mode-in-binary-search-tree)
* [99. Recover Binary Search Tree](#99-recover-binary-search-tree)
* [700. Search in a Binary Search Tree](#700-search-in-a-binary-search-tree)

* [530. Minimum Absolute Difference in BST](#530-minimum-absolute-difference-in-bst)
* [783. Minimum Distance Between BST Nodes](#783-minimum-distance-between-bst-nodes)
* [1305. All Elements in Two Binary Search Trees](#1305-all-elements-in-two-binary-search-trees)
* [938. Range Sum of BST](#938-range-sum-of-bst)
* [653. Two Sum IV - Input is a BST](#653-two-sum-iv---input-is-a-bst)

* [897. Increasing Order Search Tree](#897-increasing-order-search-tree)
* [2476. Closest Nodes Queries in a Binary Search Tree](#2476-closest-nodes-queries-in-a-binary-search-tree)
* [270. Closest Binary Search Tree Value](#270-closest-binary-search-tree-value)
* [272. Closest Binary Search Tree Value II](#272-closest-binary-search-tree-value-ii)
* [285. Inorder Successor in BST](#285-inorder-successor-in-bst)

* [510. Inorder Successor in BST II](#510-inorder-successor-in-bst-ii)
* [255. Verify Preorder Sequence in Binary Search Tree](#255-verify-preorder-sequence-in-binary-search-tree)
* [1902. Depth of BST Given Insertion Order](#1902-depth-of-bst-given-insertion-order)
* [1373. Maximum Sum BST in Binary Tree](#1373-maximum-sum-bst-in-binary-tree)
* [1932. Merge BSTs to Create Single BST](#1932-merge-bsts-to-create-single-bst)

### 98. Validate Binary Search Tree

```python
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        def dfs(node, l, r):
            if not node:
                return True
            m = node.val
            if m <= l or m >= r:
                return False
            return dfs(node.left, l, m) and dfs(node.right, m, r)
        return dfs(root, -inf, inf)
```

### 230. Kth Smallest Element in a BST

```python
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        def inorder(root, res):
            if root:
                inorder(root.left, res)
                res.append(root.val)
                inorder(root.right, res)
            return res
        res = inorder(root, [])
        return res[k - 1]
```

### 501. Find Mode in Binary Search Tree

```python
class Solution:
    def findMode(self, root: Optional[TreeNode]) -> List[int]:
        res = []
        def dfs(node):
            if node:
                res.append(node.val)
                dfs(node.left)
                dfs(node.right)
        dfs(root)
        freqMax = Counter(res).most_common()[0][1]
        return [k for k, v in Counter(res).items() if v == freqMax]
```

### 99. Recover Binary Search Tree

```python
class Solution:
    def recoverTree(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        def inorder(node, res):
            if node:
                inorder(node.left, res)
                res.append(node.val)
                inorder(node.right, res)
            return res
        res = inorder(root, [])
        swap = []
        for a, b in zip(res, sorted(res)):
            if a != b:
                swap.append(a)
        def inorderSwap(node):
            if node:
                inorderSwap(node.left)
                if node.val == swap[0]:
                    node.val = swap[1]
                elif node.val == swap[1]:
                    node.val = swap[0]
                inorderSwap(node.right)
        inorderSwap(root)
        return root
```

### 700. Search in a Binary Search Tree

```python
class Solution:
    def searchBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        self.res = None
        def dfs(node):
            if node:
                if node.val > val:
                    dfs(node.left)
                elif node.val < val:
                    dfs(node.right)
                else:
                    self.res = node
        dfs(root)
        return self.res
```

### 530. Minimum Absolute Difference in BST

```python
class Solution:
    def getMinimumDifference(self, root: Optional[TreeNode]) -> int:
        def inorder(root, res):
            if root:
                inorder(root.left, res)
                res.append(root.val)
                inorder(root.right, res)
            return res
        res = inorder(root, [])
        return min(abs(res[i] - res[i - 1]) for i in range(1, len(res)))
```

### 783. Minimum Distance Between BST Nodes

```python
class Solution:
    def minDiffInBST(self, root: Optional[TreeNode]) -> int:
        def inorder(root, res):
            if root:
                inorder(root.left, res)
                res.append(root.val)
                inorder(root.right, res)
            return res
        res = inorder(root, [])
        return min(abs(res[i] - res[i - 1]) for i in range(1, len(res)))
```

### 1305. All Elements in Two Binary Search Trees

```python
from sortedcontainers import SortedList
class Solution:
    def getAllElements(self, root1: TreeNode, root2: TreeNode) -> List[int]:
        sl = SortedList()
        def dfs(node):
            if node:
                sl.add(node.val)
                dfs(node.left)
                dfs(node.right)
        dfs(root1)
        dfs(root2)
        return sl
```

### 938. Range Sum of BST

```python
class Solution:
    def rangeSumBST(self, root: Optional[TreeNode], low: int, high: int) -> int:
        self.res = 0
        def dfs(node):
            if node:
                if node.val < low:
                    dfs(node.right)
                elif node.val > high:
                    dfs(node.left)
                else:
                    self.res += node.val
                    dfs(node.left)
                    dfs(node.right)
        dfs(root)
        return self.res
```

### 653. Two Sum IV - Input is a BST

```python
class Solution:
    def findTarget(self, root: Optional[TreeNode], k: int) -> bool:
        s = set()
        def dfs(node):
            if node:
                res = k - node.val
                if res in s:
                    return True
                s.add(node.val)
                return dfs(node.left) or dfs(node.right)
            return False
        return dfs(root)
```

### 897. Increasing Order Search Tree

```python
class Solution:
    def increasingBST(self, root: TreeNode) -> TreeNode:
        def dfs(node, res):
            if node:
                dfs(node.left, res)
                res.append(node)
                dfs(node.right, res)
            return res
        res = dfs(root, [])
        for i in range(1, len(res)):
            res[i - 1].left = None
            res[i - 1].right = res[i]
        res[-1].left = None
        return res[0]
```

### 2476. Closest Nodes Queries in a Binary Search Tree

```python
class Solution:
    def closestNodes(self, root: Optional[TreeNode], queries: List[int]) -> List[List[int]]:
        def dfs(node, res):
            if node:
                dfs(node.left, res)
                res.append(node.val)
                dfs(node.right, res)
            return res 
        res = dfs(root, [])
        res = [-1] + res + [inf]
        ans = []
        for q in queries:
            i = bisect_left(res, q)
            mn = res[i] if res[i] == q else res[i - 1]
            j = bisect_right(res, q)
            mx = res[i] if res[i] != inf else -1
            ans.append([mn, mx])
        return ans
```

### 270. Closest Binary Search Tree Value

```python
class Solution:
    def closestValue(self, root: Optional[TreeNode], target: float) -> int:
        self.res = inf
        def dfs(node):
            if node:
                dfs(node.left)
                if abs(node.val - target) < abs(self.res - target):
                    self.res = node.val
                dfs(node.right)
        dfs(root)
        return self.res
```

### 272. Closest Binary Search Tree Value II

```python
class Solution:
    def closestKValues(self, root: Optional[TreeNode], target: float, k: int) -> List[int]:
        pq = []
        def dfs(node):
            if node:
                heappush(pq, (abs(node.val - target), node.val))
                dfs(node.left)
                dfs(node.right)
        dfs(root)

        res = []
        for i in range(k):
            res.append(heappop(pq)[1])
        return res
```

### 285. Inorder Successor in BST

```python
class Solution:
    def inorderSuccessor(self, root: TreeNode, p: TreeNode) -> Optional[TreeNode]:
        def dfs(node, res):
            if node:
                dfs(node.left, res)
                res.append(node)
                dfs(node.right, res)
            return res 
        res = dfs(root, [])

        n = len(res)
        for i in range(n - 1):
            if res[i] == p:
                return res[i + 1]
        return None
```

### 510. Inorder Successor in BST II

```python
class Solution:
    def inorderSuccessor(self, node: 'Node') -> 'Optional[Node]':
        p = node
        while p.parent:
            p = p.parent
        def dfs(node, res):
            if node:
                dfs(node.left, res)
                res.append(node)
                dfs(node.right, res)
            return res 
        res = dfs(p, [])

        n = len(res)
        for i in range(n - 1):
            if res[i] == node:
                return res[i + 1]
        return None
```

### 669. Trim a Binary Search Tree

```python
class Solution:
    def trimBST(self, root: Optional[TreeNode], low: int, high: int) -> Optional[TreeNode]:
        def dfs(node):
            if not node:
                return None 
            left, right = dfs(node.left), dfs(node.right) 
            if node.val > high:
                return left
            if node.val < low:
                return right
            node.left, node.right = left, right
            return node 
        return dfs(root)
```


### 426. Convert Binary Search Tree to Sorted Doubly Linked List

```python
class Solution:
    def treeToDoublyList(self, root: 'Optional[Node]') -> 'Optional[Node]':
        if not root:
            return None 
        self.first = self.last = None 
        self.inorder(root)
        self.first.left = self.last 
        self.last.right = self.first 
        return self.first 
    def inorder(self, node):
        if node:
            self.inorder(node.left)
            if not self.last:
                self.first = node 
            else:
                node.left = self.last 
                self.last.right = node 
            self.last = node 
            self.inorder(node.right)
```

### 776. Split BST

```python
class Solution:
    def splitBST(self, root: Optional[TreeNode], target: int) -> List[Optional[TreeNode]]:
        if not root:
            return None, None 
        if root.val <= target:
            L, R = self.splitBST(root.right, target)
            root.right = L 
            return [root, R]
        else:
            L, R = self.splitBST(root.left, target)
            root.left = R 
            return [L, root]
```

- need to do

### 255. Verify Preorder Sequence in Binary Search Tree

```python
class Solution:
    def verifyPreorder(self, preorder: List[int]) -> bool:
        minVal, stack = -inf, []
        for i, n in enumerate(preorder):
            if n < minVal:
                return False
            while stack and preorder[stack[-1]] < preorder[i]:
                minVal = preorder[stack.pop()] 
            stack.append(i)
        return True
```

### 1902. Depth of BST Given Insertion Order

### 1373. Maximum Sum BST in Binary Tree

### 1932. Merge BSTs to Create Single BST
