## template

```python
def dfs(root):
    if not root:
        return
    res = 0
    # some code
    dfs(root.left)
    dfs(root.right)
    return res
```

## template

```python
def dfs(root):
    stack = [root]
    res = 0
    while stack:
        node = stack.pop()
        # some code
        if node.left:
            stack.append(node.left)
        if node.right:
            stack.append(node.right)
    return res
```
### Tree Traversal

* `404. Sum of Left Leaves`
* `129. Sum Root to Leaf Numbers`
* `257. Binary Tree Paths`
* `988. Smallest String Starting From Leaf`
* `270. Closest Binary Search Tree Value`
* `272. Closest Binary Search Tree Value II`
* `226. Invert Binary Tree`
* `590. N-ary Tree Postorder Traversal`

### Construct Binary Tree

* `105. Construct Binary Tree from Preorder and Inorder Traversal`
* `106. Construct Binary Tree from Inorder and Postorder Traversal`
* `889. Construct Binary Tree from Preorder and Postorder Traversal`

### Convert to BST

* `108. Convert Sorted Array to Binary Search Tree`
* `109. Convert Sorted List to Binary Search Tree`
* `1008. Construct Binary Search Tree from Preorder Traversal`

### Balanced Binary Tree

* `110. Balanced Binary Tree`

### BST

* `222. Count Complete Tree Nodes`
* `333. Largest BST Subtree`

### 105. Construct Binary Tree from Preorder and Inorder Traversal

```python
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        def dfs(pre_l, pre_r, in_l, in_r):
            if pre_l > pre_r:
                return None
            pre_root = pre_l
            in_root = idx[preorder[pre_root]]
            root = TreeNode(preorder[pre_root])
            subtree_l = in_root - in_l
            root.left = dfs(pre_l + 1, pre_l + subtree_l, in_l, in_root - 1)
            root.right = dfs(pre_l + subtree_l + 1, pre_r, in_root + 1, in_r)
            return root

        n = len(preorder)
        idx = {v: i for i, v in enumerate(inorder)}
        return dfs(0, n - 1, 0, n - 1)
```

### 106. Construct Binary Tree from Inorder and Postorder Traversal

```python
class Solution:
    def buildTree(self, inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        def dfs(post_l, post_r, in_l, in_r):
            if post_l > post_r:
                return None
            post_root = post_r
            in_root = idx[postorder[post_root]]
            subtree_l = in_root - in_l
            root = TreeNode(postorder[post_root])
            root.left = dfs(post_l, post_l + subtree_l - 1, in_l, in_root - 1)
            root.right = dfs(post_l + subtree_l, post_r - 1, in_root + 1, in_r)
            return root

        n = len(postorder)
        idx = {v: i for i, v in enumerate(inorder)}
        return dfs(0, n - 1, 0, n - 1)
```

### 889. Construct Binary Tree from Preorder and Postorder Traversal

```python
class Solution:
    def constructFromPrePost(self, preorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        def dfs(pre, post):
            if not pre: 
                return None
            root = TreeNode(pre[0])
            if len(pre) == 1: 
                return root

            L = post.index(pre[1]) + 1
            root.left = dfs(pre[1:L+1], post[:L])
            root.right = dfs(pre[L+1:], post[L:-1])
            return root
        return dfs(preorder, postorder)
```

### Question list

* `1448. Count Good Nodes in Binary Tree`
* `1026. Maximum Difference Between Node and Ancestor`
* `2458. Height of Binary Tree After Subtree Removal Queries`

### 1448. Count Good Nodes in Binary Tree

```python
class Solution:
    def goodNodes(self, root: TreeNode, mx=-inf) -> int:
        if not root:
            return 0
        l = self.goodNodes(root.left, max(mx, root.val))
        r = self.goodNodes(root.right, max(mx, root.val))
        return l + r + (root.val >= mx)
```

### 1026. Maximum Difference Between Node and Ancestor

```python
```

### 2458. Height of Binary Tree After Subtree Removal Queries

```python
class Solution:
    def treeQueries(self, root: Optional[TreeNode], queries: List[int]) -> List[int]:
        d = defaultdict(int)
        def dfs_height(node):
            if not node:
                return 0
            h = max(dfs_height(node.left), dfs_height(node.right)) + 1
            d[node] = h 
            return h 
        dfs_height(root)
        
        res = [0] * (len(d) + 1)
        def dfs(node, height, rest_h):
            if not node:
                return 
            height += 1
            res[node.val] = rest_h
            dfs(node.left, height, max(rest_h, height + d[node.right]))
            dfs(node.right, height, max(rest_h, height + d[node.left]))
        dfs(root, -1, 0)

        for i, q in enumerate(queries):
            queries[i] = res[q]
        return queries
```

### Question list: convert to BST

* `108. Convert Sorted Array to Binary Search Tree`
* `109. Convert Sorted List to Binary Search Tree`

### 108. Convert Sorted Array to Binary Search Tree

```python
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        n = len(nums)
        def dfs(l, r):
            if l > r:
                return None
            m = l + (r - l) // 2
            root = TreeNode(nums[m])
            root.left = dfs(l, m - 1)
            root.right = dfs(m + 1, r)
            return root
        return dfs(0, n - 1)
```

### 109. Convert Sorted List to Binary Search Tree

```python
class Solution:
    def sortedListToBST(self, head: Optional[ListNode]) -> Optional[TreeNode]:
        nums = []
        while head:
            nums.append(head.val)
            head = head.next
            
        n = len(nums)
        def dfs(l, r):
            if l > r:
                return None
            m = l + (r - l) // 2
            root = TreeNode(nums[m])
            root.left = dfs(l, m - 1)
            root.right = dfs(m + 1, r)
            return root
        return dfs(0, n - 1)
```

### 110. Balanced Binary Tree

```python
class Solution:
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        def depth(node):
            if not node:
                return 0
            return max(depth(node.left), depth(node.right)) + 1
        
        def isBalanced(node) :
            if not node:
                return True
            if abs(depth(node.left) - depth(node.right)) > 1:
                return False
            return isBalanced(node.left) and isBalanced(node.right)
        return isBalanced(root)
```

- O(n): directly using depth of binary tree

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

### 404. Sum of Left Leaves

```python
class Solution:
    def sumOfLeftLeaves(self, root: Optional[TreeNode]) -> int:
        def dfs(node):
            if node:
                if node.left and not node.left.left and not node.left.right:
                    self.res += node.left.val
                dfs(node.left)
                dfs(node.right)
        self.res = 0
        dfs(root)
        return self.res
```

### 129. Sum Root to Leaf Numbers

```python
class Solution:
    def sumNumbers(self, root: Optional[TreeNode]) -> int:
        def check(res):
            for i, a in enumerate(res):
                n = len(a)
                res[i] = sum(a[j] * 10 ** (n - j - 1) for j in range(n))
            return sum(res)
            
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

### 988. Smallest String Starting From Leaf

```python
class Solution:
    def smallestFromLeaf(self, root: Optional[TreeNode]) -> str:
        letters = ascii_lowercase
        def check(res):
            for i, a in enumerate(res):
                res[i] = ''.join([letters[i] for i in a])
            return sorted(res)[0]
            
        self.res = []
        def dfs(node, path):
            if node:
                if not node.left and not node.right:
                    self.res.append([node.val] + path)
                    return
                dfs(node.left, [node.val] + path)
                dfs(node.right, [node.val] + path)
        dfs(root, [])
        return check(self.res)
```

### 270. Closest Binary Search Tree Value

```python
class Solution:
    def closestValue(self, root: Optional[TreeNode], target: float) -> int:
        self.res = inf
        def dfs(node):
            if node:
                if abs(node.val - target) < abs(self.res - target):
                    self.res = node.val
                if abs(node.val - target) == abs(self.res - target):
                    self.res = min(self.res, node.val)
                dfs(node.left)
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

### 222. Count Complete Tree Nodes

```python
class Solution:
    def countNodes(self, root: Optional[TreeNode]) -> int:
        def depth(node):
            if not node:
                return 0
            return depth(node.left) + 1
        
        def dfs(node):
            if not node:
                return 0
            l, r = depth(node.left), depth(node.right)
            if l == r:
                print(l)
                return (1 << l) + dfs(node.right)
            else:
                return (1 << r) + dfs(node.left)
        return dfs(root)
```

### 226. Invert Binary Tree

```python
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root:
            return None
        root.left, root.right = self.invertTree(root.right), self.invertTree(root.left)
        return root
```

### 114. Flatten Binary Tree to Linked List

```python
class Solution:
    def flatten(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        def dfs(node, res):
            if node:
                res.append(node)
                dfs(node.left, res)
                dfs(node.right, res)
            return res
        res = dfs(root, [])
        if not res:
            return []
        temp = R = res[0]
        for i in range(1, len(res)):
            R.left = None
            R.right = res[i]
            R = R.right
        return temp
```

### 116. Populating Next Right Pointers in Each Node

```python
class Solution:
    def connect(self, root: 'Optional[Node]') -> 'Optional[Node]':
        if not root:
            return None
        q = deque([root])
        res = []
        while q:
            level = []
            for i in range(len(q)):
                node = q.popleft()
                level.append(node)
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
            for i in range(len(level) - 1):
                level[i].next = level[i + 1]
        return root
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

### 298. Binary Tree Longest Consecutive Sequence

```python
class Solution:
    def longestConsecutive(self, root: Optional[TreeNode]) -> int:
        def longest(nums):
            n = len(nums)
            dp = [1] * n
            for i in range(1, n):
                if nums[i] == nums[i - 1] + 1:
                    dp[i] += dp[i - 1]
            return max(dp)
        def check(res):
            ans = 0
            for a in res:
                ans = max(ans, longest(a))
            return ans
            
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
        print(res)
        return [v for v in res.values()]
```

### 450. Delete Node in a BST

```python
class Solution:
    def deleteNode(self, root: Optional[TreeNode], key: int) -> Optional[TreeNode]:
        if not root:
            return None
        elif root.val > key:
            root.left = self.deleteNode(root.left, key)
        elif root.val < key:
            root.right = self.deleteNode(root.right, key)
        else:
            if not root.left and not root.right:
                return None
            elif not root.left and root.right:
                return root.right
            elif not root.right and root.left:
                return root.left
            temp = root.right
            while temp.left:
                temp = temp.left
            root.val = temp.val
            root.right = self.deleteNode(root.right, root.val)
        return root
```

### design

### 173. Binary Search Tree Iterator

```python
class BSTIterator:

    def __init__(self, root: Optional[TreeNode]):
        self.q = deque()
        def dfs(node):
            if node:
                dfs(node.left)
                self.q.append(node)
                dfs(node.right)
        dfs(root)

    def next(self) -> int:
        return self.q.popleft().val

    def hasNext(self) -> bool:
        return len(self.q) > 0
```

### stack

### 331. Verify Preorder Serialization of a Binary Tree

```python
class Solution:
    def isValidSerialization(self, preorder: str) -> bool:
        stack = []
        for c in preorder.split(','):
            stack.append(c)
            while len(stack) >= 3 and stack[-1] == '#' and stack[-2] == '#' and stack[-3] != '#':
                for i in range(3):
                    stack.pop()
                stack.append('#')
        return stack == ['#']
```

### design

### 341. Flatten Nested List Iterator

```python
class NestedIterator:
    def __init__(self, nestedList: [NestedInteger]):
        self.q = deque()
        def dfs(nestedList):
            for item in nestedList:
                if item.isInteger():
                    self.q.append(item.getInteger())
                else:
                    dfs(item.getList())
        dfs(nestedList)
    
    def next(self) -> int:
        return self.q.popleft()
    
    def hasNext(self) -> bool:
         return len(self.q) > 0
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

### 559. Maximum Depth of N-ary Tree

```python
class Solution:
    def maxDepth(self, root: 'Node') -> int:
        if not root:
            return 0
        res = 0
        for child in root.children:
            res = max(res, self.maxDepth(child))
        return res + 1
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
        return dfs(root) if dfs(root) else False
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

### 671. Second Minimum Node In a Binary Tree

```python
class Solution:
    def findSecondMinimumValue(self, root: Optional[TreeNode]) -> int:
        s = set()
        def dfs(node):
            if node:
                s.add(node.val)
                dfs(node.left)
                dfs(node.right)
        dfs(root)
        res = sorted(list(s))
        return res[1] if len(res) > 1 else -1
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
                    return
        dfs(root)
        return self.res if self.res else None
```

### 701. Insert into a Binary Search Tree

```python
class Solution:
    def insertIntoBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        def dfs(node):
            if node:
                if not node.left and not node.right:
                    if node.val > val:
                        node.left = TreeNode(val)
                    else:
                        node.right = TreeNode(val)
                    return
                elif node.left and not node.right and val > node.val:
                    node.right = TreeNode(val)
                    return
                elif node.right and not node.left and val < node.val:
                    node.left = TreeNode(val)
                    return

                if node.val > val:
                    dfs(node.left)
                elif node.val < val:
                    dfs(node.right)
        if not root:
            return TreeNode(val)
        dfs(root)
        return root
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

### 703. Kth Largest Element in a Stream

```python
from sortedcontainers import SortedList
class KthLargest:

    def __init__(self, k: int, nums: List[int]):
        self.nums = nums
        self.k = k
        self.sl = SortedList()
        for n in self.nums:
            self.sl.add(n)

    def add(self, val: int) -> int:
        self.sl.add(val)
        return self.sl[len(self.sl) - self.k]
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

### 872. Leaf-Similar Trees

- post order

```python
class Solution:
    def leafSimilar(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> bool:
        def dfs(node, res):
            if node:
                dfs(node.left, res)
                dfs(node.right, res)
                if not node.left and not node.right:
                    res.append(node.val)
            return res
        res1 = dfs(root1, [])
        res2 = dfs(root2, [])
        return res1 == res2
```

### 112. Path Sum

```python
class Solution:
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        def dfs(node, presum):
            if not node:
                return False
            if not node.left and not node.right:
                return presum + node.val == targetSum
            return dfs(node.right, presum + node.val) or dfs(node.left, presum + node.val)
        return dfs(root, 0)
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

### 1022. Sum of Root To Leaf Binary Numbers

- path sum

```python
class Solution:
    def sumRootToLeaf(self, root: Optional[TreeNode]) -> int:
        def check(a):
            return int(''.join([str(i) for i in a]), 2)

        res = []
        def dfs(node, path, presum):
            if not node:
                return []
            if not node.left and not node.right:
                res.append(path + [node.val])
            dfs(node.right, path + [node.val], presum + node.val)
            dfs(node.left, path + [node.val], presum + node.val)
            return res
        arr = dfs(root, [], 0)
        ans = 0
        for a in arr:
            ans += check(a)
        return ans
```

### 572. Subtree of Another Tree

```python
class Solution:
    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        def sameTree(node1, node2):
            if not node1 and not node2:
                return True
            if node1 and not node2 or node2 and not node1:
                return False
            if node1.val != node2.val:
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

### 2689. Extract Kth Character From The Rope Tree

```python
class Solution:
    def getKthCharacter(self, root: Optional[object], k: int) -> str:
        """
        :type root: Optional[RopeTreeNode]
        """
        self.res = ''
        def dfs(node):
            if not node:
                return ''
            if not node.left and not node.right:
                self.res += node.val
            dfs(node.left)
            dfs(node.right)
        dfs(root)
        return self.res[k - 1]
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

### 654. Maximum Binary Tree

```python
class Solution:
    def constructMaximumBinaryTree(self, nums: List[int]) -> Optional[TreeNode]:
        def dfs(nums):
            if not nums:
                return 
            mx = max(nums)
            root = TreeNode(mx)
            root.left = dfs(nums[: nums.index(mx)])
            root.right = dfs(nums[nums.index(mx) + 1:])
            return root
        return dfs(nums)
```

### 655. Print Binary Tree

```python
class Solution:
    def printTree(self, root: Optional[TreeNode]) -> List[List[str]]:
        def depth(node):
            if not node:
                return -1
            return max(depth(node.left), depth(node.right)) + 1
        d = depth(root)
        R = d + 1
        C = 2 ** (d + 1) - 1
        grid = [[''] * C for r in range(R)]
        q = deque([(root, 0, (C - 1) // 2)])
        while q:
            node, r, c = q.popleft()
            grid[r][c] = str(node.val)
            if node.left:
                q.append((node.left, r + 1, c - int(2 ** (d - 1 - r))))
            if node.right:
                q.append((node.right, r + 1, c + int(2 ** (d - 1 - r))))
        return grid
```

### 1443. Minimum Time to Collect All Apples in a Tree

- tree dp

```python
class Solution:
    def minTime(self, n: int, edges: List[List[int]], hasApple: List[bool]) -> int:
        g = defaultdict(list)
        for u, v in edges:
            g[u].append(v)
            g[v].append(u)

        def dfs(x, pa):
            res = 0
            for y in g[x]:
                if y != pa:
                    time = dfs(y, x)
                    if time:
                        res += time + 2
                    elif hasApple[y]:
                        res += 2
            return res
        return dfs(0, -1)
```

### 1026. Maximum Difference Between Node and Ancestor

```python
class Solution:
    def maxAncestorDiff(self, root: Optional[TreeNode]) -> int:
        self.res = 0
        def dfs(root):
            if not root:
                return inf, -inf
            mn = mx = root.val
            min_l, max_l = dfs(root.left)
            min_r, max_r = dfs(root.right)
            mn = min(mn, min_l, min_r)
            mx = max(mx, max_l, max_r)
            self.res = max(self.res, mx - root.val, root.val - mn)
            return mn, mx
        dfs(root)
        return self.res
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

### 1008. Construct Binary Search Tree from Preorder Traversal

```python
class Solution:
    def bstFromPreorder(self, preorder: List[int]) -> Optional[TreeNode]:
        def check(nums):
            for i in range(1, len(nums) - 1):
                if nums[i] <= nums[0] <= nums[i + 1]:
                    return i + 1
            if nums[0] == min(nums):
                return 1
            return len(nums)

        def dfs(nums):
            if not nums:
                return None
            root = TreeNode(nums[0])
            i = check(nums)
            left = dfs(nums[1: i])
            right = dfs(nums[i:])
            root.left = left
            root.right = right
            return root
        return dfs(preorder)
```

### 236. Lowest Common Ancestor of a Binary Tree

```python
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        def dfs(node):
            if not node:
                return None
            if node == p or node == q:
                return node
            left = dfs(node.left)
            right = dfs(node.right)
            if left and right:
                return node
            elif left:
                return left
            elif right:
                return right
        return dfs(root)
```

### 1644. Lowest Common Ancestor of a Binary Tree II

```python
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        def dfs(node):
            if not node:
                return None
            if node == p or node == q:
                return node
            left = dfs(node.left)
            right = dfs(node.right)
            if left and right:
                return node
            elif left:
                return left
            elif right:
                return right
            return None
        
        self.hasP, self.hasQ = False, False
        def hasNode(root, p, q):
            if root:
                if root == p:
                    self.hasP = True
                if root == q:
                    self.hasQ = True
                hasNode(root.left, p, q)
                hasNode(root.right, p, q)
            return self.hasP and self.hasQ
        hasNode = hasNode(root, p, q)
        if not hasNode:
            return None
        return dfs(root)
```

### 1650. Lowest Common Ancestor of a Binary Tree III

```python
class Solution:
    def lowestCommonAncestor(self, p: 'Node', q: 'Node') -> 'Node':
        pSet, qSet = set(), set()
        while p or q:
            if pSet & qSet:
                return (pSet & qSet).pop()
            if p: pSet.add(p)
            if q: qSet.add(q)
            if p.parent: p = p.parent
            if q.parent: q = q.parent
```

### 1382. Balance a Binary Search Tree

```python
class Solution:
    def balanceBST(self, root: TreeNode) -> TreeNode:
        def dfs(node):
            if node:
                dfs(node.left)
                res.append(node.val)
                dfs(node.right)
        
        def build(l, r):
            m = l + (r - l) // 2
            root = TreeNode(res[m])
            if l <= m - 1:
                root.left = build(l, m - 1)
            if m + 1 <= r:
                root.right = build(m + 1, r)
            return root
        res = []
        dfs(root)
        return build(0, len(res) - 1)
```

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