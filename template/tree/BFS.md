### BFS(30)

* [102. Binary Tree Level Order Traversal](#102-binary-tree-level-order-traversal)
* [103. Binary Tree Zigzag Level Order Traversal](#103-binary-tree-zigzag-level-order-traversal)
* [107. Binary Tree Level Order Traversal II](#107-binary-tree-level-order-traversal-ii)
* [199. Binary Tree Right Side View](#199-binary-tree-right-side-view)
* [513. Find Bottom Left Tree Value](#513-find-bottom-left-tree-value)

* [515. Find Largest Value in Each Tree Row](#515-find-largest-value-in-each-tree-row)
* [637. Average of Levels in Binary Tree](#637-average-of-levels-in-binary-tree)
* [1161. Maximum Level Sum of a Binary Tree](#1161-maximum-level-sum-of-a-binary-tree)
* [993. Cousins in Binary Tree](#993-cousins-in-binary-tree)
* [2583. Kth Largest Sum in a Binary Tree](#2583-kth-largest-sum-in-a-binary-tree)

* [1302. Deepest Leaves Sum](#1302-deepest-leaves-sum)
* [2415. Reverse Odd Levels of Binary Tree](#2415-reverse-odd-levels-of-binary-tree)
* [1609. Even Odd Tree](#1609-even-odd-tree)
* [623. Add One Row to Tree](#623-add-one-row-to-tree)
* [662. Maximum Width of Binary Tree](#662-maximum-width-of-binary-tree)

* [2471. Minimum Number of Operations to Sort a Binary Tree by Level](#2471-minimum-number-of-operations-to-sort-a-binary-tree-by-level)
* [1602. Find Nearest Right Node in Binary Tree](#1602-find-nearest-right-node-in-binary-tree)
* [3157. Find the Level of Tree with Minimum Sum](#3157-find-the-level-of-tree-with-minimum-sum)
* [742. Closest Leaf in a Binary Tree](#742-closest-leaf-in-a-binary-tree)
* [863. All Nodes Distance K in Binary Tree](#863-all-nodes-distance-k-in-binary-tree)

* [1660. Correct a Binary Tree](#1660-correct-a-binary-tree)
* [2641. Cousins in Binary Tree II](#2641-cousins-in-binary-tree-ii)
* [919. Complete Binary Tree Inserter](#919-complete-binary-tree-inserter)
* [958. Check Completeness of a Binary Tree](#958-check-completeness-of-a-binary-tree)
* [331. Verify Preorder Serialization of a Binary Tree](#331-verify-preorder-serialization-of-a-binary-tree)

- need to check

* [117. Populating Next Right Pointers in Each Node II](#117-populating-next-right-pointers-in-each-node-ii)
* [314. Binary Tree Vertical Order Traversal](#314-binary-tree-vertical-order-traversal)
* [1469. Find All The Lonely Nodes](#1469-find-all-the-lonely-nodes)
* [987. Vertical Order Traversal of a Binary Tree](#987-vertical-order-traversal-of-a-binary-tree)
* [2368. Reachable Nodes With Restrictions](#2368-reachable-nodes-with-restrictions)

### 102. Binary Tree Level Order Traversal

```python
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        q = deque([root])
        res = []
        while q:
            level = []
            for _ in range(len(q)):
                node = q.popleft()
                level.append(node.val)
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
            res.append(level)
        return res
```

### 103. Binary Tree Zigzag Level Order Traversal

- same as 102

```python
class Solution:
    def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        q = deque([root])
        res = []
        cnt = 0
        while q:
            level = []
            cnt += 1
            for _ in range(len(q)):
                node = q.popleft()
                level.append(node.val)
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
            if cnt % 2 == 1:
                res.append(level)
            else:
                res.append(level[::-1])
        return res
```

### 107. Binary Tree Level Order Traversal II

- same as 102

```python
class Solution:
    def levelOrderBottom(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        q = deque([root])
        res = []
        while q:
            level = []
            for _ in range(len(q)):
                node = q.popleft()
                level.append(node.val)
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
            res.append(level)
        return res[::-1]
```

### 199. Binary Tree Right Side View

- same as 102

```python
class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []
        q = deque([root])
        res = []
        while q:
            n = len(q)
            for i in range(n):
                node = q.popleft()
                if i == n - 1:
                    res.append(node.val)
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
        return res
```

### 513. Find Bottom Left Tree Value

- same as 102

```python
class Solution:
    def findBottomLeftValue(self, root: Optional[TreeNode]) -> int:
        q = deque([root])
        res = 0
        while q:
            for i in range(len(q)):
                node = q.popleft()
                if i == 0:
                    res = node.val
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
        return res
```

### 515. Find Largest Value in Each Tree Row

```python
class Solution:
    def largestValues(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []
        q, res = deque([root]), []
        while q:
            ans = -inf
            for i in range(len(q)):
                node = q.popleft()
                ans = max(ans, node.val)
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
            res.append(ans)
        return res
```

### 637. Average of Levels in Binary Tree

```python
class Solution:
    def averageOfLevels(self, root: Optional[TreeNode]) -> List[float]:
        if not root:
            return []
        q = deque([root])
        res = []
        while q:
            level = []
            for _ in range(len(q)):
                node = q.popleft()
                level.append(node.val)
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
            res.append(sum(level) / len(level))
        return res
```

### 1161. Maximum Level Sum of a Binary Tree

```python
class Solution:
    def maxLevelSum(self, root: Optional[TreeNode]) -> int:
        q = deque([root])
        res = -inf
        level, mx_level = 0, 0
        while q:
            level += 1
            ans = 0
            for i in range(len(q)):
                node = q.popleft()
                ans += node.val
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
            if ans > res:
                res = ans 
                mx_level = level
        return mx_level
```

### 993. Cousins in Binary Tree

```python
class Solution:
    def isCousins(self, root: Optional[TreeNode], x: int, y: int) -> bool:
        q = deque([(root, 0, -1)]) # root, depth, parent
        d = defaultdict(list)
        while q:
            node, depth, parent = q.popleft()
            d[node.val].extend([depth, parent])
            if node.left:
                q.append((node.left, depth + 1, node))
            if node.right:
                q.append((node.right, depth + 1, node))
        return d[x][0] == d[y][0] and d[x][1] != d[y][1]
```

### 2583. Kth Largest Sum in a Binary Tree

```python
class Solution:
    def kthLargestLevelSum(self, root: Optional[TreeNode], k: int) -> int:
        q = deque([root])
        res = []
        while q:
            level = []
            for _ in range(len(q)):
                node = q.popleft()
                level.append(node.val)
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
            res.append(sum(level))
        res.sort(reverse = True)
        return res[k - 1] if k - 1 < len(res) else -1
```

### 1302. Deepest Leaves Sum

```python
class Solution:
    def deepestLeavesSum(self, root: Optional[TreeNode]) -> int:
        q = deque([root])
        res = []
        while q:
            level = []
            for _ in range(len(q)):
                node = q.popleft()
                level.append(node.val)
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
            res.append(level)
        return sum(res[-1])
```

### 2415. Reverse Odd Levels of Binary Tree

```python
class Solution:
    def reverseOddLevels(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        def check(level):
            for node, val in zip(level, [node.val for node in level][::-1]):
                node.val = val 

        q = deque([root])
        res = []
        cnt = -1
        while q:
            level = []
            cnt += 1
            for _ in range(len(q)):
                node = q.popleft()
                level.append(node)
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
            if cnt % 2 == 1:
                check(level)
        return root
```

### 1609. Even Odd Tree

```python
class Solution:
    def isEvenOddTree(self, root: Optional[TreeNode]) -> bool:
        q, even = deque([root]), True
        while q:
            level = []
            for i in range(len(q)):
                node = q.popleft()
                level.append(node.val)
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
            if even and (any(n % 2 == 0 for n in level) or level != sorted(level)):
                return False
            if not even and (any(n % 2 == 1 for n in level) or level != sorted(level, reverse = True)):
                return False
            if len(level) != len(set(level)):
                return False
            even = not even
        return True
```


### 623. Add One Row to Tree

```python
class Solution:
    def addOneRow(self, root: Optional[TreeNode], val: int, depth: int) -> Optional[TreeNode]:
        if depth == 1:
            return TreeNode(val, root, None)
        level = [root]
        for i in range(2, depth):
            temp = []
            for node in level:
                if node.left:
                    temp.append(node.left)
                if node.right:
                    temp.append(node.right)
            level = temp
        for node in level:
            node.left = TreeNode(val, node.left, None)
            node.right = TreeNode(val, None, node.right)
        return root
```

### 662. Maximum Width of Binary Tree

```python
class Solution:
    def widthOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        q = deque([(root, 1)]) # node, index
        res = 1
        while q:
            res = max(res,  q[-1][1] - q[0][1] + 1)
            for _ in range(len(q)):
                node, index = q.popleft()
                if node.left:
                    q.append((node.left, 2 * index))
                if node.right:
                    q.append((node.right, 2 * index + 1))
        return res
```

### 2471. Minimum Number of Operations to Sort a Binary Tree by Level

```python
class Solution:
    def minimumOperations(self, root: Optional[TreeNode]) -> int:
        def check(a):
            a = sorted(range(len(a)), key = lambda i: a[i])
            res = 0
            visited = set()
            for n in a:
                cnt = 0
                while n not in visited:
                    visited.add(n)
                    n = a[n]
                    cnt += 1
                if cnt >= 1:
                    res += cnt - 1
            return res 
            
        q = deque([root])
        res = 0
        while q:
            level = []
            for _ in range(len(q)):
                node = q.popleft()
                level.append(node.val)
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
            res += check(level)
        return res
```

### 1602. Find Nearest Right Node in Binary Tree

```python
class Solution:
    def findNearestRightNode(self, root: TreeNode, u: TreeNode) -> Optional[TreeNode]:
        q = deque([root])
        res = []
        while q:
            level = []
            for _ in range(len(q)):
                node = q.popleft()
                level.append(node)
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
            res.append(level)
        
        for a in res:
            for i, node in enumerate(a):
                if node == u and i != len(a) - 1:
                    return a[i + 1]
        return None
```

### 3157. Find the Level of Tree with Minimum Sum

```python
class Solution:
    def minimumLevel(self, root: Optional[TreeNode]) -> int:
        q = deque([root])
        res = []
        while q:
            total = 0
            for _ in range(len(q)):
                node = q.popleft()
                total += node.val
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
            res.append(total)
        mn = min(res)
        for i, n in enumerate(res):
            if n == mn:
                return i + 1
```

### 742. Closest Leaf in a Binary Tree

```python
class Solution:
    def findClosestLeaf(self, root: Optional[TreeNode], k: int) -> int:
        g = defaultdict(list)
        q = deque([root])
        while q:
            node = q.popleft()
            if node.val == k:
                start = node 
            if node.left:
                g[node].append(node.left)
                g[node.left].append(node)
                q.append(node.left)
            if node.right:
                g[node].append(node.right)
                g[node.right].append(node)
                q.append(node.right)
        
        q2 = deque([start])
        visited = set([start])
        while q2:
            node = q2.popleft()
            if not node.left and not node.right:
                return node.val 
            for nei in g[node]:
                if nei not in visited:
                    visited.add(nei)
                    q2.append(nei)
```

### 863. All Nodes Distance K in Binary Tree

```python
class Solution:
    def distanceK(self, root: TreeNode, target: TreeNode, k: int) -> List[int]:
        g = defaultdict(list)
        q = deque([root])
        while q:
            node = q.popleft()
            if node.left:
                q.append(node.left)
                g[node.val].append(node.left.val)
                g[node.left.val].append(node.val)
            if node.right:
                q.append(node.right)
                g[node.val].append(node.right.val)
                g[node.right.val].append(node.val)
        
        q = deque([(target.val, 0)])
        visited = set([target.val])
        res = []
        while q:
            node, d = q.popleft()
            if d == k:
                res.append(node)
                continue
            for nei in g[node]:
                if nei not in visited:
                    visited.add(nei)
                    q.append((nei, d + 1))
        return res
```

### 1660. Correct a Binary Tree

```python
class Solution:
    def correctBinaryTree(self, root: TreeNode) -> TreeNode:
        q = deque([(root, None, 1)])
        s = set([root])
        while q:
            for _ in range(len(q)):
                node, parent, depth = q.popleft()
                if node.left:
                    q.append((node.left, node, depth + 1))
                    s.add(node.left)
                    
                if node.right:
                    if node.right in s:
                        if node == parent.left:
                            parent.left = None
                        else:
                            parent.right = None
                        break
                    q.append((node.right, node, depth + 1))
                    s.add(node.right)
        return root 
```

### 2641. Cousins in Binary Tree II

```python
class Solution:
    def replaceValueInTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        q = deque([root])
        d = defaultdict(list)
        res = []
        while q:
            level = []
            total = 0
            for _ in range(len(q)):
                node = q.popleft()
                total += node.val
                level.append(node)
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
                if node.left and node.right:
                    d[node.left].extend([node.right, node.right.val])
                    d[node.right].extend([node.left, node.left.val])
            res.append(total)

        q = deque([root])
        i = 0
        while q:
            for _ in range(len(q)):
                node = q.popleft()
                if node in d:
                    node.val = res[i] - node.val - d[node][1]
                else:
                    node.val = res[i] - node.val 
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
            i += 1
        return root
```

### 919. Complete Binary Tree Inserter

```python
class CBTInserter:

    def __init__(self, root: Optional[TreeNode]):
        self.root = root 

    def insert(self, val: int) -> int:
        q = deque([self.root])
        while q:
            node = q.popleft()
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
            if not node.left:
                node.left = TreeNode(val)
                break
            if not node.right:
                node.right = TreeNode(val)
                break
        return node.val 

    def get_root(self) -> Optional[TreeNode]:
        return self.root
```

### 958. Check Completeness of a Binary Tree

```python
class Solution:
    def isCompleteTree(self, root: Optional[TreeNode]) -> bool:
        q = deque([(root, 1)])
        res = []
        while q:
            for i in range(len(q)):
                node, idx = q.popleft()
                res.append(idx)
                if node.left:
                    q.append((node.left, idx * 2))
                if node.right:
                    q.append((node.right, idx * 2 + 1))
        return res[-1] == len(res)
```

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

### 117. Populating Next Right Pointers in Each Node II

```python
class Solution:
    def connect(self, root: 'Node') -> 'Node':
        if not root:
            return None
        res, q = [], deque([root])
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

### 1302. Deepest Leaves Sum

```python
from collections import deque
class Solution:
    def deepestLeavesSum(self, root: Optional[TreeNode]) -> int:
        q = deque([root])
        while q:
            total = 0
            for i in range(len(q)):
                node = q.popleft()
                total += node.val
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
        return total
```

### 2415. Reverse Odd Levels of Binary Tree

```python
class Solution:
    def reverseOddLevels(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        def check(res):
            val = [r.val for r in res]
            val.reverse()
            for node, val in zip(res, val):
                node.val = val

        q, odd = deque([root]), True
        while q:
            level = []
            odd = not odd
            for i in range(len(q)):
                node = q.popleft()
                level.append(node)
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
            if odd:
                check(level)
        return root
```

### 1609. Even Odd Tree

```python
class Solution:
    def isEvenOddTree(self, root: Optional[TreeNode]) -> bool:
        q, even = deque([root]), True
        while q:
            level = []
            for i in range(len(q)):
                node = q.popleft()
                level.append(node.val)
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
            if even and (any(n % 2 == 0 for n in level) or level != sorted(level)) :
                return False
            if not even and (any(n % 2 == 1 for n in level) or level != sorted(level, reverse = True)):
                return False
            if len(level) != len(set(level)):
                return False
            even = not even
        return True
```

### 314. Binary Tree Vertical Order Traversal

```python
class Solution:
    def verticalOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        d = defaultdict(list)
        q = deque([(root, 0)])
        while q:
            node, level = q.popleft()
            d[level].append(node.val)
            if node.left:
                q.append((node.left, level - 1))
            if node.right:
                q.append((node.right, level + 1))
        return [d[i] for i in sorted(d.keys())]
```

### 1469. Find All The Lonely Nodes

```python
class Solution:
    def getLonelyNodes(self, root: Optional[TreeNode]) -> List[int]:
        d = defaultdict(list) # parent:children
        q = deque([root])
        while q:
            node = q.popleft()
            if node.left:
                d[node].append(node.left.val)
                q.append(node.left)
            if node.right:
                d[node].append(node.right.val)
                q.append(node.right)
        res = []
        for k in d:
            if len(d[k]) == 1:
                res.append(d[k][0])
        return res
```

### 987. Vertical Order Traversal of a Binary Tree

```python
class Solution:
    def verticalTraversal(self, root: Optional[TreeNode]) -> List[List[int]]:
        d = defaultdict(list)
        q = deque([(root, 0, 0)]) # root, row, col
        while q:
            node, row, col = q.popleft()
            d[col].append((row, node.val))
            if node.left:
                q.append((node.left, row + 1, col - 1))
            if node.right:
                q.append((node.right, row + 1, col + 1))
        
        res = []
        for k in sorted(d.keys()):
            res.append([item[1] for item in sorted(d[k])])
        return res
```

### 2368. Reachable Nodes With Restrictions

```python
class Solution:
    def reachableNodes(self, n: int, edges: List[List[int]], restricted: List[int]) -> int:
        if 0 in restricted:
            return 0
        q = deque([0])
        visited = set([0])
        for r in restricted:
            visited.add(r)
        g = defaultdict(list)
        for u, v in edges:
            g[u].append(v)
            g[v].append(u)

        res = 0
        while q:
            node = q.popleft()
            res += 1
            for nei in g[node]:
                if nei not in visited:
                    visited.add(nei)
                    q.append(nei)
        return res
```