## template: 1302. Deepest Leaves Sum

```python
from collections import deque

def fn(root):
    q = deque([root])
    while queue:
        # some code
        # res = 0
        for _ in range(len(q)):
            node = queue.popleft()
            # some code
            # res += node.val
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
    return res
```

### Level order traversal

* `117. Populating Next Right Pointers in Each Node II`
* `1161. Maximum Level Sum of a Binary Tree`
* `1302. Deepest Leaves Sum`
* `2415. Reverse Odd Levels of Binary Tree`
* `314. Binary Tree Vertical Order Traversal`
* `429. N-ary Tree Level Order Traversal`

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

### 1161. Maximum Level Sum of a Binary Tree

```python
class Solution:
    def maxLevelSum(self, root: Optional[TreeNode]) -> int:
        q, res = deque([root]), []
        while q:
            level = []
            for i in range(len(q)):
                node = q.popleft()
                level.append(node.val)
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
            res.append(sum(level))
    
        maxNum = max(res)
        for i, n in enumerate(res):
            if n == maxNum:
                return i + 1
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

### 429. N-ary Tree Level Order Traversal

```python
class Solution:
    def levelOrder(self, root: 'Node') -> List[List[int]]:
        if not root:
            return []
        q = deque([root])
        res = []
        while q:
            level = []
            for i in range(len(q)):
                node = q.popleft()
                level.append(node.val)
                for child in node.children:
                    q.append(child)
            res.append(level)
        return res
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

### 386. Lexicographical Numbers

```python
class Solution:
    def lexicalOrder(self, n: int) -> List[int]:
        def dfs(cur, limit):
            if cur > limit:
                return
            res.append(cur)
            for i in range(10):
                dfs(cur * 10 + i, limit)

        res = []
        for i in range(1, 10):
            dfs(i, n)
        return res
```

```python
class Solution:
    def lexicalOrder(self, n: int) -> List[int]:
        res = []
        num = 1
        for i in range(n):
            res.append(num)
            if num * 10 <= n:
                num *= 10
            else:
                while num % 10 == 9 or num + 1 > n:
                    num //= 10
                num += 1
        return res
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

### 513. Find Bottom Left Tree Value

```python
class Solution:
    def findBottomLeftValue(self, root: Optional[TreeNode]) -> int:
        if not root:
            return []
        q, res = deque([root]), -inf
        while q:
            for i in range(len(q)):
                node = q.popleft()
                res = node.val
                if node.right:
                    q.append(node.right)
                if node.left:
                    q.append(node.left)
        return res
```

### 623. Add One Row to Tree

```python
class Solution:
    def addOneRow(self, root: Optional[TreeNode], val: int, depth: int) -> Optional[TreeNode]:
        if depth == 1:
            return TreeNode(val, root, None)
        curLevel = [root]
        for _ in range(2, depth):
            tmpt = []
            for node in curLevel:
                if node.left:
                    tmpt.append(node.left)
                if node.right:
                    tmpt.append(node.right)
            curLevel = tmpt
        for node in curLevel:
            node.left = TreeNode(val, node.left, None)
            node.right = TreeNode(val, None, node.right)
        return root
```

### 662. Maximum Width of Binary Tree

```python
class Solution:
    def widthOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        q, res = deque([(root, 1)]), 1
        while q:
            for i in range(len(q)):
                node, idx = q.popleft()
                if node.left:
                    q.append((node.left, 2 * idx))
                if node.right:
                    q.append((node.right, 2 * idx + 1))
            if q:
                res = max(res, q[-1][1] - q[0][1] + 1)
        return res
```

### 2583. Kth Largest Sum in a Binary Tree

```python
from sortedcontainers import SortedList
class Solution:
    def kthLargestLevelSum(self, root: Optional[TreeNode], k: int) -> int:
        sl = SortedList()
        q = deque([root])
        while q:
            ans = 0
            for i in range(len(q)):
                node = q.popleft()
                ans += node.val
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
            sl.add(ans)
        n = len(sl)
        return sl[-k] if k <= n else -1
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