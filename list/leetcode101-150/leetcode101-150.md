
### 101. Symmetric Tree

- tree

```python
class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        q = deque([root, root])
        while q:
            t1, t2 = q.popleft(), q.popleft()
            if not t1 and not t2:
                continue
            if not t1 or not t2 or t1.val != t2.val:
                return False
            q.append(t1.right)
            q.append(t2.left)
            q.append(t1.left)
            q.append(t2.right)
        return True
```

### 102. Binary Tree Level Order Traversal

```python
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root: return []
        queue, res = deque([root]), []

        while queue:
            level = []
            for i in range(len(queue)): # level count
                node = queue.popleft()
                level.append(node.val)
                if node.left:  queue.append(node.left)
                if node.right: queue.append(node.right)
            res.append(level)
        return res
```

### 103. Binary Tree Zigzag Level Order Traversal

```python
class Solution:
    def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root: return []
        q, res, even_level = deque([root]), [], False
        while q:
            level = []
            for i in range(len(q)):
                node = q.popleft()
                level.append(node.val)
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
            res.append(level[::-1]) if even_level else res.append(level)
            even_level = not even_level
        return res
```

### 104. Maximum Depth of Binary Tree

```python
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1
```

### 105. Construct Binary Tree from Preorder and Inorder Traversal

```python
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        if not preorder: 
            return None
        root = TreeNode(preorder[0])
        mid = inorder.index(preorder[0])
        root.left = self.buildTree(preorder[1: mid + 1], inorder[: mid])
        root.right = self.buildTree(preorder[mid + 1: ], inorder[mid + 1:])
        return root
```

### 106. Construct Binary Tree from Inorder and Postorder Traversal

```python
class Solution:
    def buildTree(self, inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        if not inorder: return None
        root = TreeNode(postorder[-1])
        mid = inorder.index(postorder[-1])
        root.left = self.buildTree(inorder[:mid], postorder[:mid])
        root.right = self.buildTree(inorder[mid + 1: ], postorder[mid: -1])
        return root
```

### 107. Binary Tree Level Order Traversal II

```python
class Solution:
    def levelOrderBottom(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root: return []
        queue, res = deque([root]), []
        while queue:
            level = []
            for i in range(len(queue)): # level count
                node = queue.popleft()
                level.append(node.val)
                if node.left:  queue.append(node.left)
                if node.right: queue.append(node.right)
            res.append(level)
        res.reverse()
        return res
```

### 108. Convert Sorted Array to Binary Search Tree

```python
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        if not nums:
            return None
        m = (len(nums) - 1) // 2
        root = TreeNode(nums[m])
        root.left = self.sortedArrayToBST(nums[: m])
        root.right = self.sortedArrayToBST(nums[m + 1: ])
        return root
```

### 109. Convert Sorted List to Binary Search Tree

```python
class Solution:
    def sortedListToBST(self, head: Optional[ListNode]) -> Optional[TreeNode]:
        nums = []
        while head:
            nums.append(head.val)
            head = head.next
        def sortedArrayToBST(nums):
            if not nums:
                return None
            m = (len(nums) - 1) // 2
            root = TreeNode(nums[m])
            root.left = sortedArrayToBST(nums[: m])
            root.right = sortedArrayToBST(nums[m + 1: ])
            return root
        return sortedArrayToBST(nums)
```

### 110. Balanced Binary Tree

```python
class Solution:
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        if not root:
            return 1
        left = self.isBalanced(root.left)
        right = self.isBalanced(root.right)
        if not left or not right or abs(left - right) > 1:
            return False
        return 1 + max(left, right)
```

### 111. Minimum Depth of Binary Tree

```python
class Solution:
    def minDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        q, res = deque([root]), 1
        while q:
            for i in range(len(q)):
                node = q.popleft()
                if not node.left and not node.right:
                    return res
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
            res += 1
```

### 112. Path Sum

```python
class Solution:
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        if not root:
            return False
        q = deque([(root, 0)])
        while q:
            node, value = q.popleft()
            if not node.left and not node.right and targetSum == value + node.val:
                return True
            if node.left:
                q.append((node.left, value + node.val))
            if node.right:
                q.append((node.right, value + node.val))
        return False
```

### 113. Path Sum II

```python
class Solution:
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> List[List[int]]:
        if not root:
            return []
        q, res = deque([(root, 0, [])]), []
        while q:
            node, value, ans = q.popleft()
            if not node.left and not node.right and targetSum == value + node.val:
                res.append(ans[::] + [node.val])
            if node.left:
                q.append((node.left,value + node.val, ans + [node.val]))
            if node.right:
                q.append((node.right,value + node.val, ans + [node.val]))
        return res
```

### 118. Pascal's Triangle

```python
class Solution:
    def generate(self, numRows: int) -> List[List[int]]:
        res = [[1]]
        for i in range(2, numRows + 1):
            addZero = [0] + res[-1] + [0]
            dp = [addZero[i - 1] + addZero[i] for i in range(1, len(addZero))]
            res.append(dp)
        return res
```

### 119. Pascal's Triangle II

```python
class Solution:
    def getRow(self, rowIndex: int) -> List[int]:
        dp = [1]
        for i in range(1, rowIndex + 1):
            addZero = [0] + dp + [0]
            dp = [addZero[i - 1] + addZero[i] for i in range(1, len(addZero))]
        return dp
```

### 121. Best Time to Buy and Sell Stock

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        lowest, profit = prices[0], 0
        for price in prices:
            lowest = min(lowest, price)
            profit = max(profit, price - lowest)
        return profit
```

### 131. Palindrome Partitioning

```python
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        def isPanlindrome(s):
            return s == ''.join(reversed(list(s)))

        res, ans, n = [], [], len(s)
        def backtrack(i):
            if i >= n:
                res.append(ans[::])
                return
            for j in range(i, n):
                if isPanlindrome(s[i: j + 1]):
                    ans.append(s[i: j + 1])
                    backtrack(j + 1)
                    ans.pop()
        backtrack(0)
        return res
```

### 132. Palindrome Partitioning II

```python
class Solution:
    def minCut(self, s: str) -> int:
        g, n = defaultdict(set), len(s)
        def helper(l, r):
            while l >= 0 and r < n and s[l] == s[r]:
                g[l].add(r)
                l -= 1
                r += 1
        for i in range(n):
            helper(i, i)
            helper(i, i + 1)
        
        @lru_cache(None)
        def dfs(i):
            if i >= n:
                return 0
            ans = n
            for j in range(i, n):
                if j in g[i]:
                    ans = min(ans, dfs(j + 1) + 1)
            return ans
        return dfs(0) - 1
```

### 139. Word Break

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        wordDict = set(wordDict)
        n = len(s)
        @cache
        def dfs(i):
            if i == n:
                return True 
            res = False
            for j in range(n):
                if s[i: j + 1] in wordDict:
                    res = res or dfs(j + 1)
            return res 
        return dfs(0)
```

### 140. Word Break II

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> List[str]:
        res, wordDict = [], set(wordDict)
        def dfs(i, ans):
            if i == len(s):
                res.append(' '.join(ans))
                return
            for j in range(i + 1, len(s) + 1):
                if s[i: j] in wordDict:
                    dfs(j, ans + [s[i: j]])     
        dfs(0, [])
        return res
```

### 141. Linked List Cycle

```python
class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return True
        return False
```

### 142. Linked List Cycle II

```python
# the first comment in the below link
# https://leetcode.com/problems/linked-list-cycle-ii/discuss/44822/Java-two-pointer-solution.
class Solution:
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        class Solution:
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                slow = head
                while True:
                    if slow == fast:
                        return slow
                    slow = slow.next
                    fast = fast.next
        return None
```

### 144. Binary Tree Preorder Traversal

```python
# recursive
class Solution:
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        def dfs(node, arr):
            if not node:
                return node
            arr.append(node.val)
            dfs(node.left, arr)
            dfs(node.right, arr)
            return arr
        return dfs(root, [])
# iterative
class Solution:
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        res, stack = [], [root]
        while stack:
            node = stack.pop()
            if node:
                res.append(node.val)
                stack.extend([node.right, node.left])
        return res
```

### 145. Binary Tree Postorder Traversal

```python
# recursive
class Solution:
    def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        def dfs(node, arr):
            if not node:
                return node
            dfs(node.left, arr)
            dfs(node.right, arr)
            arr.append(node.val)
            return arr
        return dfs(root, [])
# iterative
class Solution:
    def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        # modified preorder
        res, stack = [], [root]
        while stack:
            node = stack.pop()
            if node:
                res.append(node.val)
                stack.extend([node.left, node.right])
        return res[::-1]
```

### 149. Max Points on a Line

- D & C

The question asked to use matrix

```python
class Solution:
    def maxPoints(self, points: List[List[int]]) -> int:
        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a

        def calcSlope(a, b): # avoid 0 of dx or dy
            dx = a[0] - b[0]
            dy = a[1] - b[1]
            divisor = gcd(dx, dy)
            return (dx / divisor, dy / divisor) # use tuple as key

        if len(points) <= 2:
            return len(points)
        res = 0
        for i in range(len(points)):
            slopes, dups = {}, 1 # [0, 0], [0, 0] or [0, 0], [1, 1], [2, 2]
            for j in range(i + 1, len(points)):
                if points[i] == points[j]:
                    dups += 1
                else:
                    slope = calcSlope(points[i], points[j])
                    if slope in slopes:
                        slopes[slope] += 1
                    else:
                        slopes[slope] = 1
            for slope in slopes:
                res = max(res, slopes[slope] + dups)
        return res
```
