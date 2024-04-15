## 1 Array & Hashing: 8

### 217. Contains Duplicate

- Hash Set

```python
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        s = set()
        for n in nums:
            if n in s:
                return True
            s.add(n)
        return False
```

### 242. Valid Anagram

- Hash Table

```python
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        return Counter(s) == Counter(t)
```

### 1. Two Sum

- Hash Table

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        d = {}
        for i, n in enumerate(nums):
            res = target - n 
            if res in d:
                return [d[res], i]
            d[n] = i 
```

### 49. Group Anagrams

- Hash Table

```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        d = defaultdict(list)
        for s in strs:
            d[''.join(sorted(list(s)))].append(s)
        return list(d.values())
```

### 347. Top K Frequent Elements

- heap

```python
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        pq = []
        for key, v in Counter(nums).items():
            heappush(pq, (-v, key))
        res = []
        for i in range(k):
            v, key = heappop(pq)
            res.append(key)
        return res
```

### 238. Product of Array Except Self

- prefix suffix sum

```python
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        n = len(nums)
        pre, suf = [nums[0]] * n, [nums[-1]] * n 
        for i in range(1, n):
            pre[i] = nums[i] * pre[i - 1]
        for i in range(n - 2, -1, -1):
            suf[i] = nums[i] * suf[i + 1]
        pre, suf = [1] + pre + [1], [1] + suf + [1]
        res = []
        for i in range(1, n + 1):
            res.append(pre[i - 1] * suf[i + 1])
        return res 
```

### 271. Encode and Decode Strings

- string

```python
class Codec:
    def encode(self, strs: List[str]) -> str:
        """Encodes a list of strings to a single string.
        """
        return chr(256).join(strs)

    def decode(self, s: str) -> List[str]:
        """Decodes a single string to a list of strings.
        """
        return s.split(chr(256))
```

### 128. Longest Consecutive Sequence

- union find 

```python
class UF:
    def __init__(self, nums):
        self.parent = {n: n for n in nums}
    
    def find(self, n):
        while n != self.parent[n]:
            self.parent[n] = self.parent[self.parent[n]]
            n = self.parent[n]
        return n 

    def union(self, n1, n2):
        p1, p2 = self.find(n1), self.find(n2)
        self.parent[n1] = n2 
        
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        uf = UF(nums)
        nums = set(nums)
        for n in nums:
            if n + 1 in nums:
                uf.union(n, n + 1)
        d = defaultdict(int)
        for n in uf.parent:
            d[uf.find(n)] += 1
        return max(d.values()) if d.values() else 0
```

## 2 Two Pointers: 3

### 125. Valid Palindrome

```python
class Solution:
    def isPalindrome(self, s: str) -> bool:
        res = ''
        for c in s:
            if c.isalnum():
                res += c.lower()
        return res == res[::-1]
```

### 15. 3Sum

```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        n = len(nums)
        res = set()
        for i in range(n):
            l, r = i + 1, n - 1
            while l < r:
                three = nums[i] + nums[l] + nums[r]
                if three == 0:
                    res.add((nums[i], nums[l], nums[r]))
                    l += 1
                    r -= 1
                elif three > 0:
                    r -= 1
                else:
                    l += 1
        return list(res)
```

### 11. Container With Most Water

```python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        l, r = 0, len(height) - 1
        res = 0
        while l < r:
            res = max(res, min(height[l], height[r]) * (r - l))
            if height[l] > height[r]:
                r -= 1
            else:
                l += 1
        return res 
```

## 3 Sliding Window: 4

### 121. Best Time to Buy and Sell Stock

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        profit, lowest = 0, prices[0]
        for p in prices:
            lowest = min(lowest, p)
            profit = max(profit, p - lowest)
        return profit
```

### 3. Longest Substring Without Repeating Characters

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        d = Counter()
        l, res = 0, 0
        for r, c in enumerate(s):
            d[c] += 1
            while d[c] > 1:
                d[s[l]] -= 1
                l += 1
            res = max(res, r - l + 1)
        return res 
```

### 424. Longest Repeating Character Replacement

```python
class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        d, maxFreq, res = Counter(), 0, 0
        l = 0
        for r, c in enumerate(s):
            d[c] += 1
            maxFreq = max(maxFreq, d[c])
            while r - l + 1 - maxFreq > k:
                d[s[l]] -= 1
                maxFreq = max(maxFreq, d[s[l]])
                l += 1
            res = max(res, r - l + 1)
        return res 
```

### 76. Minimum Window Substring

```python
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        d, d_t = Counter(), Counter(t)
        res = s + '#'
        l = 0
        for r, c in enumerate(s):
            d[c] += 1
            while d >= d_t:
                res = min(res, s[l: r + 1], key= len)
                d[s[l]] -= 1
                l += 1
        return res if res != s + '#' else ''
```

## 4 stack

### 20. Valid Parentheses

```python
class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        for c in s:
            if stack and ((stack[-1] == '(' and c == ')') or (stack[-1] == '[' and c == ']') or (stack[-1] == '{' and c == '}')):
                stack.pop()
            else:
                stack.append(c)
        return not stack
```

## 5 Binary Search

### 153. Find Minimum in Rotated Sorted Array

```python
class Solution:
    def findMin(self, nums: List[int]) -> int:
        l, r = 0, len(nums) - 1
        while l < r:
            m = l + (r - l) // 2
            if nums[m] > nums[r]:
                l = m + 1
            else:
                r = m 
        return nums[l]
```

### 33. Search in Rotated Sorted Array

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        l, r = 0, len(nums) - 1
        while l <= r:
            m = l + (r - l) // 2
            if nums[m] == target:
                return m 
            if nums[m] >= nums[l]:
                if nums[l] <= target < nums[m]:
                    r = m - 1
                else:
                    l = m + 1
            else:
                if nums[m] < target <= nums[r]:
                    l = m + 1
                else:
                    r = m - 1
        return -1
```

## 6 Linked List: 6

### 206. Reverse Linked List

```python
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev, cur = None, head 
        while cur:
            nxt = cur.next 
            cur.next = prev 
            prev, cur = cur, nxt 
        return prev 
```

### 21. Merge Two Sorted Lists

```python
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        p = dummy = ListNode()
        while list1 and list2:
            if list1.val < list2.val:
                p.next = ListNode(list1.val)
                list1 = list1.next
            else:
                p.next = ListNode(list2.val) 
                list2 = list2.next 
            p = p.next 
        p.next = list1 or list2 
        return dummy.next 
```

### 143. Reorder List

```python
class Solution:
    def reorderList(self, head: Optional[ListNode]) -> None:
        """
        Do not return anything, modify head in-place instead.
        """
        # find the mid node
        slow = fast = ListNode()
        slow.next = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next

        p = slow.next
        slow.next = None
        # reverse
        def reverseList(prev, cur):
            if not cur:
                return prev
            nxt = cur.next
            cur.next = prev
            return reverseList(cur, nxt)
        tail = reverseList(None, p)
        # merge
        l1, l2 = head, tail
        while l1 and l2:
            l1_temp, l2_temp = l1.next, l2.next 
            l1.next = l2 
            l2.next = l1_temp 
            l1, l2 = l1_temp, l2_temp
```

### 19. Remove Nth Node From End of List

```python
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        total = 0
        p = dummy = ListNode()
        p.next = head 
        while head:
            total += 1
            head = head.next 
        cnt = 0
        while p:
            cnt += 1
            if cnt == total - n + 1:
                p.next = p.next.next 
            else:
                p = p.next 
        return dummy.next
```

### 141. Linked List Cycle

```python
class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        slow, fast = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return True
        return False
```

### 23. Merge k Sorted Lists

```python
class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        pq = []
        for head in lists:
            while head:
                heappush(pq, head.val)
                head = head.next 
        dummy = p = ListNode()
        while pq:
            v = heappop(pq)
            p.next = ListNode(v)
            p = p.next 
        return dummy.next
```

### 7 tree: 11

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

### 104. Maximum Depth of Binary Tree

```python
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        def dfs(node):
            if not node:
                return 0
            return max(dfs(node.left), dfs(node.right)) + 1
        return dfs(root)
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

### 236. Lowest Common Ancestor of a Binary Tree

```python
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        def dfs(node):
            if not node or node == p or node == q:
                return node 
            left, right = dfs(node.left), dfs(node.right)
            if left and right:
                return node 
            elif left:
                return left  
            elif right:
                return right 
        return dfs(root)
```

### 102. Binary Tree Level Order Traversal

```python
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        res, q = [], deque([root])
        while q:
            level = []
            for i in range(len(q)):
                node = q.popleft()
                level.append(node.val)
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
            res.append(level)
        return res
```

### 98. Validate Binary Search Tree

```python
class Solution:
    def isValidBST(self, root: Optional[TreeNode], l = -inf, r = inf) -> bool:
        if not root:
            return True
        v = root.val 
        if v <= l or v >= r:
            return False
        return self.isValidBST(root.left, l, v) and self.isValidBST(root.right, v, r)
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

### 105. Construct Binary Tree from Preorder and Inorder Traversal

```python
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        def dfs(preorder, inorder):
            if not preorder or not inorder:
                return None 
            node = TreeNode(preorder[0])
            m = inorder.index(preorder[0])
            node.left = dfs(preorder[1: m + 1], inorder[: m])
            node.right = dfs(preorder[m + 1:], inorder[m + 1:])
            return node 
        return dfs(preorder, inorder)
```

### 124. Binary Tree Maximum Path Sum

```python
class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        self.res = -inf
        def dfs(node):
            if not node:
                return 0
            l, r = max(dfs(node.left), 0), max(dfs(node.right), 0)
            self.res = max(self.res, node.val + l + r)
            return node.val + max(l, r)
        dfs(root)
        return self.res
```

### 297. Serialize and Deserialize Binary Tree

```python
class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        res = []
        def dfs(root):
            if not root:
                res.append('N')
                return
            res.append(str(root.val))
            dfs(root.left)
            dfs(root.right)
        dfs(root)
        return ' '.join(res)

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        vals = deque(data.split(' '))
        def dfs():
            val = vals.popleft()
            if val == 'N':
                return None
            node = TreeNode(int(val))
            node.left = dfs()
            node.right = dfs()
            return node
        return dfs()
```

## 8 heap: 1

### 295. Find Median from Data Stream

```python
from sortedcontainers import SortedList
class MedianFinder:

    def __init__(self):
        self.sl = SortedList()

    def addNum(self, num: int) -> None:
        self.sl.add(num)

    def findMedian(self) -> float:
        n = len(self.sl)
        if n % 2 == 1:
            return self.sl[n // 2]
        return (self.sl[n // 2 - 1] + self.sl[n // 2]) / 2
```

## 9 backtracking: 2

### 39. Combination Sum

```python
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        res, n = [], len(candidates)
        def backtrack(idx, ans, total):
            if total == target:
                res.append(ans)
                return 
            if total > target:
                return
            for i in range(idx, n):
                backtrack(i, ans + [candidates[i]], total + candidates[i])
        backtrack(0, [], 0)
        return res
```

### 79. Word Search

```python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        R, C = len(board), len(board[0])
        def dfs(r, c, idx):
            if idx == len(word):
                return True
            if r < 0 or r >= R or c < 0 or c >= C or board[r][c] != word[idx]:
                return False
            temp = board[r][c]
            board[r][c] = " "
            found = dfs(r + 1, c, idx + 1) or dfs(r - 1, c, idx + 1) or dfs(r, c + 1, idx + 1) or dfs(r, c - 1, idx + 1)
            board[r][c] = temp
            return found 
        
        for r in range(R):
            for c in range(C):
                if board[r][c] == word[0] and dfs(r, c, 0):
                    return True
        return False
```

## 10 Trie: 3

### 208. Implement Trie (Prefix Tree)

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.endOfWord = False

class Trie:

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        cur = self.root
        for c in word:
            if c not in cur.children:
                cur.children[c] = TrieNode()
            cur = cur.children[c]
        cur.endOfWord = True

    def search(self, word: str) -> bool:
        cur = self.root
        for c in word:
            if c not in cur.children:
                return False
            cur = cur.children[c]
        return cur.endOfWord == True

    def startsWith(self, prefix: str) -> bool:
        cur = self.root
        for c in prefix:
            if c not in cur.children:
                return False
            cur = cur.children[c]
        return True
```

### 211. Design Add and Search Words Data Structure

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.endOfWord = False

class WordDictionary:

    def __init__(self):
        self.root = TrieNode()

    def addWord(self, word: str) -> None:
        cur = self.root
        for c in word:
            if c not in cur.children:
                cur.children[c] = TrieNode()
            cur = cur.children[c]
        cur.endOfWord = True

    def search(self, word: str) -> bool:
        def dfs(idx, root):
            if not root:
                return False
            if idx == len(word):
                return root.endOfWord
            if word[idx] != '.':
                return dfs(idx + 1, root.children.get(word[idx], None))
            else:
                for child in root.children.values():
                    if dfs(idx + 1, child):
                        return True
                return False
        return dfs(0, self.root)
```

### 212. Word Search II

```python
class TrieNode: 
    def __init__(self):
        self.children = {} 
        self.endOfWord = False 

class Trie: 
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        cur = self.root
        for c in word:
            if c not in cur.children:
                cur.children[c] = TrieNode()
            cur = cur.children[c]
        cur.endOfWord = True

class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        t = Trie()
        words = set(words)
        for word in words:
            t.insert(word)
        
        def dfs(r, c, node, path):
            char = board[r][c]
            if char not in node.children:
                return
            child = node.children[char]
            if child.endOfWord:
                res.add(path + char)
                if not child.children:
                    return

            board[r][c] = "#"
            for dr, dc in directions:
                row, col = r + dr, c + dc
                if 0 <= row < R and 0 <= col < C:
                    dfs(row, col, child, path + char)
            board[r][c] = char

        R, C = len(board), len(board[0])
        directions = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        res = set()
        for r in range(R):
            for c in range(C):
                dfs(r, c, t.root, '')
        return list(res)
```

## 11 dp 1d: 10

### 70. Climbing Stairs

```python
class Solution:
    def climbStairs(self, n: int) -> int:
        first, second = 1, 2
        for i in range(3, n + 1):
            second, first = second + first, second 
        return second if n >= 2 else first
```

### 198. House Robber

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        dp = [0] * (len(nums) + 1)
        dp[1] = nums[0]
        for i in range(2, len(dp)):
            dp[i] = max(dp[i - 1], dp[i - 2] + nums[i - 1])
        return dp[-1]
```

### 213. House Robber II

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        def rob_a_line(nums):
            dp = [0] + nums 
            for i in range(2, len(dp)):
                dp[i] = max(dp[i - 1], dp[i - 2] + nums[i - 1])
            return dp[-1]
        if len(nums) == 1:
            return nums[0]
        return max(rob_a_line(nums[1: ]), rob_a_line(nums[: -1]))
```

### 5. Longest Palindromic Substring

```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        n, res = len(s), s[0]
        dp = [[False] * n for r in range(n)]
        for c in range(1, n):
            for r in range(c):
                if s[r] == s[c] and (c - r + 1 <= 3 or dp[r + 1][c - 1]):
                    dp[r][c] = True
                    if c - r + 1 > len(res):
                        res = s[r: c + 1]
        return res 
```

### 647. Palindromic Substrings

```python
class Solution:
    def countSubstrings(self, s: str) -> int:
        def helper(i, j):
            res = 0
            while i >= 0 and j < len(s) and s[i] == s[j]:
                i -= 1
                j += 1
                res += 1
            return res 
        res = 0
        for i in range(len(s)):
            res += helper(i, i)
            res += helper(i, i + 1)
        return res 
```

### 91. Decode Ways

```python
class Solution:
    def numDecodings(self, s: str) -> int:
        n = len(s)
        f = [1] + [0] * n
        for i in range(1, n + 1):
            if s[i - 1] != '0':
                f[i] = f[i - 1]
            if i > 1 and 10 <= int(s[i-2:i]) <= 26:
                f[i] += f[i - 2]
        return f[n]
```

### 322. Coin Change

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        @cache
        def dfs(t, i):
            if t > amount:
                return inf 
            if i == len(coins):
                return 0 if t == amount else inf 
            return min(dfs(t, i + 1), dfs(t + coins[i], i) + 1)
        res = dfs(0, 0)
        return res if res != inf else -1
```

### 152. Maximum Product Subarray

```python
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        mx, mn, res = nums[0], nums[0], nums[0]
        for i in range(1, len(nums)):
            temp = mx
            mx = max(mx * nums[i], mn * nums[i], nums[i])
            mn = min(temp * nums[i], mn * nums[i], nums[i])
            res = max(res, mx)
        return res
```

### 139. Word Break

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        n, wordDict = len(s), set(wordDict)
        dp = [True] + [False] * n
        for i in range(n + 1):
            for j in range(i + 1, n + 1):
                if dp[i] and s[i:j] in wordDict:
                    dp[j] = True
        return dp[-1]
```

### 300. Longest Increasing Subsequence

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [1] * n
        for r in range(1, n):
            for l in range(r):
                if nums[r] > nums[l]:
                    dp[r] = max(dp[r], dp[l] + 1)
        return max(dp)
```

## 11 dp 2d: 2

### 62. Unique Paths

```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        R, C = m, n
        dp = [[1] * C for i in range(R)]
        for r in range(1, m):
            for c in range(1, n):
                dp[r][c] = dp[r - 1][c] + dp[r][c - 1]
        return dp[-1][-1]
```
### 1143. Longest Common Subsequence

```python
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        R, C = len(text1) + 1, len(text2) + 1
        dp = [[0] * C for r in range(R)]
        for r in range(1, R):
            for c in range(1, C):
                if text1[r - 1] != text2[c - 1]:
                    dp[r][c] = max(dp[r - 1][c], dp[r][c - 1])
                else:
                    dp[r][c] = dp[r - 1][c - 1] + 1
        return dp[-1][-1]
```

## 12 greedy: 2

### 53. Maximum Subarray

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [nums[0]] * n 
        for i in range(1, n):
            dp[i] = max(nums[i], dp[i - 1] + nums[i])
        return max(dp)
```

### 55. Jump Game

```python
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        n = len(nums)
        furthest = 0
        for i in range(n):
            if i <= furthest:
                furthest = max(i + nums[i], furthest)
            if furthest >= n - 1:
                return True
        return False
```

## 13 interval: 5

### 57. Insert Interval

```python
class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        intervals.append(newInterval)
        events = []
        for s, e in intervals:
            events.append((s, -1))
            events.append((e, 1))
        events.sort()

        count, res = 0, []
        start, end = inf, -inf
        for point, sign in events:
            if sign < 0:
                count += 1
                start = min(start, point)
            else:
                count -= 1
                end = max(end, point)
            if count == 0:
                res.append([start, end])
                start, end = inf, -inf
        return res
```

### 56. Merge Intervals

```python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        events = []
        for s, e in intervals:
            events.append((s, -1))
            events.append((e, 1))
        events.sort()

        start, end = inf, -inf 
        res, count = [], 0
        for point, sign in events:
            if sign < 0:
                count += 1
                start = min(start, point)
            else:
                count -= 1
                end = max(end, point)
            if count == 0:
                res.append([start, end])
                start, end = inf, -inf 
        return res
```

### 435. Non-overlapping Intervals

```python
class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        intervals.sort(key = lambda x: x[1])
        prev, count = -inf, 0
        for s, e in intervals:
            if s >= prev:
                count += 1
                prev = e
        return len(intervals) - count
```

### 252. Meeting Room

```python
class Solution:
    def canAttendMeetings(self, intervals: List[List[int]]) -> bool:
        events = []
        for start, end in intervals:
            events.append([start, 1])
            events.append([end, -1])
        events.sort()
        
        count = 0
        for time, sign in events:
            count += sign
            if count > 1:
                return False
        return True
```

### 253. Meeting Rooms II

```python
class Solution:
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        events = []
        for start, end in intervals:
            events.append([start, 1])
            events.append([end, -1])
        events.sort()
        
        count, res = 0, 1
        for time, sign in events:
            count += sign
            res = max(res, count)
        return res
```

## 14 math & geometry

### 48. Rotate Image

```python
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        matrix.reverse()
        R, C = len(matrix), len(matrix[0])
        for r in range(R):
            for c in range(r):
                matrix[r][c], matrix[c][r] = matrix[c][r], matrix[r][c]
        return matrix
```

### 54. Spiral Matrix

```python
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        R, C = len(matrix), len(matrix[0])
        top, bottom, left, right = 0, R - 1, 0, C - 1
        res = []
        while top <= bottom and left <= right:
            # top
            for c in range(left, right + 1):
                res.append(matrix[top][c])
            top += 1
            # right
            for r in range(top, bottom + 1):
                res.append(matrix[r][right])
            right -= 1
            if left > right or top > bottom:
                continue
            # bottom
            for c in range(right, left - 1, -1):
                res.append(matrix[bottom][c])
            bottom -= 1
            # left
            for r in range(bottom, top - 1, -1):
                res.append(matrix[r][left])
            left += 1
        return res 
```

### 73. Set Matrix Zeroes

```python
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        R, C, row, col = len(matrix), len(matrix[0]), set(), set()
        for r in range(R):
            for c in range(C):
                if matrix[r][c] == 0:
                    row.add(r)
                    col.add(c)
        for r in range(R):
            for c in range(C):
                if r in row or c in col:
                    matrix[r][c] = 0
```

## 15 bit manipulation: 5

### 191. Number of 1 Bits

```python
class Solution:
    def hammingWeight(self, n: int) -> int:
        return n.bit_count()
```

### 338. Counting Bits

```python
class Solution:
    def countBits(self, n: int) -> List[int]:
        dp = [0] * (n + 1)
        for i in range(1, n + 1):
            dp[i] = dp[i // 2] + 1 if i % 2 else dp[i // 2]
        return dp
```

### 190. Reverse Bits

```python
class Solution:
    def reverseBits(self, n: int) -> int:
        n = bin(n)[2:].zfill(32)
        n = n[::-1]
        return int(n, 2)
```

### 268. Missing Number

```python
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        s = set(nums)
        n = len(nums)
        for i in range(n + 1):
            if i not in s:
                return i 
```

### 371. Sum of Two Integers

```python
class Solution:
    def getSum(self, a: int, b: int) -> int:
        return int(log2(2 ** a * 2 ** b))
```

## 16 graph: 7

### 200. Number of Islands

```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        def dfs(r, c):
            grid[r][c] = '0'
            for dr, dc in directions:
                row, col = r + dr, c + dc 
                if 0 <= row < R and 0 <= col < C and grid[row][col] == '1':
                    dfs(row, col)

        R, C, res = len(grid), len(grid[0]), 0
        directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        for r in range(R):
            for c in range(C):
                if grid[r][c] == '1':
                    dfs(r, c)
                    res += 1
        return res
```

### 133. Clone Graph

```python
from typing import Optional
class Solution:
    def cloneGraph(self, node: Optional['Node']) -> Optional['Node']:
        if not node:
            return None
        cloned = {node: Node(node.val)}
        q = deque([node])
        while q:
            cur = q.popleft()
            for nei in cur.neighbors:
                if nei not in cloned:
                    cloned[nei] = Node(nei.val)
                    q.append(nei)
                cloned[cur].neighbors.append(cloned[nei])
        return cloned[node]
```

### 417. Pacific Atlantic Water Flow

```python
class Solution:
    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        R, C = len(heights), len(heights[0])
        P, A = set(), set()
        directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        
        def dfs(r, c, visited, prevHeight):
            for dr, dc in directions:
                row, col = r + dr, c + dc 
                if 0 <= row < R and 0 <= col < C and (row, col) not in visited and heights[row][col] >= prevHeight:
                    visited.add((row, col))
                    dfs(row, col, visited, heights[row][col])
            
        for c in range(C):
            P.add((0, c))
            dfs(0, c, P, heights[0][c])
            A.add((R - 1, c))
            dfs(R - 1, c, A, heights[R - 1][c])
        for r in range(R):
            P.add((r, 0))
            dfs(r, 0, P, heights[r][0])
            A.add((r, C - 1))
            dfs(r, C - 1, A, heights[r][C - 1])
        
        res = []
        for r in range(R):
            for c in range(C):
                if (r, c) in P and (r, c) in A:
                    res.append([r, c])
        return res
```

### 207. Course Schedule

```python
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        g, indegree = defaultdict(list), [0] * numCourses
        for a, b in prerequisites:
            g[b].append(a)
            indegree[a] += 1

        res, q = 0, deque([i for i, d in enumerate(indegree) if d == 0])
        while q:
            node = q.popleft()
            res += 1
            for nei in g[node]:
                indegree[nei] -= 1
                if indegree[nei] == 0:
                    q.append(nei)
        return res == numCourses
```

### 323. Number of Connected Components in an Undirected Graph

```python
class UF:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, root):
        n = root # use n for path compression
        while self.parent[root] != root:
            root = self.parent[root] # find root first

        while n != root: # start path compression
            nxt = self.parent[n]
            self.parent[n] = root
            n = nxt
        return root # get root

    def connected(self, n1, n2):
        return self.find(n1) == self.find(n2)

    def union(self, n1, n2):
        p1, p2 = self.find(n1), self.find(n2)
        self.parent[p2] = p1

class Solution:
    def countComponents(self, n: int, edges: List[List[int]]) -> int:
        uf = UF(n)
        for u, v in edges:
            if not uf.connected(u, v):
                uf.union(u, v)
                n -= 1
        return n
```

### 261. Graph Valid Tree

```python
class UF:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, n):
        while n != self.parent[n]:
            self.parent[n] = self.parent[self.parent[n]]
            n = self.parent[n]
        return n 

    def union(self, n1, n2):
        p1, p2 = self.find(n1), self.find(n2)
        self.parent[p1] = p2

    def isConnected(self, n1, n2):
        return self.find(n1) == self.find(n2)

class Solution:
    def validTree(self, n: int, edges: List[List[int]]) -> bool:
        if n != len(edges) + 1:
            return False
        uf = UF(n)
        for u, v in edges:
            if uf.isConnected(u, v):
                return False
            uf.union(u, v)
        return True
```

### 269. Alien Dictionary

```python
```