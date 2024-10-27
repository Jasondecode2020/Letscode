## template: backtracking

- terminations (res)
- tracking (dfs)
- trimming (timing)


```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]: # 46. Permutations
        def backtrack(nums, path):
            if not nums:
                res.append(path[::])
                return 
            for i in range(len(nums)):
                backtrack(nums[: i] + nums[i + 1: ], path + [nums[i]])
        res = []
        backtrack(nums, [])
        return res
```

### Backtracking

> select/not select (select/not select, select left/select right)

* [257. Binary Tree Paths](#257-Binary-Tree-Paths)[E]
* [988. Smallest String Starting From Leaf](#988-Smallest-String-Starting-From-Leaf)[M][same as 257]
* [113. Path Sum II](#113-Path-Sum-II)[M][same as 257]
* [1863. Sum of All Subset XOR Totals](#1863-Sum-of-All-Subset-XOR-Totals)[E]
* [22. Generate Parentheses](#22-Generate-Parentheses)[M]
* [78. Subsets](#78-Subsets)[M]
* [90. Subsets II](#90-Subsets-II)[M]
* [784. Letter Case Permutation](#784-Letter-Case-Permutation)
* [1239. Maximum Length of a Concatenated String with Unique Characters](#784-Letter-Case-Permutation)
* [2044. Count Number of Maximum Bitwise-OR Subsets](#2044-count-number-of-maximum-bitwise-or-subsets) 1567
* [1255. Maximum Score Words Formed by Letters](#1255-maximum-score-words-formed-by-letters)
* [2397. Maximum Rows Covered by Columns](#2397-maximum-rows-covered-by-columns)
* [1079. Letter Tile Possibilities](#1079-letter-tile-possibilities)
* [1219. Path with Maximum Gold](#1219-path-with-maximum-gold)

> enumerate pattern (cut) (need a for loop to enumerate all conditions)

* [39. Combination Sum](#39-Combination-Sum)[M]
* [40. Combination Sum II](#40-Combination-Sum-II)[M]
* [216. Combination Sum III](#216-Combination-Sum-III)[M]
* [377. Combination Sum IV](#377-Combination-Sum-IV)[M](solved by dp, backtrack idea)
* [46. Permutations](#46-Permutations)[M]
* [47. Permutations II](#47-Permutations-II)[M]
* [77. Combinations](#77-Combinations)[M]
* [17. Letter Combinations of a Phone Number](#17-Letter-Combinations-of-a-Phone-Number)[M]
* [996. Number of Squareful Arrays](#996-Number-of-Squareful-Arrays)
* [291. Word Pattern II](#291-word-pattern-ii)

> games (need to find how to play the game)

* [37. Sudoku Solver](#37-Sudoku-Solver)
* [51. N-Queens](#51-N-Queens)
* [52. N-Queens II](#52-N-Queens-II)
* [679. 24 Game](#679-24-Game)

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
        def backtrack(node, path): # preorder traversal/backtrack
            if node:
                if not node.left and not node.right: # find leaf
                    self.res.append(path + [node.val])
                    return
                backtrack(node.left, path + [node.val]) # select left
                backtrack(node.right, path + [node.val]) # select right
        backtrack(root, [])
        return check(self.res)
```

### 988. Smallest String Starting From Leaf

- only change check function

```python
class Solution:
    def smallestFromLeaf(self, root: Optional[TreeNode]) -> str:
        letters = ascii_lowercase
        def check(res):
            for i, a in enumerate(res):
                res[i] = ''.join([letters[i] for i in a])
            return sorted(res)[0]
            
        self.res = []
        def backtrack(node, path):
            if node:
                if not node.left and not node.right:
                    self.res.append([node.val] + path)
                    return
                backtrack(node.left, [node.val] + path) # select left
                backtrack(node.right, [node.val] + path) # select right
        backtrack(root, [])
        return check(self.res)
```

### 113. Path Sum II

```python
class Solution:
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> List[List[int]]:
        res = []
        def backtrack(node, path, presum):
            if node:
                if not node.left and not node.right and presum + node.val == targetSum:
                    res.append(path + [node.val])
                backtrack(node.right, path + [node.val], presum + node.val) # select right
                backtrack(node.left, path + [node.val], presum + node.val) # select left
            return res
        return backtrack(root, [], 0)
```

### 1863. Sum of All Subset XOR Totals

- brute force

```python
class Solution:
    def subsetXORSum(self, nums: List[int]) -> int:
        def check(a):
            res = 0
            for n in a:
                res ^= n
            return res

        n, res = len(nums), 0
        for i in range(n + 1):
            for a in combinations(nums, i):
                res += check(a)
        return res
```

- backtrack

```python
class Solution:
    def subsetXORSum(self, nums: List[int]) -> int:
        self.res, n = 0, len(nums)
        def backtrack(idx, val):
            if idx == n:
                self.res += val
                return
            backtrack(idx + 1, val) # not select
            backtrack(idx + 1, val ^ nums[idx]) # select
        backtrack(0, 0)
        return self.res
```

### 17. Letter Combinations of a Phone Number

```python
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        d = {
            '2': 'abc',
            '3': 'def',
            '4': 'ghi',
            '5': 'jkl',
            '6': 'mno',
            '7': 'pqrs',
            '8': 'tuv',
            '9': 'wxyz'
        }

        res = ['']
        for digit in digits:
            temp = []
            for c in d[digit]:
                for item in res:
                    temp.append(item + c)
            res = temp
        return res if res[0] else []
```

```python
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        d = {
            '2': 'abc',
            '3': 'def',
            '4': 'ghi',
            '5': 'jkl',
            '6': 'mno',
            '7': 'pqrs',
            '8': 'tuv',
            '9': 'wxyz'
        }
        res, n = [], len(digits)
        def backtrack(i, cur):
            if len(cur) == n:
                if cur:
                    res.append(cur)
                return
            for c in d[digits[i]]:
                backtrack(i + 1, cur + c)

        backtrack(0, '')
        return res
```

### 22. Generate Parentheses

```python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        res = []
        def backtrack(open, close, ans):
            if open == close == n:
                res.append(ans)
                return 
            if open < n:
                backtrack(open + 1, close, ans + '(')
            if close < open:
                backtrack(open, close + 1, ans + ')')
        backtrack(0, 0, '')
        return res
```

### 39. Combination Sum

- backtracking

```python
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        res, n = [], len(candidates)
        def backtrack(idx, ans, cur):
            if cur == target:
                res.append(ans)
                return 
            if cur > target:
                return
            for i in range(idx, n):
                backtrack(i, ans + [candidates[i]], cur + candidates[i])
        backtrack(0, [], 0)
        return res
```

### 40. Combination Sum II

- backtracking

```python
class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        res, n = [], len(candidates)
        candidates.sort()
        def backtrack(idx, ans, cur):
            if cur == target:
                res.append(ans)
                return 
            if cur > target:
                return
            for i in range(idx, n):
                # for each layer of backtracking, no duplicate
                if i > idx and candidates[i] == candidates[i - 1]: 
                    continue
                backtrack(i + 1, ans + [candidates[i]], cur + candidates[i])
        backtrack(0, [], 0)
        return res
```

### 46. Permutations

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        def backtrack(nums, path):
            if not nums:
                res.append(path[::])
                return 
            for i in range(len(nums)):
                backtrack(nums[: i] + nums[i + 1: ], path + [nums[i]])
        res = []
        backtrack(nums, [])
        return res
```

### 47. Permutations II

```python
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        res = []
        def backtrack(nums, path):
            if not nums:
                res.append(path[::])
                return 
            for i in range(len(nums)):
                if i > 0 and nums[i] == nums[i - 1]: # trim branch, check duplicate
                    continue
                backtrack(nums[: i] + nums[i + 1: ], path + [nums[i]])
        backtrack(sorted(nums), []) # sort to avoid duplicate
        return res
```

### 77. Combinations

```python
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        return list(combinations(range(1, n + 1), k))
```

```python
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        def backtrack(nums, idx):
            if len(nums) == k:
                res.append((nums))
                return
            if len(nums) > k:
                return
            for i in range(idx, n + 1):
                backtrack(nums + [i], i + 1)

        res = []
        backtrack([], 1)
        return res
```

### 78. Subsets

```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        res, n = [], len(nums)
        def backtrack(i, subset):
            if i == n:
                res.append(subset)
                return
            backtrack(i + 1, subset + [nums[i]]) # select
            backtrack(i + 1, subset) # not select
        backtrack(0, [])
        return res
```

### 90. Subsets II

```python
class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        res, n = [], len(nums)
        def backtrack(i, subset):
            if i == n:
                res.append(subset)
                return
            backtrack(i + 1, subset + [nums[i]]) # select
            while i + 1 < len(nums) and nums[i] == nums[i + 1]:
                i += 1
            backtrack(i + 1, subset) # not select

        nums.sort()
        backtrack(0, [])
        return res
```

### 784. Letter Case Permutation

```python
class Solution:
    def letterCasePermutation(self, s: str) -> List[str]:
        res, n = [], len(s)
        lowercase = set(list(ascii_lowercase))
        uppercase = set(list(ascii_uppercase))
        def backtrack(i, path):
            if i == n:
                res.append(path)
                return
            if s[i] in lowercase:
                backtrack(i + 1, path + s[i].upper())
            if s[i] in uppercase:
                backtrack(i + 1, path + s[i].lower())
            backtrack(i + 1, path + s[i])
        backtrack(0, '')
        return res
```

### 216. Combination Sum III

```python
class Solution:
    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        nums, res = list(range(1, 10)), []
        L = len(nums)
        def backtrack(idx, ans, cur, length):
            if cur == n and length == k:
                res.append(ans)
                return
            if cur > n or length > k:
                return
            for i in range(idx, L):
                backtrack(i + 1, ans + [nums[i]], cur + nums[i], length + 1)
        backtrack(0, [], 0, 0)
        return res
```

### 377. Combination Sum IV

- backtrack not possible (time limited)

```python
class Solution:
    def combinationSum4(self, nums: List[int], target: int) -> int:
        self.res, n = 0, len(nums)
        def backtrack(idx, cur):
            if cur == target:
                self.res += 1
                return 
            if cur > target:
                return
            for i in range(0, n):
                backtrack(i, cur + nums[i])
        backtrack(0, 0)
        return self.res
```

- unbounded knapsack

```python
class Solution:
    def combinationSum4(self, nums: List[int], target: int) -> int:
        dp = [1] + [0] * target
        for i in range(1, target + 1):
            for num in nums:
                if num <= i:
                    dp[i] += dp[i - num]
        return dp[-1]
```

### 996. Number of Squareful Arrays

```python
class Solution:
    def numSquarefulPerms(self, nums: List[int]) -> int:
        def is_perfect_square(n):
            return pow(int(sqrt(n)), 2) == n

        self.res = 0
        def backtrack(nums, path):
            if not nums:
                self.res += 1
                return
            for i in range(len(nums)):
                if i > 0 and nums[i] == nums[i - 1]:
                    continue
                if not path or is_perfect_square(path[-1] + nums[i]):
                    path.append(nums[i])
                    backtrack(nums[: i] + nums[i + 1: ], path)
                    path.pop()
                    
        backtrack(sorted(nums), [])
        return self.res
```

### 37. Sudoku Solver

```python
class Solution:
    def solveSudoku(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        rows, cols, squares, q = defaultdict(set), defaultdict(set), defaultdict(set), deque([])
        for i in range(9):
            for j in range(9):
                if board[i][j] != ".":
                    rows[i].add(board[i][j])
                    cols[j].add(board[i][j])
                    squares[(i // 3, j // 3)].add(board[i][j])
                else:
                    q.append((i,j))

        def backtrack():
            if not q:
                self.valid = True
                return
            r, c = q[0]
            square = (r // 3, c // 3)
            for n in numbers: # try 9 ways
                if n not in (rows[r] | cols[c] | squares[square]):
                    board[r][c] = n
                    rows[r].add(n)
                    cols[c].add(n)
                    squares[square].add(n)
                    q.popleft()
                    backtrack()
                    if not self.valid: # backtrack
                        board[r][c] = "."
                        rows[r].remove(n)
                        cols[c].remove(n)
                        squares[square].remove(n)
                        q.appendleft((r,c))
        numbers = set([str(i) for i in range(1, 10)])
        self.valid = False
        backtrack()
```

### 51. N-Queens

- backtracking

```python
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        col, posDiag, negDiag = set(), set(), set()
        res, board = [], [['.'] * n for i in range(n)]
        def backtrack(r):
            if r == n:
                res.append([''.join(row) for row in board])
                return

            for c in range(n):
                if c in col or (r + c) in posDiag or (r - c) in negDiag:
                    continue
                col.add(c)
                posDiag.add(r + c)
                negDiag.add(r - c)
                board[r][c] = 'Q'
                backtrack(r + 1)
                col.remove(c)
                posDiag.remove(r + c)
                negDiag.remove(r - c)
                board[r][c] = '.'
        backtrack(0)
        return res
```

### 52. N-Queens II

- backtracking

```python
class Solution:
    def totalNQueens(self, n: int) -> int:
        col, posDiag, negDiag = set(), set(), set()
        res, board = 0, [['.'] * n for i in range(n)]
        def backtrack(r):
            if r == n:
                nonlocal res
                res += 1
                return

            for c in range(n):
                if c in col or (r + c) in posDiag or (r - c) in negDiag:
                    continue
                col.add(c)
                posDiag.add(r + c)
                negDiag.add(r - c)
                backtrack(r + 1)
                col.remove(c)
                posDiag.remove(r + c)
                negDiag.remove(r - c)
        backtrack(0)
        return res
```

### 679. 24 Game

```python
class Solution:
    def judgePoint24(self, cards: List[int]) -> bool:
        ADD, MULTIPLY, SUBTRACT, DIVIDE = 0, 1, 2, 3
        def backtrack(nums):
            if len(nums) == 1:
                return math.isclose(nums[0], 24) == True
            for i, x in enumerate(nums):
                for j, y in enumerate(nums):
                    if i != j:
                        res = []
                        for k, z in enumerate(nums):
                            if k != i and k != j:
                                res.append(z)
                        for k in range(4):
                            if k == ADD:
                                res.append(x + y)
                            elif k == MULTIPLY:
                                res.append(x * y)
                            elif k == SUBTRACT:
                                res.append(x - y)
                            elif k == DIVIDE:
                                res.append(y and x / y)
                            if backtrack(res):
                                return True
                            res.pop()
            return False
        return backtrack(cards)
```

### 140. Word Break II

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> List[str]:
        res, wordDict = [], set(wordDict)
        def dfs(idx, ans):
            if idx == len(s):
                res.append(' '.join(ans))
                return
            for i in range(idx + 1, len(s) + 1):
                if s[idx:i] in wordDict:
                    dfs(i, ans + [s[idx:i]])     
        dfs(0, [])
        return res
```

### 491. Non-decreasing Subsequences

```python
class Solution:
    def findSubsequences(self, nums: List[int]) -> List[List[int]]:
        res = []
        def backtrack(nums, ans):
            if len(ans) > 1:
                res.append(ans)
            s = set()
            for i, n in enumerate(nums):
                if n in s:
                    continue
                if not ans or n >= ans[-1]:
                    s.add(n)
                    backtrack(nums[i + 1:], ans + [n])
        
        backtrack(nums, [])
        return res
```

### 93. Restore IP Addresses

```python
class Solution:
    def restoreIpAddresses(self, s: str) -> List[str]:
        def backtrack(i, ans):
            if len(ans) == 4 and i == len(s):
                res.append('.'.join(ans))
                return
            if len(ans) > 4:
                return 
            for j in range(i, min(i + 3, len(s))):
                if int(s[i: j + 1]) < 256 and (s[i] != '0' or i == j):
                # if int(s[i: j + 1]) < 256 and s[i] != '0' or i == j: # also ok
                    backtrack(j + 1, ans + [s[i: j + 1]])
        res = []
        backtrack(0, [])
        return res
```

### 90. Subsets II

```python
class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        def backtrack(i, ans):
            res.append(ans)
            for j in range(i, n):
                if j > i and nums[j] == nums[j - 1]:
                    continue 
                backtrack(j + 1, ans + [nums[j]])
        res, n = [], len(nums)
        nums.sort()
        backtrack(0, [])
        return res
```


### 1239. Maximum Length of a Concatenated String with Unique Characters

```python
class Solution:
    def maxLength(self, arr: List[str]) -> int:
        n = len(arr)
        self.res = 0
        def backtrack(i, ans):
            if len(ans) != len(set(list(ans))):
                return 
            if i == n:
                self.res = max(self.res, len(ans))
                return 
            backtrack(i + 1, ans)
            backtrack(i + 1, ans + arr[i])
        backtrack(0, '')
        return self.res
```

### 923. 3Sum With Multiplicity

```python
class Solution:
    def threeSumMulti(self, arr: List[int], target: int) -> int:
        arr.sort(reverse = True)
        n = len(arr)
        mod = 10 ** 9 + 7
        @cache
        def dfs(i, total, count):
            if i >= n:
                return 1 if total == target and count == 3 else 0
            if total > target or count > 3:
                return 0
            return dfs(i + 1, total, count) + dfs(i + 1, total + arr[i], count + 1)
        return dfs(0, 0, 0) % mod
```

### 1415. The k-th Lexicographical String of All Happy Strings of Length n

```python
class Solution:
    def getHappyString(self, n: int, k: int) -> str:
        res = []
        def backtrack(ans):
            if len(ans) == n:
                res.append(ans)
                return 
            if len(ans) > n:
                return 
            for c in 'abc':
                if ans and ans[-1] != c:
                    backtrack(ans + c)
                if not ans:
                    backtrack(c)
        backtrack('')
        res.sort()
        return res[k - 1] if k <= len(res) else ''
```

### 967. Numbers With Same Consecutive Differences

```python
class Solution:
    def numsSameConsecDiff(self, n: int, k: int) -> List[int]:
        res = []
        def backtrack(ans):
            if len(ans) == n:
                res.append(int(ans))
                return
            if len(ans) > n:
                return 
            for c in '0123456789':
                if not ans and c != '0':
                    backtrack(c)
                if ans and abs(ord(ans[-1]) - ord(c)) == k:
                    backtrack(ans + c)
        backtrack('')
        return res 
```

### 842. Split Array into Fibonacci Sequence

```python
class Solution:
    def splitIntoFibonacci(self, num: str) -> List[int]:
        def backtrack(i):
            if i == n:
                return len(res) >= 3 
            cur = 0
            for j in range(i, n):
                if j > i and num[i] == '0': # with leading zero
                    break
                cur = cur * 10 + ord(num[j]) - ord('0')
                if cur > 2 ** 31 - 1:
                    break 
                if len(res) < 2 or cur == res[-2] + res[-1]:
                    res.append(cur)
                    if backtrack(j + 1):
                        return True # break until find one group
                    res.pop()
                if len(res) >= 2 and cur > res[-2] + res[-1]:
                    break
        n = len(num)
        res = []
        backtrack(0)
        return res
```

### 842. Split Array into Fibonacci Sequence

```python
class Solution:
    def splitIntoFibonacci(self, num: str) -> List[int]:
        n = len(num)
        def backtrack(i):
            if i == n:
                return len(res) >= 3
            cur = 0
            for j in range(i, n):
                if j > i and num[i] == '0':
                    break 
                cur = cur * 10 + ord(num[j]) - ord('0')
                if cur >= 2 ** 31:
                    break 
                if len(res) < 2 or cur == res[-2] + res[-1]:
                    res.append(cur)
                    if backtrack(j + 1):
                        return True
                    res.pop()
                if len(res) >= 2 and cur > res[-2] + res[-1]:
                    break
            return False
        res = []
        backtrack(0)
        return res
```

### 267. Palindrome Permutation II

```python
class Solution:
    def generatePalindromes(self, s: str) -> List[str]:
        n = len(s)
        def backtrack(a, ans):
            if len(ans) == n // 2:
                res.append(ans)
                return 
            for i in range(len(a)):
                if i > 0 and a[i] == a[i - 1]:
                    continue
                backtrack(a[:i] + a[i+1:], ans + a[i])
        d = Counter(s)
        new_string = ''
        count = 0
        mid_string = ''
        for k, v in d.items():
            if v % 2 == 1:
                mid_string = k
                count += 1
                if count > 1:
                    return []
            new_string += k * (v // 2)
        res = []
        new_string = ''.join(sorted(list(new_string)))
        backtrack(new_string, '')
        if count == 0:
            res = [item + item[::-1] for item in res]
        else:
            res = [item + mid_string + item[::-1] for item in res]
        return res
```

### 1087. Brace Expansion

```python
class Solution:
    def expand(self, s: str) -> List[str]:
        # "{a,b}c{d,e}f"
        s = s.replace(',', '')
        i, n = 0, len(s)
        res = []
        while i < n:
            if s[i] == '{':
                start = i 
                j = start + 1
                ans = ''
                while j < n and s[j] != '}':
                    ans += s[j]
                    j += 1
                res.append(ans)
                i = j + 1
            else:
                res.append(s[i])
                i += 1
        ans = []
        L = len(res)
        def backtrack(valid, i):
            if i == L:
                ans.append(valid)
                return 
            for c in res[i]:
                backtrack(valid + c, i + 1)
        backtrack('', 0)
        return sorted(ans)
```

### 291. Word Pattern II

```python
class Solution:
    def wordPatternMatch(self, pattern: str, s: str) -> bool:
        m = len(pattern)
        res = []
        def backtrack(s, ans):
            if len(ans) == m - 1:
                ans.append(s)
                res.append(ans)
                return 
            for i in range(1, len(s)):
                backtrack(s[i:], ans + [s[:i]])
        backtrack(s, [])
        
        def check(words, pattern):
            d1, d2 = {}, {}
            for w, p in zip(words, pattern):
                d1[w] = p 
                d2[p] = w 
            return [(k, v) for k, v in d1.items()] == [(v, k) for k, v in d2.items()]

        for words in res:
            if check(words, pattern):
                return True
        return False
```

### 2397. Maximum Rows Covered by Columns

```python
class Solution:
    def maximumRows(self, matrix: List[List[int]], numSelect: int) -> int:
        R = len(matrix)
        n = len(matrix[0])
        res = []
        def backtrack(i, ans):
            if len(ans) > numSelect:
                return 
            if i == n:
                if len(ans) == numSelect:
                    res.append(ans)
                return 
            backtrack(i + 1, ans)
            backtrack(i + 1, ans + [i])
        backtrack(0, [])
        
        d = dict((i, v.count(1)) for i, v in enumerate(matrix))
        ans = 0
        for arr in res:
            c = Counter()
            for col in arr:
                for row in range(R):
                    if matrix[row][col] == 1:
                        c[row] += 1
            
            cnt = 0
            for k in d:
                if d[k] == c[k]:
                    cnt += 1
            ans = max(ans, cnt)
        return ans
```

### 1079. Letter Tile Possibilities

```python
class Solution:
    def numTilePossibilities(self, tiles: str) -> int:
        def backtrack(i, res):
            if i == n:
                count.add(res)
                return 
            backtrack(i + 1, res)
            backtrack(i + 1, res + perm[i])

        count = set()
        for perm in permutations(tiles):
            perm = ''.join(perm)
            n = len(perm)
            backtrack(0, '')
        return len(count) - 1
```

### 1219. Path with Maximum Gold

```python
class Solution:
    def getMaximumGold(self, grid: List[List[int]]) -> int:
        directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        def backtrack(r, c, cur):
            flag = True
            for dr, dc in directions:
                row, col = r + dr, c + dc 
                if 0 <= row < R and 0 <= col < C and grid[row][col] and (row, col) not in visited:
                    flag = False
                    visited.add((row, col))
                    backtrack(row, col, cur + [grid[row][col]])
                    visited.remove((row, col))
            if flag:
                ans.append(cur)

        R, C = len(grid), len(grid[0])
        res = 0
        for r in range(R):
            for c in range(C):
                if grid[r][c]:
                    ans = []
                    visited = set([(r, c)])
                    backtrack(r, c, [grid[r][c]])
                    res = max(res, max([sum(i) for i in ans]))
        return res 
```

### 1255. Maximum Score Words Formed by Letters

```python
class Solution:
    def maxScoreWords(self, words: List[str], letters: List[str], score: List[int]) -> int:
        n = len(words)
        c = Counter(letters)
        res = []
        def backtrack(i, ans):
            if i == n:
                res.append(ans)
                return 
            backtrack(i + 1, ans)
            if Counter(words[i]) <= c:
                for letter in words[i]:
                    c[letter] -= 1
                backtrack(i + 1, ans + [words[i]])
                for letter in words[i]:
                    c[letter] += 1
        backtrack(0, [])
        score = dict(zip(ascii_lowercase, score))
        ans = 0
        def getScore(a):
            res = 0
            for word in a:
                res += sum(score[c] for c in word)
            return res
        for arr in res:
            ans = max(ans, getScore(arr))
        return ans
```

### 2044. Count Number of Maximum Bitwise-OR Subsets

```python
class Solution:
    def countMaxOrSubsets(self, nums: List[int]) -> int:
        n = len(nums)
        res = []
        def backtrack(i, ans):
            if i == n:
                res.append(ans)
                return 
            backtrack(i + 1, ans)
            backtrack(i + 1, ans + [nums[i]])
        backtrack(0, [])

        def check(a):
            ans = 0
            for n in a:
                ans |= n 
            return ans 
        
        c = Counter()
        for arr in res:
            c[check(arr)] += 1
        mx = max(c.keys())
        return c[mx]
```

### 247. Strobogrammatic Number II

```python
class Solution:
    def findStrobogrammatic(self, n: int) -> List[str]:
        pairs = [('1', '1'), ('8', '8'), ('6', '9'), ('9', '6')]
        def dfs(x):
            if x == 0:
                return ['']
            if x == 1:
                return ['0', '1', '8']
            res = []
            for num in dfs(x - 2):
                for a, b in pairs:
                    res.append(a + num + b)
                if x != n:
                    res.append('0' + num + '0')
            return res 
        return dfs(n)
```