# tree

## 1 Traversal(8)

* [144. Binary Tree Preorder Traversal](#144-binary-tree-preorder-traversal)
* [94. Binary Tree Inorder Traversal](#94-binary-tree-inorder-traversal)
* [145. Binary Tree Postorder Traversal](#145-binary-tree-postorder-traversal)
* [872. Leaf-Similar Trees](#872-leaf-similar-trees)
* [404. Sum of Left Leaves](#404-sum-of-left-leaves)
* [671. Second Minimum Node In a Binary Tree](#671-second-minimum-node-in-a-binary-tree)
* [1469. Find All The Lonely Nodes](#1469-find-all-the-lonely-nodes)
* [1214. Two Sum BSTs](#1214-two-sum-bsts)


## 2 top down DFS(16)

* [104. Maximum Depth of Binary Tree](#104-maximum-depth-of-binary-tree)
* [111. Minimum Depth of Binary Tree](#111-minimum-depth-of-binary-tree)
* [112. Path Sum](#112-path-sum)
* [129. Sum Root to Leaf Numbers](#129-sum-root-to-leaf-numbers)
* [199. Binary Tree Right Side View](#199-binary-tree-right-side-view)

* [1448. Count Good Nodes in Binary Tree](#1448-count-good-nodes-in-binary-tree)
* [1315. Sum of Nodes with Even-Valued Grandparent](#1315-sum-of-nodes-with-even-valued-grandparent)
* [988. Smallest String Starting From Leaf](#988-smallest-string-starting-from-leaf)
* [1026. Maximum Difference Between Node and Ancestor](#1026-maximum-difference-between-node-and-ancestor)
* [1022. Sum of Root To Leaf Binary Numbers](#1022-sum-of-root-to-leaf-binary-numbers)

* [623. Add One Row to Tree](#623-add-one-row-to-tree)
* [1372. Longest ZigZag Path in a Binary Tree](#1372-longest-zigzag-path-in-a-binary-tree)
* [1457. Pseudo-Palindromic Paths in a Binary Tree](#1457-pseudo-palindromic-paths-in-a-binary-tree)
* [2689. Extract Kth Character From The Rope Tree](#2689-extract-kth-character-from-the-rope-tree)
* [971. Flip Binary Tree To Match Preorder Traversal](#971-flip-binary-tree-to-match-preorder-traversal)

* [1430. Check If a String Is a Valid Sequence from Root to Leaves Path in a Binary Tree](#1430-check-if-a-string-is-a-valid-sequence-from-root-to-leaves-path-in-a-binary-tree)
* [545. Boundary of Binary Tree](#545-boundary-of-binary-tree)

## bottom up DFS

* [111. Minimum Depth of Binary Tree](#111-minimum-depth-of-binary-tree)
* [951. Flip Equivalent Binary Trees](#951-flip-equivalent-binary-trees)
* [965. Univalued Binary Tree](#965-univalued-binary-tree)
* [663. Equal Tree Partition](#663-equal-tree-partition)
* [298. Binary Tree Longest Consecutive Sequence](#298-binary-tree-longest-consecutive-sequence)

* [1973. Count Nodes Equal to Sum of Descendants](#1973-count-nodes-equal-to-sum-of-descendants)

## bottom up DFS deletion

* [814. Binary Tree Pruning]

### 144. Binary Tree Preorder Traversal

```python
class Solution:
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        def dfs(node, res):
            if node:
                res.append(node.val)
                dfs(node.left, res)
                dfs(node.right, res)
            return res 
        return dfs(root, [])
```

```java
class Solution {
    public List<Integer> preorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<Integer>();
        dfs(root, res);
        return res;
    }

    public void dfs(TreeNode node, List<Integer> res) {
        if (node != null) {
            res.add(node.val);
            dfs(node.left, res);
            dfs(node.right,res);
        }
    }
}
```

### 94. Binary Tree Inorder Traversal

```python
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        def dfs(node, res):
            if node:
                dfs(node.left, res)
                res.append(node.val)
                dfs(node.right, res)
            return res 
        return dfs(root, [])
```

```java
class Solution {
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<Integer>();
        dfs(root, res);
        return res;
    }

    public void dfs(TreeNode node, List<Integer> res) {
        if (node != null) {
            dfs(node.left, res);
            res.add(node.val);
            dfs(node.right,res);
        }
    }
}
```

### 145. Binary Tree Postorder Traversal

```python
class Solution:
    def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        def dfs(node, res):
            if node:
                dfs(node.left, res)
                dfs(node.right, res)
                res.append(node.val)
            return res 
        return dfs(root, [])
```

```java
class Solution {
    public List<Integer> postorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<Integer>();
        dfs(root, res);
        return res;
    }

    public void dfs(TreeNode node, List<Integer> res) {
        if (node != null) {
            dfs(node.left, res);
            dfs(node.right,res);
            res.add(node.val);
        }
    }
}
```

### 872. Leaf-Similar Trees

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

```java
class Solution {
    public boolean leafSimilar(TreeNode root1, TreeNode root2) {
        List<Integer> res1 = new ArrayList<Integer>();
        dfs(root1, res1);
        List<Integer> res2 = new ArrayList<Integer>();
        dfs(root2, res2);
        return res1.equals(res2);
    }

    public void dfs(TreeNode node, List<Integer> res) {
        if (node != null) {
            dfs(node.left, res);
            dfs(node.right,res);
            if (node.left == null && node.right == null) {
                res.add(node.val);
            }
        }
    }
}
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

```java
class Solution {
    private int res;

    public int sumOfLeftLeaves(TreeNode root) {
        res = 0;
        dfs(root);
        return res;
    }

    public void dfs(TreeNode node) {
        if (node != null) {
            dfs(node.left);
            dfs(node.right);
            if (node.left != null && node.left.left == null && node.left.right == null) {
                res += node.left.val;
            }
        }
    }
}
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

```java
class Solution {
    public int findSecondMinimumValue(TreeNode root) {
        Set<Integer> s = new HashSet<Integer>();
        dfs(root, s);
        List<Integer> res = new ArrayList<Integer>(s);
        Collections.sort(res);
        return res.size() > 1 ? res.get(1) : -1;
    }

    public void dfs(TreeNode node, Set<Integer> s) {
        if (node != null) {
            s.add(node.val);
            dfs(node.left, s);
            dfs(node.right, s);
        }
    }
}
```

### 1469. Find All The Lonely Nodes

```python
class Solution:
    def getLonelyNodes(self, root: Optional[TreeNode]) -> List[int]:
        def dfs(node, res):
            if node:
                if node.left and not node.right:
                    res.append(node.left.val)
                if node.right and not node.left:
                    res.append(node.right.val)
                dfs(node.left, res)
                dfs(node.right, res)
            return res 
        return dfs(root, [])
```

```java
class Solution {
    public List<Integer> getLonelyNodes(TreeNode root) {
        List<Integer> res = new ArrayList<Integer>();
        dfs(root, res);
        return res;
    }

    public void dfs(TreeNode node, List<Integer> res) {
        if (node != null) {
            if (node.left != null && node.right == null) {
                res.add(node.left.val);
            }
            if (node.right != null && node.left == null) {
                res.add(node.right.val);
            }
            dfs(node.left, res);
            dfs(node.right, res);
        }
    }
}
```

### 1214. Two Sum BSTs

```python
class Solution:
    def twoSumBSTs(self, root1: Optional[TreeNode], root2: Optional[TreeNode], target: int) -> bool:
        s = set()
        def dfs(node):
            if node:
                s.add(node.val)
                dfs(node.left)
                dfs(node.right)
    
        def dfs2(node):
            if node:
                res = target - node.val 
                if res in s:
                    return True
                return dfs2(node.left) or dfs2(node.right)
            return False
        dfs(root1)
        return dfs2(root2)
```

```java
class Solution {
    Set<Integer> s = new HashSet<Integer>();

    public boolean twoSumBSTs(TreeNode root1, TreeNode root2, int target) {
        dfs1(root1);
        return dfs2(root2, target);
    }

    private void dfs1(TreeNode node) {
        if (node != null) {
            s.add(node.val);
            dfs1(node.left);
            dfs1(node.right);
        }
    }

    private boolean dfs2(TreeNode node, int target) {
        if (node != null) {
            int res = target - node.val;
            if (s.contains(res)) {
                return true;
            }
            return dfs2(node.left, target) || dfs2(node.right, target);
        }
        return false;
    }
}
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

### 111. Minimum Depth of Binary Tree

```python
class Solution:
    def minDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        q = deque([(root, 1)])
        while q:
            node, depth = q.popleft()
            if not node.left and not node.right:
                return depth
            if node.left:
                q.append((node.left, depth + 1))
            if node.right:
                q.append((node.right, depth + 1))
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
            return dfs(node.left, presum + node.val) or dfs(node.right, presum + node.val)
            
        return dfs(root, 0)
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

### 199. Binary Tree Right Side View

```python
class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []
        res, q = [], deque([root])
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

### 1315. Sum of Nodes with Even-Valued Grandparent

```python
class Solution:
    def sumEvenGrandparent(self, root: TreeNode) -> int:
        def dfs(node):
            if node:
                if node.val % 2 == 0:
                    if node.left:
                        if node.left.left:
                            self.res += node.left.left.val 
                        if node.left.right:
                            self.res += node.left.right.val 
                    if node.right:
                        if node.right.left:
                            self.res += node.right.left.val
                        if node.right.right:
                            self.res += node.right.right.val
                dfs(node.left)
                dfs(node.right)
        self.res = 0
        dfs(root)
        return self.res 
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
        def backtrack(node, path):
            if node:
                if not node.left and not node.right:
                    self.res.append([node.val] + path)
                    return
                backtrack(node.left, [node.val] + path)
                backtrack(node.right, [node.val] + path)
        backtrack(root, [])
        return check(self.res)
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

### 1022. Sum of Root To Leaf Binary Numbers

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

### 1457. Pseudo-Palindromic Paths in a Binary Tree

```python
class Solution:
    def pseudoPalindromicPaths (self, root: Optional[TreeNode]) -> int:
        p = [0] * 10
        def dfs(node):
            if not node:
                return 0
            p[node.val] ^= 1
            if node.left == node.right:
                res = 1 if sum(p) <= 1 else 0
            else:
                res = dfs(node.left) + dfs(node.right)
            p[node.val] ^= 1
            return res 
        return dfs(root)
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
            if node:
                if not node.left and not node.right:
                    self.res += node.val
                dfs(node.left)
                dfs(node.right)
        dfs(root)
        return self.res[k - 1]
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

### 971. Flip Binary Tree To Match Preorder Traversal

```python
class Solution:
    def flipMatchVoyage(self, root: Optional[TreeNode], voyage: List[int]) -> List[int]:
        stack = [root] 
        i = 0
        res = []
        while stack:
            node = stack.pop()
            if node.val != voyage[i]:
                return [-1]
            i += 1
            if node.left and node.left.val != voyage[i]:
                node.left, node.right = node.right, node.left 
                res.append(node.val)
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
        return res 
```

### 1430. Check If a String Is a Valid Sequence from Root to Leaves Path in a Binary Tree

```python
class Solution:
    def isValidSequence(self, root: Optional[TreeNode], arr: List[int]) -> bool:
        n = len(arr)
        def dfs(node, i):
            if i >= n or not node or node.val != arr[i]:
                return False
            elif not node.left and not node.right and i == n - 1:
                return True
            return dfs(node.left, i + 1) or dfs(node.right, i + 1)
        return dfs(root, 0)
```

### 545. Boundary of Binary Tree

```python
class Solution:
    def boundaryOfBinaryTree(self, root: Optional[TreeNode]) -> List[int]:
        if root and not root.left and not root.right:
            return [root.val]
        left, bottom, right = [], [], []
        def dfsLeft(node):
            if node and (node.left or node.right):
                left.append(node.val)
                if node.left:
                    dfsLeft(node.left)
                else:
                    dfsLeft(node.right)
        
        def dfsRight(node):
            if node and (node.left or node.right):
                right.append(node.val)
                if node.right:
                    dfsRight(node.right)
                else:
                    dfsRight(node.left)

        def dfsBottom(node):
            if node:
                if not node.left and not node.right:
                    bottom.append(node.val)
                dfsBottom(node.left)
                dfsBottom(node.right)

        dfsLeft(root.left)
        dfsBottom(root)
        dfsRight(root.right)
        return [root.val] + left + bottom + right[::-1]
```

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