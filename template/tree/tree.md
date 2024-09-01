# tree

## 1 Traversal

* [144. Binary Tree Preorder Traversal](#144-binary-tree-preorder-traversal)
* [94. Binary Tree Inorder Traversal](#94-binary-tree-inorder-traversal)
* [145. Binary Tree Postorder Traversal](#145-binary-tree-postorder-traversal)
* [872. Leaf-Similar Trees](#872-leaf-similar-trees)
* [404. Sum of Left Leaves](#404-sum-of-left-leaves)
* [671. Second Minimum Node In a Binary Tree](#671-second-minimum-node-in-a-binary-tree)
* [1469. Find All The Lonely Nodes](#1469-find-all-the-lonely-nodes)
* [1214. Two Sum BSTs](#1214-two-sum-bsts)


## 2 top down DFS

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

## bottom up DFS

* [298. Binary Tree Longest Consecutive Sequence]()
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