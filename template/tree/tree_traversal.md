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

