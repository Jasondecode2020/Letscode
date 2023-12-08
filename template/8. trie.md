## template

```python
class TrieNode: # has children and endOfWord
    def __init__(self):
        self.children = {} # can be 26 for English letters
        self.endOfWord = False # check if word ends with a letter

class Trie: # a tree like data structure to solve prefix problems in string
    def __init__(self): # init the node
        self.root = TrieNode()

    def insert(self, word: str) -> None: # insert a word inside a Trie
        cur = self.root
        for c in word:
            if c not in cur.children:
                cur.children[c] = TrieNode()
            cur = cur.children[c]
        cur.endOfWord = True

    def search(self, word: str) -> bool: # check if a word inside trie
        cur = self.root
        for c in word:
            if c not in cur.children:
                return False
            cur = cur.children[c]
        return cur.endOfWord == True

    def startsWith(self, prefix: str) -> bool: # check if a prefix inside a word
        cur = self.root
        for c in prefix:
            if c not in cur.children:
                return False
            cur = cur.children[c]
        return True
        
class Solution:
    def fn(self, strs: List[str]) -> str:
        trie = Trie()
        for word in strs:
            trie.insert(word) # insert all words to build trie for search
            
        root = trie.root # start search in a trie
        res = ''
        while root:
            # do something
        return res
```

### 14. Longest Common Prefix

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.endOfWord = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        cur = self.root
        for c in word:
            if c not in cur.children:
                cur.children[c] = TrieNode()
            cur = cur.children[c]
        cur.endOfWord = True

class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        t = Trie()
        for word in strs:
            t.insert(word)

        res = ''
        cur = t.root
        while cur:
            if len(cur.children) > 1 or cur.endOfWord: # if a word ends or has more than one children
                break
            c = list(cur.children.keys())[0]
            cur = cur.children[c]
            res += c
        return res
```

### 139. Word Break

- prefix idea: dp[i] means if dp[:i] is combined by the wordDict or not

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

- brute force

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
    
    def find(self, word):
        cur = self.root
        for c in word:
            if c not in cur.children:
                return False
            cur = cur.children[c]
        return cur.endOfWord == True

    def search(self, word: str) -> bool:
        word_set = ['']
        for c in word:
            if c == '.':
                temp = []
                for w in word_set:
                    for letter in ascii_lowercase:
                        temp.append(w + letter)
                word_set = temp
            else:
                word_set = [w + c for w in word_set]
        for w in word_set:
            if self.find(w):
                return True
        return False
```

- dfs

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
        def find(idx, root):
            if not root:
                return False
            if idx == len(word):
                return root.endOfWord
            if word[idx] != '.':
                return root and find(idx + 1, root.children.get(word[idx], None))
            else:
                for child in root.children.values():
                    if find(idx + 1, child):
                        return True
            return False
        return find(0, self.root)
```

### 212. Word Search II

```python
class TrieNode: # has children and endOfWord
    def __init__(self):
        self.children = {} # can be 26 for English letters
        self.endOfWord = False # check if word ends with a letter

class Trie: # a tree like data structure to solve prefix problems in string
    def __init__(self): # init the node
        self.root = TrieNode()

    def insert(self, word: str) -> None: # insert a word inside a Trie
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

### 648. Replace Words

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.endOfWord = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        cur = self.root
        for c in word:
            if c not in cur.children:
                cur.children[c] = TrieNode()
            cur = cur.children[c]
        cur.endOfWord = True

    def search(self, word):
        cur = self.root
        for c in word:
            if c not in cur.children:
                return False
            cur = cur.children[c]
        return cur.endOfWord == True

class Solution:
    def replaceWords(self, dictionary: List[str], sentence: str) -> str:
        t = Trie()
        for word in dictionary:
            t.insert(word)

        words = sentence.split(' ')
        dictionary = set(dictionary)
        for i, w in enumerate(words):
            for j in range(len(w)):
                if t.search(w[:j+1]):
                    words[i] = w[:j+1]
                    break
        return ' '.join(words)
```

- set

```python
class Solution:
    def replaceWords(self, dictionary: List[str], sentence: str) -> str:
        words = sentence.split(' ')
        dictionary = set(dictionary)
        for i, w in enumerate(words):
            for j in range(len(w)):
                if w[:j+1] in dictionary:
                    words[i] = w[:j+1]
                    break
        return ' '.join(words)
```

### 676. Implement Magic Dictionary

```python
class TrieNode: # has children and endOfWord
    def __init__(self):
        self.children = {} # can be 26 for English letters
        self.endOfWord = False # check if word ends with a letter

class MagicDictionary:

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None: # insert a word inside a Trie
        cur = self.root
        for c in word:
            if c not in cur.children:
                cur.children[c] = TrieNode()
            cur = cur.children[c]
        cur.endOfWord = True

    def buildDict(self, dictionary: List[str]) -> None:
        for word in dictionary:
            self.insert(word) # insert all words to build trie for search

    def search(self, searchWord: str) -> bool:
        def dfs(idx, root, count):
            if idx == len(searchWord):
                if count == 1:
                    return root.endOfWord
                return False
            ch = searchWord[idx]
            if ch in root.children:
                if dfs(idx + 1, root.children[ch], count):
                    return True
            if count == 0:
                for child in root.children:
                    if child != ch:
                        if dfs(idx + 1, root.children[child], count + 1):
                            return True
            return False
        return dfs(0, self.root, 0)
```

### 720. Longest Word in Dictionary

```python
class TrieNode: # has children and endOfWord
    def __init__(self):
        self.children = {} # can be 26 for English letters
        self.endOfWord = False # check if word ends with a letter

class Trie: # a tree like data structure to solve prefix problems in string
    def __init__(self): # init the node
        self.root = TrieNode()

    def insert(self, word: str) -> None: # insert a word inside a Trie
        cur = self.root
        for c in word:
            if c not in cur.children:
                cur.children[c] = TrieNode()
            cur = cur.children[c]
        cur.endOfWord = True

    def search(self, word: str) -> bool: # check if a word inside trie
        cur = self.root
        for c in word:
            if c not in cur.children:
                return False
            cur = cur.children[c]
            if cur.endOfWord == False:
                return False
        return True

class Solution:
    def longestWord(self, words: List[str]) -> str:
        t = Trie()
        for word in words:
            t.insert(word)

        res = []
        for w in words:
            if t.search(w):
                res.append(w)
        if not res:
            return ''
        res.sort(key = lambda x: (-len(x), x))
        return res[0]
```