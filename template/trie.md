## template

```python
'''
Leetcode 208
'''
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
        return cur.endOfWord

    def startsWith(self, prefix: str) -> bool:
        cur = self.root 
        for c in prefix:
            if c not in cur.children:
                return False
            cur = cur.children[c]
        return True
```

### Trie(12)

* [14. Longest Common Prefix](#14-Longest-Common-Prefix)
* [208. Implement Trie](#208-Implement-Trie)
* [3597. Partition String](#3597-partition-string)
* [648. Replace Words](#648-replace-words)
* [720. Longest Word in Dictionary](#720-Longest-Word-in-Dictionary)

* [2416. Sum of Prefix Scores of Strings](#2416-sum-of-prefix-scores-of-strings)
* [677. Map Sum Pairs](#677-map-sum-pairs)
* [1268. Search Suggestions System](#1268-search-suggestions-system)
* [1804. Implement Trie II](#1804-Implement-Trie-II)
* [211. Design Add and Search Words Data Structure](#211-design-add-and-search-words-data-structure)

* [212. Word Search II](#212-word-search-ii)
* [676. Implement Magic Dictionary](#676-implement-magic-dictionary)
* [820. Short Encoding of Words](#820-short-encoding-of-words)
* [1032. Stream of Characters](#1032-stream-of-characters)
* [745. Prefix and Suffix Search](#745-prefix-and-suffix-search)

* [425. Word Squares](#425-word-squares)

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

### 208. Implement Trie

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
        return cur.endOfWord

    def startsWith(self, prefix: str) -> bool:
        cur = self.root 
        for c in prefix:
            if c not in cur.children:
                return False
            cur = cur.children[c]
        return True
```

### 3597. Partition String 

```python
class Solution:
    def partitionString(self, s: str) -> List[str]:
        hash_set = set()
        lst = []
        res = ''
        for c in s:
            res += c 
            if not res in hash_set:
                hash_set.add(res)
                lst.append(res)
                res = ''
        return lst 

# trie
class TrieNode:
    def __init__(self):
        self.children = {}

class Trie:
    def __init__(self):
        self.root = TrieNode()
        self.res = []

    def insert(self, word):
        cur = self.root 
        left = 0
        for i, c in enumerate(word):
            if c not in cur.children:
                cur.children[c] = TrieNode()
                self.res.append(word[left: i + 1])
                left = i + 1
                cur = self.root 
            else:
                cur = cur.children[c]

class Solution:
    def partitionString(self, s: str) -> List[str]:
        t = Trie()
        t.insert(s)
        return t.res
```

```java
class Solution {
    public List<String> partitionString(String s) {
        Set seen = new HashSet<>();
        List res = new ArrayList<>();
        StringBuilder cur = new StringBuilder();
        for (char c: s.toCharArray()) {
            cur.append(c);
            String str = cur.toString();
            if (!seen.contains(str)) {
                seen.add(str);
                res.add(str);
                cur = new StringBuilder();

            }
        }
        return res;
    }
}
// trie
class TrieNode {
    Map<Character, TrieNode> children = new HashMap<>();
}

class Trie {
    private TrieNode root = new TrieNode();
    private List<String> result = new ArrayList<>();
    
    public void insert(String word) {
        TrieNode current = root;
        int left = 0;
        
        for (int i = 0; i < word.length(); i++) {
            char c = word.charAt(i);
            
            if (!current.children.containsKey(c)) {
                current.children.put(c, new TrieNode());
                result.add(word.substring(left, i + 1));
                left = i + 1;
                current = root;
            } else {
                current = current.children.get(c);
            }
        }
    }
    
    public List<String> getResult() {
        return result;
    }
}

class Solution {
    
    public List<String> partitionString(String s) {
        Trie trie = new Trie();
        trie.insert(s);
        return trie.getResult();
    }
}
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

    def find_prefix(self, word):
        cur = self.root
        for i in range(len(word)):
            if cur.endOfWord:
                return word[: i]
            if word[i] in cur.children:
                cur = cur.children[word[i]]
            else:
                break 
        return word

class Solution:
    def replaceWords(self, dictionary: List[str], sentence: str) -> str:
        sentence = sentence.split(' ')
        t = Trie()
        for word in dictionary:
            t.insert(word)

        for i, word in enumerate(sentence):
            sentence[i] = t.find_prefix(word)
        return ' '.join(sentence)
```

- set

```python
class Solution:
    def replaceWords(self, dictionary: List[str], sentence: str) -> str:
        dictionary = set(dictionary)
        sentence = sentence.split(' ')
        for i, word in enumerate(sentence):
            for j in range(len(word)):
                if word[:j + 1] in dictionary:
                    sentence[i] = word[: j + 1]
                    break
        return ' '.join(sentence)
```

### 720. Longest Word in Dictionary

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
            cur = cur.children[c]
            if not cur.endOfWord:
                return False
        return True

class Solution:
    def longestWord(self, words: List[str]) -> str:
        words.sort(key=lambda x: (-len(x), x))
        t = Trie()
        for word in words:
            t.insert(word)

        res = ''
        for word in words:
            if t.search(word):
                res = word 
                break
        return res
```

### 2416. Sum of Prefix Scores of Strings

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.count = 0

class Trie:

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        cur = self.root 
        for c in word:
            if c not in cur.children:
                cur.children[c] = TrieNode()
            cur = cur.children[c]
            cur.count += 1

    def search(self, word: str) -> bool:
        cur = self.root 
        res = 0
        for c in word:
            if c not in cur.children:
                return False
            cur = cur.children[c]
            res += cur.count 
        return res

class Solution:
    def sumPrefixScores(self, words: List[str]) -> List[int]:
        t = Trie()
        for word in words:
            t.insert(word)

        res = []
        cur = t.root 
        for word in words:
            res.append(t.search(word))
        return res 
```

### 677. Map Sum Pairs

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.count = 0

class Trie:

    def __init__(self):
        self.root = TrieNode()
        self.d = {}

    def insert(self, word, val):
        origin = val 
        if word in self.d:
            val = val - self.d[word]
        cur = self.root 
        for c in word:
            if c not in cur.children:
                cur.children[c] = TrieNode()
            cur = cur.children[c]
            cur.count += val 
        self.d[word] = origin

    def search(self, word: str) -> bool:
        cur = self.root 
        for c in word:
            if c not in cur.children:
                return 0
            cur = cur.children[c]
        return cur.count

class MapSum:

    def __init__(self):
        self.t = Trie()

    def insert(self, key: str, val: int) -> None:
        self.t.insert(key, val)

    def sum(self, prefix: str) -> int:
        return self.t.search(prefix)
```

### 1268. Search Suggestions System

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.words = []

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        cur = self.root 
        for c in word:
            if c not in cur.children:
                cur.children[c] = TrieNode()
            cur = cur.children[c]
            cur.words.append(word)
            cur.words.sort()
            if len(cur.words) > 3:
                cur.words.pop()

class Solution:
    def suggestedProducts(self, products: List[str], searchWord: str) -> List[List[str]]:
        t = Trie()
        for product in products:
            t.insert(product)
        cur = t.root 
        res = [[] for _ in range(len(searchWord))]
        for i, c in enumerate(searchWord):
            if c not in cur.children:
                break
            cur = cur.children[c]
            res[i] = cur.words
        return res 
```

### 1804. Implement Trie II

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.prefix_count = 0
        self.word_count = 0

class Trie:

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        cur = self.root 
        for c in word:
            if c not in cur.children:
                cur.children[c] = TrieNode()
            cur = cur.children[c]
            cur.prefix_count += 1
        cur.word_count += 1

    def countWordsEqualTo(self, word: str) -> int:
        cur = self.root 
        for c in word:
            if c not in cur.children:
                return 0
            cur = cur.children[c]
        return cur.word_count 

    def countWordsStartingWith(self, prefix: str) -> int:
        cur = self.root 
        for c in prefix:
            if c not in cur.children:
                return 0
            cur = cur.children[c]
        return cur.prefix_count

    def erase(self, word: str) -> None:
        cur = self.root 
        for c in word:
            if c not in cur.children:
                break
            cur = cur.children[c]
            cur.prefix_count -= 1
        cur.word_count -= 1
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
                if 0 <= row < R and 0 <= col < C and board[row][col] != '#':
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

### 820. Short Encoding of Words

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

    def tialOfWord(self, word):
        cur = self.root
        for c in word:
            cur = cur.children[c]
        return len(cur.children) == 0

class Solution:
    def minimumLengthEncoding(self, words: List[str]) -> int:
        t = Trie()
        res = 0
        words = set(words)
        for word in words:
            t.insert(word[::-1])
        for word in words:
            if t.tialOfWord(word[::-1]):
                res += len(word) + 1
        return res 
```

### 1032. Stream of Characters

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

class StreamChecker:

    def __init__(self, words: List[str]):
        self.words = set(words)
        self.t = Trie()
        self.s = ''
        for word in words:
            self.t.insert(word[::-1])

    def query(self, letter: str) -> bool:
        self.s += letter
        if len(self.s) > 200:
            reversed_str = self.s[-200:][::-1]
        else:
            reversed_str = self.s[::-1]
        for i in range(1, len(reversed_str) + 1):
            if self.t.search(reversed_str[:i]):
                return True
        return False



# Your StreamChecker object will be instantiated and called as such:
# obj = StreamChecker(words)
# param_1 = obj.query(letter)
```

### 745. Prefix and Suffix Search

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.endOfWord = False
        self.index = set()

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word, i):
        cur = self.root 
        for c in word:
            if c not in cur.children:
                cur.children[c] = TrieNode()
                cur.index.add(i)
            cur = cur.children[c]
            cur.index.add(i)
        cur.endOfWord = True 

    def prefix(self, word):
        cur = self.root 
        for c in word:
            if c not in cur.children:
                return -1
            cur = cur.children[c]
        return cur.index

class WordFilter:

    def __init__(self, words: List[str]):
        self.t1, self.t2 = Trie(), Trie()
        d = defaultdict(int)
        for i, word in enumerate(words):
            d[word] = i
        for word in d.keys():
            self.t1.insert(word, d[word])
            self.t2.insert(word[::-1], d[word])

    def f(self, pref: str, suff: str) -> int:
        preIndexSet, sufIndexSet = self.t1.prefix(pref), self.t2.prefix(suff[::-1])
        if preIndexSet == -1 or sufIndexSet == -1:
            return -1
        res = -inf
        for cnt in preIndexSet:
            if cnt in sufIndexSet:
                res = max(res, cnt)
        return res if res != -inf else -1
        

# Your WordFilter object will be instantiated and called as such:
# obj = WordFilter(words)
# param_1 = obj.f(pref,suff)

```
### 425. Word Squares

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.words = []

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        cur = self.root 
        for c in word:
            if c not in cur.children:
                cur.children[c] = TrieNode()
            cur = cur.children[c]
            cur.words.append(word)
        
    def search(self, prefix):
        cur = self.root 
        for c in prefix:
            if c not in cur.children:
                return []
            cur = cur.children[c]
        return cur.words

class Solution:
    def wordSquares(self, words: List[str]) -> List[List[str]]:
        t = Trie()
        for word in words:
            t.insert(word)

        res = []
        n = len(words[0])
        def backtrack(i, cur_words):
            if i == n:
                res.append(cur_words)
                return 

            prefix = ''.join(word[i] for word in cur_words)
            words = t.search(prefix)
            for word in words:
                backtrack(i + 1, cur_words + [word])
        
        for word in words:
            backtrack(1, [word])
        return res
```
