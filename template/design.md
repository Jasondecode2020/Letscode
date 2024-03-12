## design

* [705. Design HashSet](#705. Design HashSet)
* [560. Subarray Sum Equals K](#560-Subarray-Sum-Equals-K)


### 705. Design HashSet

```python
class MyHashSet:

    def __init__(self):
        self.buckets = 1009
        self.table = [[] for _ in range(self.buckets)]

    def hash(self, key):
        return key % self.buckets
    
    def add(self, key):
        hashkey = self.hash(key)
        if key in self.table[hashkey]:
            return
        self.table[hashkey].append(key)
        
    def remove(self, key):
        hashkey = self.hash(key)
        if key not in self.table[hashkey]:
            return
        self.table[hashkey].remove(key)

    def contains(self, key):
        hashkey = self.hash(key)
        return key in self.table[hashkey]
```

### 2043. Simple Bank System

```python
class Bank:

    def __init__(self, balance: List[int]):
        self.balance = balance

    def transfer(self, account1: int, account2: int, money: int) -> bool:
        if account1 - 1 < len(self.balance) and self.balance[account1 - 1] >= money:
            if account2 - 1 < len(self.balance):
                self.balance[account1 - 1] -= money
                self.balance[account2 - 1] += money
                return True
        return False
    def deposit(self, account: int, money: int) -> bool:
        if account - 1 < len(self.balance):
            self.balance[account - 1] += money
            return True
        return False

    def withdraw(self, account: int, money: int) -> bool:
        if account - 1 < len(self.balance) and self.balance[account - 1] >= money:
            self.balance[account - 1] -= money
            return True
        return False


# Your Bank object will be instantiated and called as such:
# obj = Bank(balance)
# param_1 = obj.transfer(account1,account2,money)
# param_2 = obj.deposit(account,money)
# param_3 = obj.withdraw(account,money)
```