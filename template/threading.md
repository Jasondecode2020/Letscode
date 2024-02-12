### 1115. Print FooBar Alternately
```python
from threading import Lock
class FooBar:
    def __init__(self, n):
        self.n = n
        self.LockFoo = Lock()
        self.LockBar = Lock()
        self.LockBar.acquire()

    def foo(self, printFoo: 'Callable[[], None]') -> None:
        
        for i in range(self.n):
            self.LockFoo.acquire() # 查看A的状态 A默认是初始是解开的,执行结束后,加了锁.
            printFoo()
            self.LockBar.release()


    def bar(self, printBar: 'Callable[[], None]') -> None:
        
        for i in range(self.n):
            self.LockBar.acquire()  # 查看B的状态，B默认是锁上的，所以不会执行，只有当FOO执行结束后，B的状态是解开的，此时会执行。所以就是 foo--> bar
            printBar()
            self.LockFoo.release()
```