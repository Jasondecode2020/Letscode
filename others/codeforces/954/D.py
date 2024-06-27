standard_input, packages = 1, 1

if 1:
  if standard_input:
    import io, os, sys
    input = lambda: sys.stdin.readline().strip()

    import math
    inf = math.inf

    def S(): # string
      return input()
    
    def I(): # int
      return int(input())

    def II(): # iterate int from a list
      return map(int, input().split())

    def LS(): # list of string
      return list(input().split())

    def LI(): # list of int
      return list(map(int, input().split()))

    def LF(): # list of float
      return list(map(float, input().split()))

    def M10(): # iterate of 1 to 0
      return map(lambda x: int(x) - 1, input().split())

    def L10(): # list of 1 to 0
      return list(map(lambda x: int(x) - 1, input().split()))

  if packages:
    import random
    import bisect
    import typing
    from collections import Counter, defaultdict, deque
    from copy import deepcopy
    from functools import cmp_to_key, lru_cache, reduce
    from heapq import merge, heapify, heappop, heappush, heappushpop, nlargest, nsmallest
    from itertools import accumulate, combinations, permutations, count, product
    from operator import add, iand, ior, itemgetter, mul, xor
    from string import ascii_lowercase, ascii_uppercase, ascii_letters

q = I()
arr = []
for _ in range(q):
  n = I()
  s = LS()
  s = ''.join(s)
  if len(s) == 2:
    arr.append(str(int(s)))
  else:
    zero = s.count('0')
    if zero >= 2:
      arr.append('0')
    elif zero == 1:
      if len(s) >= 4:
        arr.append('0')
      else:
        if s[0] == '0' or s[-1] == '0':
          arr.append('0')
        else:
          mn = min(int(s[0]) * int(s[1:]), int(s[0]) + int(s[1:]))
          arr.append(str(mn))
    elif zero == 0:
      ans = inf
      for k in range(n - 1):
        j = 0
        a = []
        while j < n:
          if j == k:
            a.append(s[j: j + 2])
            j += 2
          else:
            a.append(s[j])
            j += 1
        a = [int(i) for i in a]
        res = 0
        for num in a:
          if num != 1:
            res += num 
        ans = min(ans, res)
      arr.append(str(ans))
        
print('\n'.join(arr))
    
  