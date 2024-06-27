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
  n, k = LI()
  a1 = list(range(1, n + 1))
  a2 = a1[::-1]
  mx = sum(abs(a - b) for a, b in zip(a1, a2))
  if k > mx or k % 2:
    arr.append('NO')
  else:
    l, r = 0, len(a1) - 1
    res = 0
    while l < r:
      ans = abs(a1[l] - a1[r]) + abs(a1[r] - a1[l])
      if k - ans > 0:
        a1[l], a1[r] = a1[r], a1[l]
        l += 1
        r -= 1
        k -= ans
      elif k - ans < 0:
        r -= 1
      else:
        a1[l], a1[r] = a1[r], a1[l]
        break 
    arr.append('YES')
    arr.append(' '.join(str(i) for i in a1))
print('\n'.join(arr))