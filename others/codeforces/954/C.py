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
  n, m = LI()
  s = S()
  indices = LI()
  c = S()
  d = Counter(indices)
  s_arr = list(s)
  c_arr = list(c)
  c_arr.sort()
  c_arr = deque(c_arr)
  t = []
  for k in sorted(d.keys()):
    val = d[k]
    while val > 1:
      t.append((k - 1, c_arr.pop()))
      val -= 1
    t.append((k - 1, c_arr.popleft()))
  for i, c in t:
    s_arr[i] = c
  arr.append(''.join(s_arr))
print('\n'.join(arr))