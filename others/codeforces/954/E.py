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
  a = LI()
  d = defaultdict(list)
  res = []
  ans1 = ans2 = 0
  flag = False
  a.sort()
  for num in a:
    d[num % k].append(num)
  odd_count = 0
  for nums in d.values():
    if len(nums) % 2 == 0:
      for i in range(0, len(nums), 2):
        ans1 += (nums[i + 1] - nums[i]) // k 
    else:
      ans2 = inf
      odd_count += 1
      if odd_count >= 2:
        flag = True
        arr.append(str(-1))
        break
      L = len(nums)
      pre, suf = [0] , deque([0])
      for i in range(1, L, 2):
        pre.append(nums[i] - nums[i - 1] + pre[-1])
      for i in range(L - 1, 0, -2):
        suf.appendleft(nums[i] - nums[i - 1] + suf[0])
      for a, b in zip(pre, suf):
        ans2 = min(ans2, (a + b) // k)
  
  if not flag:
    arr.append(str(ans1 + ans2))
print('\n'.join(arr))