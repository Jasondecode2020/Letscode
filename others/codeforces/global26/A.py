from collections import Counter
q = int(input())
for i in range(q):
  n = int(input())
  arr = [int(i) for i in input().split()]
  c = Counter(arr)
  if len(c) == 1:
    print('NO')
  else:
    mx = c.most_common()[0][0]
    print('YES')
    flag = True
    ans = []
    for n in arr:
      if flag and n == mx:
        ans.append('B')
        flag = False
      else:
        ans.append('R')
    print(''.join(ans))