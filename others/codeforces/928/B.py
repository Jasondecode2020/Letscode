q = int(input())
for i in range(q):
  n = int(input())
  s = set()
  for j in range(n):
    row = input()
    if row.count('1') != 0:
      s.add(row.count('1'))
  if len(s) == 1:
    print('SQUARE')
  else:
    print('TRIANGLE')