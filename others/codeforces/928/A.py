from collections import Counter
n = int(input())
for i in range(n):
  s = input()
  c = Counter(s)
  if c['A'] > c['B']:
    print('A')
  else:
    print('B')