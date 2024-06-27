from collections import defaultdict

q = int(input())
mask = 2 ** 31 - 1
for _ in range(q):
    n = int(input())
    count = 0
    nums = list(map(int, input().split()))
    d = defaultdict(int)
    for x in nums:
        d[x] += 1
    s = set(d.keys())
    for num in s:
        x_or_num = num ^ mask 
        if x_or_num in d:
            count += min(d[num], d[x_or_num])
            del d[num]
    print(n - count)
