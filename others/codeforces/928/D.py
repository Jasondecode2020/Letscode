q = int(input())
mask = 2 ** 31 - 1
for _ in range(q):
    n = int(input())
    count = 0
    nums = list(map(int, input().split()))
    nums.sort()
    l, r = 0, len(nums) - 1
    while l < r:
        if nums[l] + nums[r] < mask:
            l += 1
        elif nums[l] + nums[r] > mask:
            r -= 1
        else:
            l += 1
            r -= 1
            count += 1
    print(n - count)
