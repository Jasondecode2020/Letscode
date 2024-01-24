## template 1: find minimum value
```python
def fn(arr):
    def check(x):
        # some code
        return BOOLEAN

    l = MINIMUM_POSSIBLE_ANSWER
    r = MAXIMUM_POSSIBLE_ANSWER
    while l <= r:
        m = l + (r - l) // 2
        if check(m):
            r = m - 1
        else:
            l = m + 1
    return l
```

## template 2: find maximum value

```python
def fn(arr):
    def check(x):
        # some code
        return BOOLEAN

    l = MINIMUM_POSSIBLE_ANSWER
    r = MAXIMUM_POSSIBLE_ANSWER
    while l <= r:
        m = (r - l) // 2
        if check(m):
            l = m + 1
        else:
            r = m - 1
    return r
```
* 275. H 指数 II
* 1283. 使结果不超过阈值的最小除数 1542
* 2187. 完成旅途的最少时间 1641
* 2226. 每个小孩最多能分到多少糖果 1646
* 1870. 准时到达的列车最小时速 1676
* 1011. 在 D 天内送达包裹的能力 1725
* 875. 爱吃香蕉的珂珂 1766
* 1898. 可移除字符的最大数目 1913
* 1482. 制作 m 束花所需的最少天数 1946
* 1642. 可以到达的最远建筑 1962
* 2861. 最大合金数 1981
* 2258. 逃离火灾 2347

### 最小化最大值
* 2064. 分配给商店的最多商品的最小值 1886
* 1760. 袋子里最少数目的球 1940
* 2439. 最小化数组中的最大值 1965
* 2560. 打家劫舍 IV 2081
* 778. 水位上升的泳池中游泳 2097 相当于最小化路径最大值
* 2616. 最小化数对的最大差值 2155
* 2513. 最小化两个数组中的最大值 2302
### 最大化最小值
* 1552. 两球之间的磁力 1920
* 2861. 最大合金数 1981
* 2517. 礼盒的最大甜蜜度 2021
* 2812. 找出最安全路径 2154
* 2528. 最大化城市的最小供电站数目 2236
### 第 K 小/大（部分题目也可以用堆解决）
* 378. 有序矩阵中第 K 小的元素
* 373. 查找和最小的 K 对数字
* 719. 找出第 K 小的数对距离
* 1439. 有序矩阵中的第 k 个最小数组和 2134
* 786. 第 K 个最小的素数分数 2169
* 2040. 两个有序数组的第 K 小乘积 2518
* 2386. 找出数组的第 K 大和 2648

三、问题难点
检查函数：检查该值是否复合答案，即定义check()函数

简单加和：最简单的线性扫描
1283. 使结果不超过阈值的最小除数
1300. 转变数组后最接近目标值的数组和 no
875. 爱吃香蕉的珂珂
剑指 Offer II 073. 狒狒吃香蕉
2064. 分配给商店的最多商品的最小值
1891. 割绳子
1870. 准时到达的列车最小时速
1760. 袋子里最少数目的球
历史标记：带有历史标记的线性扫描问题，稍微复杂了点
1552. 两球之间的磁力
1482. 制作 m 束花所需的最少天数
1011. 在 D 天内送达包裹的能力
滑动窗口：最复杂的扫描形式
LCP 12. 小张刷题计划
规律计算：找数学规律求解
1954. 收集足够苹果的最小花园周长
1802. 有界数组中指定下标处的最大值
暴力匹配：字符串问题
1062. 最长重复子串
1898. 可移除字符的最大数目
隐式最值：有的问题没有直接告诉是求边界最值，需要自己转换成最值问题，难度瞬间加大！

287. 寻找重复数
1648. 销售价值减少的颜色球
274. H 指数
400. 第 N 位数字
特殊二分

浮点二分：将浮点数转为非浮点数便可
2137. 通过倒水操作让所有的水桶所含水量相等
边界未知：定义最大数便可
2187. 完成旅途的最少时间

### 275. H-Index II

```python
class Solution:
    def hIndex(self, citations: List[int]) -> int:
        def check(threshold):
            res = 0
            for c in citations:
                if c >= threshold:
                    res += 1
            return res >= threshold
            
        l, r, res = 0, 1000, 0
        while l <= r:
            m = l + (r - l) // 2
            if check(m):
                res = m
                l = m + 1
            else:
                r = m - 1
        return res
```

### 1292. Maximum Side Length of a Square with Sum Less than or Equal to Threshold

```python
class Solution:
    def maxSideLength(self, mat: List[List[int]], threshold: int) -> int:
        R, C = len(mat), len(mat[0])
        dp = [[0] * (C + 1) for r in range(R + 1)]
        for r in range(1, R + 1):
            for c in range(1, C + 1):
                dp[r][c] = dp[r - 1][c] + dp[r][c - 1] - dp[r - 1][c - 1] + mat[r - 1][c - 1]
                
        def check(m):
            for r in range(m, R + 1):
                for c in range(m, C + 1):
                    s = dp[r][c] - dp[r - m][c] - dp[r][c - m] + dp[r - m][c - m]
                    if s <= threshold:
                        return True
            return False

        l, r, res = 0, 10 ** 9, 0
        while l <= r:
            m = l + (r - l) // 2
            if check(m):
                res = m 
                l = m + 1
            else:
                r = m - 1
        return res
```

### 2187. Minimum Time to Complete Trips

```python
class Solution:
    def minimumTime(self, time: List[int], totalTrips: int) -> int:
        def check(threshold):
            count = 0
            for t in time:
                count += threshold // t
            return count >= totalTrips
            
        l, r, res = 1, max(time) * totalTrips, 1
        while l <= r:
            m = l + (r - l) // 2
            if check(m):
                res = m
                r = m - 1
            else:
                l = m + 1
        return res
```

### 2226. Maximum Candies Allocated to K Children

```python
class Solution:
    def maximumCandies(self, candies: List[int], k: int) -> int:
        def check(threshold):
            count = 0
            if threshold == 0:
                return True
            for c in candies:
                count += c // threshold
            return count >= k

        l, r, res = 0, max(candies), 0
        while l <= r:
            m = l + (r - l) // 2
            if check(m):
                res = m 
                l = m + 1
            else:
                r = m - 1
        return res
```

### 1870. Minimum Speed to Arrive on Time

```python
class Solution:
    def minSpeedOnTime(self, dist: List[int], hour: float) -> int:
        n = len(dist)
        def check(threshold):
            count = 0
            for i, d in enumerate(dist):
                if i < n - 1:
                    count += ceil(d / threshold)
                else:
                    count += d / threshold
            return count <= hour

        l, r, res = 1, 10 ** 7, 0
        while l <= r:
            m = l + (r - l) // 2
            if check(m):
                res = m 
                r = m - 1
            else:
                l = m + 1
        return res if res != 0 else -1
```

### 1011. Capacity To Ship Packages Within D Days

```python
class Solution:
    def shipWithinDays(self, weights: List[int], days: int) -> int:
        def check(threshold):
            count, total = 0, 0
            for w in weights:
                total += w
                if total > threshold:
                    count += 1
                    total = w
            if total > threshold:
                return False
            return count + 1 <= days

        l, r, res = max(weights), sum(weights), max(weights)
        while l <= r:
            m = l + (r - l) // 2
            if check(m):
                res = m
                r = m - 1
            else:
                l = m + 1
        return res
```

### 1898. Maximum Number of Removable Characters

```python
class Solution:
    def maximumRemovals(self, s: str, p: str, removable: List[int]) -> int:
        def isSub(s, p):
            i, j = 0, 0
            while i < len(s) and j < len(p):
                if p[j] == s[i]:
                    i += 1
                    j += 1
                else:
                    i += 1
            return j == len(p)

        def check(threshold):
            res = ''
            seen = set(removable[:threshold])
            for i, c in enumerate(s):
                if i not in seen:
                    res += c 
            return isSub(res, p)

        l, r, res = 0, len(removable), 0
        while l <= r:
            m = l + (r - l) // 2
            if check(m):
                res = m 
                l = m + 1
            else:
                r = m - 1
        return res
```