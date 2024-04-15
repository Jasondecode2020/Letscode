### 1

输入描述：
输入包括两个正整数a,b(1 <= a, b <= 1000),输入数据包括多组。
输出描述：
输出a+b的结果

示例1
输入例子：
1 5
10 20
输出例子：
6
30

```python
# import sys

# for line in sys.stdin:
#     a = line.split()
#     print(int(a[0]) + int(a[1]))


# import sys

# for line in sys.stdin:
#     a, b = line.split(' ')
#     print(int(a) + int(b))
# while True:
#     try:
#         input_list = list(map(int, input().split()))
#         print(input_list[0] + input_list[1])
#     except:
#         break
while True:
    try:
        a, b = input().split()
        print(int(a) + int(b))
    except:
        break
```

### 2

输入描述：
输入第一行包括一个数据组数t(1 <= t <= 100)
接下来每行包括两个正整数a,b(1 <= a, b <= 1000)
输出描述：
输出a+b的结果
示例1
输入例子：
2
1 5
10 20
输出例子：
6
30

```python
n = int(input())
for i in range(n):
    a, b = input().split()
    print(int(a) + int(b))
```

### 3 

输入描述：
输入包括两个正整数a,b(1 <= a, b <= 10^9),输入数据有多组, 如果输入为0 0则结束输入
输出描述：
输出a+b的结果
示例1
输入例子：
1 5
10 20
0 0
输出例子：
6
30

```python
while True:
    a, b = input().split()
    if a == b == '0':
        break
    print(int(a) + int(b))
```

### 4 

输入描述：
输入数据包括多组。
每组数据一行,每行的第一个整数为整数的个数n(1 <= n <= 100), n为0的时候结束输入。
接下来n个正整数,即需要求和的每个正整数。
输出描述：
每组数据输出求和的结果
示例1
输入例子：
4 1 2 3 4
5 1 2 3 4 5
0
输出例子：
10
15

```python
while True:
    arr = list(map(int, input().split()))
    if arr[0] == 0:
        break 
    print(sum(arr[1: ]))
```

### 5

输入描述：
输入的第一行包括一个正整数t(1 <= t <= 100), 表示数据组数。
接下来t行, 每行一组数据。
每行的第一个整数为整数的个数n(1 <= n <= 100)。
接下来n个正整数, 即需要求和的每个正整数。
输出描述：
每组数据输出求和的结果
示例1
输入例子：
2
4 1 2 3 4
5 1 2 3 4 5
输出例子：
10
15

```python
n = int(input())
for i in range(n):
    arr = list(map(int,input().split()))
    print(sum(arr[1 :]))
```

### 6 

输入描述：
输入数据有多组, 每行表示一组输入数据。
每行的第一个整数为整数的个数n(1 <= n <= 100)。
接下来n个正整数, 即需要求和的每个正整数。
输出描述：
每组数据输出求和的结果
示例1
输入例子：
4 1 2 3 4
5 1 2 3 4 5
输出例子：
10
15

```python
while True:
    try:
        arr = list(map(int, input().split()))
        print(sum(arr[1:]))
    except:
        break 
```

### 7 

输入描述：
输入数据有多组, 每行表示一组输入数据。

每行不定有n个整数，空格隔开。(1 <= n <= 100)。
输出描述：
每组数据输出求和的结果
示例1
输入例子：
1 2 3
4 5
0 0 0 0 0
输出例子：
6
9
0

```python
while True:
    try:
        arr = list(map(int, input().split()))
        print(sum(arr))
    except:
        break 
```

### 8

链接：https://ac.nowcoder.com/acm/contest/5657/H
来源：牛客网

输入描述:
输入有两行，第一行n

第二行是n个字符串，字符串之间用空格隔开
输出描述:
输出一行排序后的字符串，空格隔开，无结尾空格
示例1
输入
复制
5
c d a bb e
输出
复制
a bb c d e

```python
n = int(input())
arr = input().split()
arr.sort()
print(' '.join(arr))
```

### 9 

链接：https://ac.nowcoder.com/acm/contest/5657/I
来源：牛客网

输入描述:
多个测试用例，每个测试用例一行。

每行通过空格隔开，有n个字符，n＜100
输出描述:
对于每组测试用例，输出一行排序过的字符串，每个字符串通过空格隔开
示例1
输入
复制
a c bb
f dddd
nowcoder
输出
复制
a bb c
dddd f
nowcoder

```python
while True:
    try:
        arr = input().split()
        arr.sort()
        print(' '.join(arr))
    except:
        break
```

### 10 

链接：https://ac.nowcoder.com/acm/contest/5657/J
来源：牛客网

输入描述:
多个测试用例，每个测试用例一行。
每行通过,隔开，有n个字符，n＜100
输出描述:
对于每组用例输出一行排序后的字符串，用','隔开，无结尾空格
示例1
输入
复制
a,c,bb
f,dddd
nowcoder
输出
复制
a,bb,c
dddd,f
nowcoder

```python
while True:
    try:
        arr = input().split(',')
        arr.sort()
        print(','.join(arr))
    except:
        break 
```