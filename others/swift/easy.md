# Question list

* [1. Two Sum](#1-two-sum)
* [9. Palindrome Number](#9-palindrome-number)
* [13. Roman to Integer](#13-roman-to-integer)
* [91. Number of 1 Bits](#91-number-of-1-bits)
* [125. Valid Palindrome](#125-valid-palindrome)
* [136. Single Number](#136-single-number)
* [169. Majority Element](#169-majority-element)
* [202. Happy Number](#202-happy-number)
* [217. Contains Duplicate](#217-contains-duplicate)
* [2404. Most Frequent Even Element](#2404-most-frequent-even-element)

### 1. Two Sum

```swift
class Solution {
    func twoSum(_ nums: [Int], _ target: Int) -> [Int] {
        var res = [Int]()
        var d = [Int: Int]()
        for i in 0 ..< nums.count {
            let ans = target - nums[i]
            if d.keys.contains(ans) {
                return [i, d[ans]!]
            }
            d[nums[i]] = i
        }
        return res
    }
}
```

### 9. Palindrome Number

```swift
class Solution {
    func isPalindrome(_ x: Int) -> Bool {
        if x < 0 { return false }
        var num = x
        var y = 0
        while num != 0 {
            y = y * 10 + num % 10
            num /= 10
        }
        return y == x
    }
}
```

### 13. Roman to Integer

```swift
class Solution {
    func romanToInt(_ s: String) -> Int {
        let d: [Character:Int] = [
            "I": 1, 
            "V": 5,
            "X": 10, 
            "L": 50, 
            "C": 100, 
            "D": 500, 
            "M": 1000
        ]
        var res = 0
        var prev = 0
        for c in s {
            let num = d[c]!
            res += num 
            if num > prev {
                res -= 2 * prev 
            }
            prev = num
        }
        return res 
    }
}
```

### 91. Number of 1 Bits

```swift
class Solution {
    func hammingWeight(_ n: Int) -> Int {
        var n = n
        var res = 0
        while n > 0 {
            res += n % 2
            n = n >> 1
        }
        return res
    }
}
```

### 125. Valid Palindrome

```swift
class Solution {
    func isPalindrome(_ s: String) -> Bool {
        let str = s.filter {
            $0.isLetter || $0.isNumber
        }.lowercased()
        return str == String(str.reversed())
    }
}
```

### 136. Single Number

```swift
class Solution {
    func singleNumber(_ nums: [Int]) -> Int {
        return nums.reduce(0, ^)
    }
}
```

### 169. Majority Element

```swift
class Solution {
    func majorityElement(_ nums: [Int]) -> Int {
        let L = nums.count
        var res = 0
        var d = [Int: Int]()
        for n in nums {
            if d.keys.contains(n) {
                d[n]! += 1
            } else {
                d[n] = 1
            }
        }
        for (k, v) in d {
            if v > L / 2 {
                res = k
            }
        }
        return res 
    }
}
```

### 202. Happy Number

```swift 
class Solution {
    func getSum(_ number: Int) -> Int {
        var sum = 0
        var num = number
        while num > 0 {
            let temp = num % 10
            sum += temp * temp
            num /= 10
        }
        return sum 
    }
    func isHappy(_ n: Int) -> Bool {
        var s = Set<Int>()
        var num = n 
        while true {
            let sum = self.getSum(num)
            if sum == 1 {
                return true
            }
            if s.contains(sum) {
                return false
            } else {
                s.insert(sum)
            }
            num = sum 
        }
    }
}
```

### 217. Contains Duplicate

```swift
class Solution {
    func containsDuplicate(_ nums: [Int]) -> Bool {
        var s = Set<Int>()
        for n in nums {
            if s.contains(n) {
                return true
            }
            s.insert(n)
        }
        return false
    }
}
```

### 2404. Most Frequent Even Element

```swift
class Solution {
    func mostFrequentEven(_ nums: [Int]) -> Int {
        var res = -1
        var cnt = 0
        var d = [Int: Int]()
        for n in nums {
            if d.keys.contains(n) {
                d[n]! += 1
            } else {
                d[n] = 1
            }
        }
        for (k, v) in d {
            if k % 2 == 0 {
                if v > cnt {
                    cnt = v 
                    res = k
                }
                if v == cnt {
                    res = min(res, k)
                }
            } 
        }
        return res
    }
}
```