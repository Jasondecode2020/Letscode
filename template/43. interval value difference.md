### 121. Best Time to Buy and Sell Stock

res = current value - previous min value

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        res, lowest = 0, prices[0]
        for j in range(1, len(prices)):
            i = j - 1
            lowest = min(lowest, prices[i])
            res = max(res, prices[j] - lowest)
        return res
```

### 2903. Find Indices With Index and Value Difference I
### 2905. Find Indices With Index and Value Difference II

```python
class Solution:
    def findIndices(self, nums: List[int], indexDifference: int, valueDifference: int) -> List[int]:
        minIdx, maxIdx = 0, 0
        for j in range(indexDifference, len(nums)):
            i = j - indexDifference
            if nums[i] > nums[maxIdx]:
                maxIdx = i
            if nums[i] < nums[minIdx]:
                minIdx = i
            if nums[j] - nums[minIdx] >= valueDifference:
                return [minIdx, j]
            if nums[maxIdx] - nums[j] >= valueDifference:
                return [maxIdx, j]
        return [-1, -1]
```