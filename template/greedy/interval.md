### 986. Interval List Intersections

```python
class Solution:
    def intervalIntersection(self, firstList: List[List[int]], secondList: List[List[int]]) -> List[List[int]]:
        firstList.extend(secondList)
        firstList.sort()
        res = []
        start, end = firstList[0]
        for s, e in firstList[1:]:
            if s <= end: 
                res.append([s, min(end, e)])
            end = max(end, e)
        return res
```

### 1229. Meeting Scheduler

```python
class Solution:
    def minAvailableDuration(self, slots1: List[List[int]], slots2: List[List[int]], duration: int) -> List[int]:
        slots1.extend(slots2)
        slots1.sort()
        start, end = slots1[0]
        for s, e in slots1[1:]:
            if s <= end: 
                if min(end, e) - s >= duration:
                    return [s, s + duration]
            end = max(end, e)
        return []
```