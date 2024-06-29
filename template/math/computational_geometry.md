### 2613. Beautiful Pairs

```python
class Solution:
    def beautifulPair(self, nums1: List[int], nums2: List[int]) -> List[int]:
        def m_distance(x1, y1, x2, y2):
            return abs(x1 - x2) + abs(y1 - y2)

        def dfs(l, r):
            if l >= r:
                return inf, -1, -1
          
            m = l + (r - l) // 2
            x = points[m][0]
            d_l, l_i, l_j = dfs(l, m)
            d_r, r_i, r_j = dfs(m + 1, r)
            if d_l > d_r or (d_l == d_r and (l_i > r_i or (l_i == r_i and l_j > r_j))):
                d_l, l_i, l_j = d_r, r_i, r_j
        
            candidates = [p for p in points[l : r + 1] if abs(p[0] - x) <= d_l]
            candidates.sort(key=lambda p: p[1])
            for i in range(len(candidates)):
                for j in range(i + 1, len(candidates)):
                    if candidates[j][1] - candidates[i][1] > d_l:
                        break
                    index_i, index_j = sorted([candidates[i][2], candidates[j][2]])
                    dist = m_distance(candidates[i][0], candidates[i][1], candidates[j][0], candidates[j][1])
                    if dist < d_l or (dist == d_l and (index_i < l_i or (index_i == l_i and index_j < l_j))):
                        d_l, l_i, l_j = dist, index_i, index_j
            return d_l, l_i, l_j

        d = defaultdict(list)
        for i, (x, y) in enumerate(zip(nums1, nums2)):
            d[(x, y)].append(i)
      
        points = []
        for i, (x, y) in enumerate(zip(nums1, nums2)):
            if len(d[(x, y)]) > 1:
                return [i, d[(x, y)][1]]
            points.append((x, y, i))
      
        points.sort()
        dist, index1, index2 = dfs(0, len(points) - 1)
        return [index1, index2]
```