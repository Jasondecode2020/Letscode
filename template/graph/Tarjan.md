## template Tarjan's algorithm

### 1192. Critical Connections in a Network

```python
class Solution:
    def criticalConnections(self, n: int, connections: List[List[int]]) -> List[List[int]]:
        g = defaultdict(list)
        for u, v in connections:
            g[u].append(v)
            g[v].append(u)

        ids, res = [-1] * n, []
        def dfs(cur_node, cur_id, par):
            ids[cur_node] = cur_id
            for nei in g[cur_node]:
                if nei == par:
                    continue
                elif ids[nei] == -1:
                    ids[cur_node] = min(ids[cur_node], dfs(nei, cur_id + 1, cur_node))
                else:
                    ids[cur_node] = min(ids[cur_node], ids[nei])
            if ids[cur_node] == cur_id and cur_node != 0:
                res.append((par, cur_node))
            return ids[cur_node]
        dfs(0, 0, -1)
        return res
```