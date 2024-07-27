### 2848. Points That Intersect With Cars

```c++
class Solution {
public:
    int numberOfPoints(vector<vector<int>>& nums) {
        int f[102]{};
        for (auto &p: nums) {
            f[p[0]]++;
            f[p[1] + 1]--;
        }
        int res = 0, pre = 0;
        for (int n: f) {
            pre += n;
            res += pre > 0;
        }
        return res;
    }
};
```

### 1893. Check if All the Integers in a Range Are Covered

```c++
class Solution {
public:
    bool isCovered(vector<vector<int>>& ranges, int left, int right) {
        int f[52]{};
        for (auto &p: ranges) {
            f[p[0]]++;
            f[p[1] + 1]--;
        }
        int pre = 0;
        for (int i = 0; i < 52; i++) {
            pre += f[i];
            if (i >= left && i <= right && pre == 0) {
                return false;
            }
        }
        return true;
    }
};
```

### 1854. Maximum Population Year

```c++
class Solution {
public:
    int maximumPopulation(vector<vector<int>>& logs) {
        int f[2052]{};
        for (auto &p: logs) {
            f[p[0]]++;
            f[p[1]]--;
        }
        int mx = 0, pre = 0, year = 0;
        for (int i = 0; i < 2052; i++) {
            pre += f[i];
            if (pre > mx) {
                mx = pre;
                year = i;
            }
        }
        return year;
    }
};
```

### 1094. Car Pooling

```c++
class Solution {
public:
    bool carPooling(vector<vector<int>>& trips, int capacity) {
        int f[1001]{};
        for (auto &p: trips) {
            f[p[1]] += p[0];
            f[p[2]] -= p[0];
        }
        int pre = 0;
        for (int n: f) {
            pre += n;
            if (pre > capacity) {
                return false;
            }
        }
        return true;
    }
};
```