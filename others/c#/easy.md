### 1. Two Sum

```c#
public class Solution {
    public int[] TwoSum(int[] nums, int target) {
        for (int i = 0; i < nums.Length; i++) {
            for (int j = i + 1; j < nums.Length; j++) {
                if (nums[i] + nums[j] == target) {
                    return [i, j];
                }
            }
        }
        return [0, 0];
    }
}
```

```c#
public class Solution {
    public int[] TwoSum(int[] nums, int target) {
        Dictionary<int, int> d = new Dictionary<int, int>();
        for (int i = 0; i < nums.Length; i++) {
            int res = target - nums[i];
            if (d.ContainsKey(res)) {
                return new int[] {d[res], i};
            }
            if (!d.ContainsKey(nums[i])) {
                d.Add(nums[i], i);
            }
        }
        return [0, 0];
    }
}
```