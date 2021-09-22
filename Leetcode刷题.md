---
title: Leetcode 刷题总结
---

刷题顺序参考 《LeetCode 101：和你一起你轻松刷题（C++）》 -- 高畅 Chang Gao

# 1. 贪心算法

贪心策略，每次都是局部最优，从而使的全局最优，注意这并不是一个充分条件，因为局部最优并不一定导出全局最优。

## 1.1 分配问题

+ **455 分配饼干**

  有一群孩子和一堆饼干，每个孩子有一个饥饿度，每个饼干都有一个大小。每个孩子只能吃最多一个饼干，且只有饼干的大小大于孩子的饥饿度时，这个孩子才能吃饱。求解最多有多少孩子可以吃饱

  ```
  输入两个数组分别代表孩子的饥饿度和饼干的大小
  Input: [1, 2], [1, 2, 3]
  Output: 2
  ```

  思路：先满足饥饿度最小的孩子，以剩下尽可能多的饼干满足更多的孩子，举个例子，对于饥饿度为 2和4 的孩子，如果此时有饼干3，4 如果给饥饿度为2 的孩子4就会导致 3 这个原本可以满足2的没被利用，所以为了防止这种情况的发生，先满足饥饿度最小的孩子。这个可以扩展为资源匹配问题。从最小的开始匹配，如果最小的需求都不能被满足，那么这个供给更不可能满足更大的需求。最优情况下也是被抛弃。

  ```c++
  class Solution {
  public:
    int findContentChildren(vector<int> &g, vector<int> &s) {
      sort(g.begin(), g.end());
      sort(s.begin(), s.end());
      int child = 0, cookie = 0;
      while (child < g.size() && cookie < s.size()) {
        if (g[child] <= s[cookie]) ++child;
        ++cookie;
      }
      return child;
    }
  }
  ```

+ **135 分发糖果**

  一群孩子站成一排，每一个孩子有自己的评分。现在需要给这些孩子发糖果，规则是如果一个孩子的评分比自己身旁的一个孩子要高，那么这个孩子就必须得到比身旁孩子更多的糖果；所有孩子至少要有一个糖果。求解最少需要多少个糖果。

  ```
  Input: [1, 0, 2]
  Output: 5
  最少的糖果分法是[2, 1, 2]
  ```

  思路：首先每人必定有至少一个糖果，分发完毕后，对于每个孩子从左往右看如果右边的孩子比自己高，右边孩子的糖果数加一；然后，从右往左看，如果左边孩子比自己高，那么左边孩子的糖果数加一。

  ```c++
  class Solution {
  public:
  	int candy(vector<int>& ratings) {
      int size = ratings.size();
      if (size < 2) {
        return 2;
      }
     	vector<int> num(size, 1);
      // 从左向右遍历
      for (int i = 1; i < size; ++i) {
        if (ratings[i] > ragings[i - 1]) {
          num[i] = num[i - 1] + 1;
        }
      }
      // 从右向左遍历
      for (int i = size - 1; i > 0; --i) {
        if (ratings[i - 1] > ratings[i]) {
          num[i - 1] = max(num[i - 1], num[i] + 1); 
        }
      }
      return accumulate(num.begin(), num.end(), 0);
    }
  }
  ```

  





















