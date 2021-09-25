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

  

## 1.3 区间问题

+ **435 无重叠区间**

  给定多个区间，计算让这些区间互不重叠所需要移除区间的最少个数。起止相连不算重叠。

  ```
  Input: [[1, 2], [2, 4], [1, 3]]
  Output: 1
  ```

  **思路：** 确定移除哪些区间等价于确定保留哪些区间， 保留的区间结尾越小，留给其他区间的空间就越大，这样就能够尽可能的保留更多的区间。

  ```c++
  // 贪婪算法求解 超出了时间限制
  class Solution {
  public:
      int eraseOverlapIntervals(vector<vector<int>>& intervals) {
          if (intervals.empty()) {
              return 0;
          }
  
          sort(intervals.begin(), intervals.end(), [](vector<int> a, vector<int> b) {
              return a[1] < b[1];
          });
  
          int n = intervals.size(), ans = 0;
          int prev = 0;
          for (int i = 1; i < n; ++i) {
              if (intervals[i][0] < intervals[prev][1]) ++ans;
              else prev = i;
          }
          
          return ans;
      }
  };
  // 动态规划求解
  
  class Solution {
  public:
    int eraseOverlapIntervals(vector<vector<int>>& intervals{
      if (intervals.empty()) {
        return 0;
      }
      sort(intervals.begin(), intervals.end(), [](const auto& u, const auto& v) {
        return u[0] < v[0];
      });
      
      int n = intervals.size();
      vector<int> f(n, 1);
      for (int i = 1; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
          if (intervals[j][1] <= intervals[i][0]) {
            f[i] = max(f[i], f[j] + 1);
          }
        }
      }
      return n - *max_element(f.begin(), f.end());
    }
  };
  ```

+ **605 种花问题**

  采取什么样的贪心策略能够种植最多的花

  对于两边1中间0能够种多少花，可以采用数学归纳法的形式求解出来，这里省略。对于中间有count 个0 ， 可以种植(count - 1) / 2 个花。**为了代码的统一性，两边补一个0**。

  ```c++
  class Solution {
  public:
    bool canPlaceFlowers(vector<int>& flowerbed, int n) {
      int countZero = 1;
      int canPlace = 0;
      for (auto flower: flowerbed) {
        if (flower == 0){
          ++countZero;
        } else {
          canPlace += (countZero - 1) / 2;
          if (canPlace >= n) return true;
          countZero = 0;
        }
      }
      ++countZero;
      canPlace += (countZero - 1) / 2;
      return canPlace >= n;
    }
  }
  ```

+ **用最少数量的箭引爆气球**

  ```
  输入：points = [[10,16],[2,8],[1,6],[7,12]]
  输出：2
  解释：对于该样例，x = 6 可以射爆 [2,8],[1,6] 两个气球，以及 x = 11 射爆另外两个气球
  ```

  ```c++
  class Solution {
  public:
    int findMinArrowShots(vector<vector<int>>& points) {
      if (points.empty()) {
        return 0;
      }
      sort(points.begin(), points.end(), [](vector<int> &u, vector<int> &v) {
        return u[1] < v[1];
      });
      int pos = points[0][1], ans=0;
      for (int i = 0; i < points.size(); ++i) {
              if (points[i][0] > pos) {
                  ++ans;
                  pos = points[i][1];
              }
          }
      return ans;
    }
  }
  
  执行用时：368 ms , 在所有 C++ 提交中击败了 49.23% 的用户
  内存消耗：87.7 MB , 在所有 C++ 提交中击败了 30.62% 的用户
  
  // 改用范围for语句
  class Solution {
  public:
    int findMinArrowShots(vector<vector<int>> &points) {
      if (points.empty()) {
        return 0;
      }
      sort(points.begin(), points.end(), [](vector<int> &u, vector<int> &v){
        return u[1] < v[1];
      });
      int pos = points[0][1], ans = 1;
      for (const vector<int> &ballon: points) {
        if (ballon[0] > pos) {
          ++ans;
          pos = ballon[1];
        }
      }
      return ans;
    }
  }
  ```

+ **763 划分字母区间**

  字符串 `S` 由小写字母组成。我们要把这个字符串划分为尽可能多的片段，同一字母最多出现在一个片段中。返回一个表示每个字符串片段的长度的列表。

  ```
  输入：S = "ababcbacadefegdehijhklij"
  输出：[9,7,8]
  解释：
  划分结果为 "ababcbaca", "defegde", "hijhklij"。
  每个字母最多出现在一个片段中。
  像 "ababcbacadefegde", "hijhklij" 的划分是错误的，因为划分的片段数较少。
  ```

  ```c++
  // 记录每个字母最后一次出现的位置，然后逐个遍历S
  class Solution {
  public:
    vector<int> partitionLabels(string s) {
      int last[26];
      int length = s.size();
      for (int i = 0; i < length; ++i) {
        last[s[i] - 'a'] = i;
      }
      
      int start = 0, end = 0;
      vector<int> partition;
      for (int i = 0; i < length; ++i) {
        end = max(end, last[s[i] - 'a']);
        if (i == end) {
          partition.push_back(end - start);
          start = end + 1;
        }
      }
      
      return partition;
    }
  }
  ```

+ 华为面试题

  餐厅支持预订服务，每一位顾客都需要预订就餐时间[start, end) 注意此处为半开区间，顾客于哥哭之间的预订存在时间交叠，如果有K个顾客存在交叠，那么就认为存在K个同时预订，请求解第一个用餐高峰期的持续时间段[start,stop)，即K 第一次达到最大时的持续时间段。

  即首先K达到最大，达到最大后，如果有多个区间求解第一个。

  ```
  ```

  











