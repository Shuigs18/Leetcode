# Leetcode
## 1. 贪心算法

### 1.1 算法介绍

#### 1.4 练习

+ **605 Can Place Flowers(Easy)**

+ **763 Partition Labels (Medium)**

  字符串 `S` 由小写字母组成。我们要把这个字符串划分为尽可能多的片段，同一字母最多出现在一个片段中。返回一个表示每个字符串片段的长度的列表。

  ```c++
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
                  partition.push_back(end - start + 1);
                  start = end + 1;
              }
          }
          return partition;
      }
  };
  ```

  ```
  执行用时：4 ms, 在所有 C++ 提交中击败了67.19%的用户
  内存消耗：6.6 MB, 在所有 C++ 提交中击败了27.44%的用户
  ```

  



