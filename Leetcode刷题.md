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

  餐厅支持预订服务，每一位顾客都需要预订就餐时间[start, end) 注意此处为半开区间，顾客于顾客之间的预订存在时间交叠，如果有K个顾客存在交叠，那么就认为存在K个同时预订，请求解第一个用餐高峰期的持续时间段[start,stop)，即K 第一次达到最大时的持续时间段。

  即首先K达到最大，达到最大后，如果有多个区间求解第一个。

  ```
  Input: nums = 6, times = [[10,15],[55,60],[10,40],[5,15],[5,10],[25, 55]]
  Output: [10, 15]
  ```

  

  ```c++
  class Solution {
  public:
    vector<int> CustomerReserve(int &nums, vector<vector<int>> &times) {
      sort(times.begin(), times.end(), [](vector<int> &u, vector<int> &v) {
        return u[0] < v[0] || u[0] == v[0] && u[1] < v[1];
      });
      int k = 1, k_start = times[0][0], k_stop = times[0][1]
      int cnt_lap = 1; l_lap = k_start, r_lap = k_stop;
      for (int i = 0; i < nums - 1; ++i) {
        if (times[i + 1][0] < r_lap) {
          ++cnt_lap;
          l_lap = times[i + 1][0];
          r_lap = min(times[i + 1][1], r_lap);
          if (cnt_lap > k) {
            k = cnt_lap;
            k_start = l_lap;
            k_stop = r_lap;
          }
        }
        cnt_lap = 1;
        l_lap = 
      }
    }
      
  }
  
  class Solution {
  public:
    vector<int> CustomerReserve(int &nums, vector<int> &times) {
      int nums;
      cin >> nums;
      vector<vector<int>> ReserveTime(nums);
      for (int i = 0; i < nums; ++i) {
        cin >> ReserveTime[i][0] >> ReserveTime[i][1];
      }
      int max_k = 1;
      vector<int> ans;
   
    }
  }
  ```

+ **122 买卖股票的最佳时机 II**

  ```
  输入: prices = [7,1,5,3,6,4]
  输出: 7
  解释: 在第 2 天（股票价格 = 1）的时候买入，在第 3 天（股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5-1 = 4 。
       随后，在第 4 天（股票价格 = 3）的时候买入，在第 5 天（股票价格 = 6）的时候卖出, 这笔交易所能获得利润 = 6-3 = 3 。
  ```

  只要把爬坡的都统计到就行了

  ```c++
  class Solution {
  public:
    int maxProfit(vector<int> &prices) {
      if (prices.size() <= 1) return 0;
      int ans = 0;
      for (int i = 0; i < prices.size() - 1; ++i) {
        if(prices[i + 1] >= prices[i]) {
          ans += prices[i + 1] - prices[i];
        }
      }
      return ans
    }
  }
  ```

+ **406 根据身高重建队列**

  假设有打乱顺序的一群人站成一个队列，数组 people 表示队列中一些人的属性（不一定按顺序）。每个 people[i] = [hi, ki] 表示第 i 个人的身高为 hi ，前面 正好 有 ki 个身高大于或等于 hi 的人。

  请你重新构造并返回输入数组 people 所表示的队列。返回的队列应该格式化为数组 queue ，其中 queue[j] = [hj, kj] 是队列中第 j 个人的属性（queue[0] 是排在队列前面的人）。

  ```
  输入：people = [[7,0],[4,4],[7,1],[5,0],[6,1],[5,2]]
  输出：[[5,0],[7,0],[5,2],[6,1],[4,4],[7,1]]
  解释：
  编号为 0 的人身高为 5 ，没有身高更高或者相同的人排在他前面。
  编号为 1 的人身高为 7 ，没有身高更高或者相同的人排在他前面。
  编号为 2 的人身高为 5 ，有 2 个身高更高或者相同的人排在他前面，即编号为 0 和 1 的人。
  编号为 3 的人身高为 6 ，有 1 个身高更高或者相同的人排在他前面，即编号为 1 的人。
  编号为 4 的人身高为 4 ，有 4 个身高更高或者相同的人排在他前面，即编号为 0、1、2、3 的人。
  编号为 5 的人身高为 7 ，有 1 个身高更高或者相同的人排在他前面，即编号为 1 的人。
  因此 [[5,0],[7,0],[5,2],[6,1],[4,4],[7,1]] 是重新构造后的队列。
  ```

  ```c++
  class Solution {
  public:
    vector<vector<int>> reconstructQueue(vector<vector<int>>& people) {
      // 先排序
      sort(people.begin(), people.end(), [](vector<int> &u, vector<int> &v) {
        return u[0] > v[0] || (u[0] == v[0] && u[1] < v[1]);
      });
      // 插入
      vector<vector<int>> ans;
      for (const vector<int> person: people){
        ans.insert(ans.begin() + person[1], person);
      }
      return ans;
    }
  }
  
  执行用时：152 ms, 在所有 C++ 提交中击败了40.60%的用户
  内存消耗：12.2 MB, 在所有 C++ 提交中击败了49.54%的用户
  ```

+ **605 非递减数列 难度中等**

  给你一个长度为 n 的整数数组，请你判断在 最多 改变 1 个元素的情况下，该数组能否变成一个非递减数列。

  我们是这样定义一个非递减数列的： 对于数组中任意的 i (0 <= i <= n-2)，总满足 nums[i] <= nums[i + 1]。

  ```
  输入: nums = [4,2,3]
  输出: true
  解释: 你可以通过把第一个4变成1来使得它成为一个非递减数列。
  ```

  分析题目，要使数组变成一个非递减数列，nums[i] > nums[i + 1] 最多只出现一个。但是满足这个条件还不够，例如[3, 4, 1, 2]，还必须保证修改后的数组是非递减数列，这里贪心算法的思想在于，我们可以将nums[i] 修改为小于等于nums[i + 1] ， 但是为了不使得i之前的数变成不是非递减的，nums[i] 应该尽可能的大。所以nums[i] 最优应该修改为和nums[i + 1] 一样，同理还要尝试修改一下nums[i + 1]，修改完之后要看看是否变成了非递减数列。

  ```c++
  class Solution {
  public:
    bool checkPossibility(vector<int> &nums) {
      int n = nums.size();
      if (n <= 2) return true;
      for (int i = 0; i < n - 1; ++i) {
        int x = nums[i], y = nums[i + 1];
        if (x > y) {
          nums[i] = y;
          if (is_sorted(nums.begin(), nums.end())) return true;
          nums[i] = x;
          nums[i + 1] = x;
          return is_sorted(nums.begin(), nums.end());
        }
      }
      return true;
    }
  }
  
  //只遍历一次数组
  class Solution {
  public:
    bool checkPossibility(vector<int> &nums) {
      int n = nums.size(), cnt = 0;
      for (int i = 0; i < n - 1; ++i) {
        int x = nums[i], y = nums[i + 1];
        if (x > y) {
          ++cnt;
          if (cnt > 1) return false;
          if (i > 0 && y < nums[i - 1]) {
            nums[i + 1] = x;
          }
        }
      }
      return true
    }
  }
  ```

# 2. 双指针

指针与常量

```
int x;
int *p1 = &x; // 指针可以被修改，值也可以被修改
const int *p2 = &x; // 指针可以被修改，但是值不可以被修改
int * const p3 = &x; // 指针不可以被修改，但是值可以被修改
const int * const p4 = &x; // 指针不可以被修改，值也不可以被修改
```

指针函数与函数指针

```c++
// addition是指针函数，一个返回类型是指针的函数
int* addition(int a, int b) {
  int* sum = new int(a + b);
  return sum;
}
```

## 2.1 Two Sum

+ **167 两数之和 II - 输入有序数组**

给定一个已按照 非递减顺序排列  的整数数组 numbers ，请你从数组中找出两个数满足相加之和等于目标数 target 。

函数应该以长度为 2 的整数数组的形式返回这两个数的下标值。numbers 的下标 从 1 开始计数 ，所以答案数组应当满足 1 <= answer[0] < answer[1] <= numbers.length 。

你可以假设每个输入 只对应唯一的答案 ，而且你 不可以 重复使用相同的元素。

```c
输入：numbers = [2,7,11,15], target = 9
输出：[1,2]
解释：2 与 7 之和等于目标数 9 。因此 index1 = 1, index2 = 2 。
```

```c++
class Solution {
public:
    vector<int> twoSum(vector<int>& numbers, int target){
    	int left = 0, right = numbers.size() - 1, sum;
      while(left < right) {
        sum = numbers[left] + numbers[right];
        if (sum == target) break;
        if (sum < target) ++left;
        else --right;
      }
      return vector<int>{left + 1, right + 1};
    } 
}
```

## 2.2 归并两个有序数组

+ **88 合并两个有序数组**

  输入是两个数组和它们分别的长度 m 和 n。其中第一个数组的长度被延长至 m + n，多出的n 位被 0 填补。题目要求把第二个数组归并到第一个数组上，不需要开辟额外空间。

  ```
  输入：nums1 = [0], m = 0, nums2 = [1], n = 1
  输出：[1]
  解释：需要合并的数组是 [] 和 [1] 。
  合并结果是 [1] 。
  注意，因为 m = 0 ，所以 nums1 中没有元素。nums1 中仅存的 0 仅仅是为了确保合并结果可以顺利存放到 nums1 中。
  ```

  ```c++
  class Solution {
  public:
      void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
        int pos = m-- + n-- - 1;
        while(m >= 0 && n >= 0) {
  				nums1[pos--] = nums1[m] >= nums2[n] ? nums1[m--]: nums2[n--];
        }
        while (n >= 0) {
          nums1[pos--] = nums2[n--];
        }
      }
  }
  ```

## 3.3 快慢指针

+ **142 环形链表2**

  给定一个链表，返回链表开始入环的第一个节点。 如果链表无环，则返回 null。

  为了表示给定链表中的环，我们使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。 如果 pos 是 -1，则在该链表中没有环。注意，pos 仅仅是用于标识环的情况，并不会作为参数传递到函数中。

  说明：不允许修改给定的链表。

  进阶：你是否可以使用 O(1) 空间解决此题？

  ```
  输入：head = [3,2,0,-4], pos = 1
  输出：返回索引为 1 的链表节点
  解释：链表中有一个环，其尾部连接到第二个节点。
  ```

  ```c++
  /**
   * Definition for singly-linked list.
   * struct ListNode {
   *     int val;
   *     ListNode *next;
   *     ListNode(int x) : val(x), next(NULL) {}
   * };
   */
  class Solution {
  public:
      ListNode *detectCycle(ListNode *head) {
        ListNode *fast = head, *slow = head;
        do {
          if (!fast || !fast->next) return nullptr;
          fast = fast->next->next;
          slow = slow->next;
        } while(fast != slow);
        
        fast = head;
        while (fast != slow){
          fast = fast->next;
          slow = slow->next;
        }
        return fast;
      }
  };
  ```

## 3.4 滑动窗口

+ **76 最小覆盖字串**

  给定两个字符串 S 和 T，求 S 中包含 T 所有字符的最短连续子字符串的长度，同时要求时间复杂度不得超过 O(n)。
  
  ```
  输入：s = "ADOBECODEBANC", t = "ABC"
  输出："BANC"
  ```
  
  ```c++
  class Solution {
  public:
    string minWindow(string s, string t) {
      vector<int> chars(128, 0);
      vector<int> flag(128, false);
      for (int i = 0; i < t.size(); ++i) {
        flag[t[i]] = true;
        ++chars[t[i]];
      }
      int cnt = 0, l = 0, min_l = 0, min_size = s.size() + 1;
      for (int r = 0; r < s.size(); ++r) {
        if (flag[s[r]]) {
          if (--chars[s[r]] >= 0) {
            ++cnt;
          }
          //左移指针
          while (cnt == t.size()) {
            if (r - l + 1 < min_size) {
              min_l = l;
              min_size = r - l + 1;
            }
            if (flag[s[l]] && ++chars[s[l]] > 0) {
              --cnt;
            }
            ++l;
          }
        }
      }
      return min_size <= s.size()? s.substr(min_l, min_size): "";
    }
  }
  ```
  
  



















