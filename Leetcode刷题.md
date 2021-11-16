

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

+ **平方数之和**

  给定一个非负整数 `c` ，你要判断是否存在两个整数 `a` 和 `b`，使得 `a2 + b2 = c`

  ```
  输入：c = 5
  输出：true
  解释：1 * 1 + 2 * 2 = 5
  ```

  ```c++
  class Solution {
  public:
      bool judgeSquareSum(int c) {
        int left = 0;
        int right = (int) (sqrt(c));
        while (left <= right) {
          sum = left * left + right * right;
          if (sum == c) return true;
          if (sum < c) ++left;
          else --right;
        }
        return false;
      }
  }
  ```

+ **验证回文字符串II**

  给定一个非空字符串 `s`，**最多**删除一个字符。判断是否能成为回文字符串。

  ```
  输入: s = "aba"
  输出: true
  
  输入: s = "abca"
  输出: true
  解释: 你可以删除c字符。
  
  输入: s = "abc"
  输出: false
  ```

  ```c++
  class Solution {
  public:
  		bool checkPalindrome(string s, int low, int high) {
        	while (low < high) {
            	if (s[low] == s[high]) {
                	++low;
                	--high;
              } else return false;
          }
        	return true;
      }
    
      bool validPalindrome(string s) {
  				int low = 0; high = s.size() - 1;
        	while (low < high) {
            	if (s[low] == s[high]) {
                --high;
                ++low;
              } else {
                	return checkPalindrome(s, low, high - 1) || checkPalindrome(s, low + 1, high);
              }
          }
        	return true;
      }
  };
  ```

+ **524 通过删除字母匹配到字典里最长单词**

  给你一个字符串 s 和一个字符串数组 dictionary ，找出并返回 dictionary 中最长的字符串，该字符串可以通过删除 s 中的某些字符得到。

  如果答案不止一个，返回长度最长且字母序最小的字符串。如果答案不存在，则返回空字符串。

  ```
  输入：s = "abpcplea", dictionary = ["ale","apple","monkey","plea"]
  输出："apple"
  ```

  ```c++
  class Solution {
  public:
      string findLongestWord(string s, vector<string>& dictionary) {
          sort(dictionary.begin(), dictionary.end(), [](const string u, const string v) {
              return u.length() > v.length() || (u.length() == v.length() && u < v);
          });
          int n = s.length();
          for (auto & ss: dictionary) {
              int m = ss.length();
              if (m > n) continue;
              for (int i = 0, j = 0; i < n && j < m; ++i) {
                  if (s[i] == ss[j]) ++j;
                  if (j == m) return ss;
              }
          }
          return "";
      }
  };
  ```

  ```c++
  // dp 序列自动机
  /*
  [&]：隐式捕获列表，采用引用捕获方式。lambda体中所使用的来自所在函数的实体都采用引用方式使用。
  [=]：隐式捕获列表，采用值捕获方式。lambda体将拷贝所使用的来自所在函数的实体的值。
  [&, identifier_list]：identifier_list是一个逗号分隔的列表，包含0个或者多个来自所在函数的变量。这些变量采用值捕获方式，而任何隐式捕获的变量都采用引用方式捕获。identifier_list中的名字前面个不能使用&。
  [=, identifier_list]：identifier_list中的变量都采用引用方式捕获，而任何隐式捕获的变量都采用值方式捕获。identifier_list中名字不能包括this，且这些名字之前必须使用&。  
  */
  class Solution {
  public:
      string findLongestWord(string s, vector<string>& dic) {
          sort(dic.rbegin(), dic.rend(), [](auto&& a, auto&& b){return a.size() < b.size() || a.size() == b.size() && a > b;});
          dic.push_back("");
          vector<vector<int>> next(s.size() + 1, vector(26, -1));
          for(int i = s.size() - 1; i >= 0; --i)
              for(int j = 0; j < 26; ++j)
                  next[i][j] = s[i] == (j+'a') ? i : next[i+1][j];
  
        auto check = [&](string& x){
              int m = x.size(), l = 0, r = 0;
              while(r < m) if(!(l = next[l][x[r++]-'a'] + 1)) return false;
              return r == m;
          };
          return *find_if(begin(dic), end(dic), check);
      }
  };
  ```

+ 340 

# 4. 二分查找

## 4.1 算法技巧

二分查找时区间的左右端取开区间还是闭区间在绝大多数时候都可以，因此有些初学者会容易搞不清楚如何定义区间开闭性。这里我提供两个小诀窍，第一是尝试熟练使用一种写法，比如左闭右开（满足 C++、 Python 等语言的习惯）或左闭右闭（便于处理边界条件），尽量只保持这一种写法；第二是在刷题时思考如果最后区间只剩下一个数或者两个数，自己的写法是否会陷入死循环，如果某种写法无法跳出死循环，则考虑尝试另一种写法。

二分查找也可以看作双指针的一种特殊情况，但我们一般会将二者区分。双指针类型的题，指针通常是一步一步移动的，而在二分查找里，指针每次移动半个区间长度。

​    

## 4.2 求开方

+ **69 Sqrt(x)**

  给你一个非负整数 x ，计算并返回 x 的 算术平方根 。

  由于返回类型是整数，结果只保留 整数部分 ，小数部分将被 舍去 。

  注意：不允许使用任何内置指数函数和算符，例如 pow(x, 0.5) 或者 x ** 0.5

  ```
  输入：x = 8
  输出：2
  解释：8 的算术平方根是 2.82842..., 由于返回类型是整数，小数部分将被舍去。
  ```

  ```c++
  // 左闭右毕
  class Solution {
  public:
      int mySqrt(int x) {
          if (x == 0) return 0;
          int l = 1, r = x, mid, sqrt;
          while (l <= r) {
              mid = l + (r - l) / 2;
              sqrt = x / mid;
              if (sqrt == mid) return sqrt;
              else if (sqrt < mid) {
                  r = mid - 1;
              } else {
                  l = mid + 1;
              }
          }
          return r;
      }
  };
  
  // 牛顿迭代法
  class Solution {
  public:
      int mySqrt(int x) {
          int m = x;
          while (m * m > x) {
              m = (m + x / m) / 2;
          }
          return m;
      }
  }
  ```

## 4.3 查找区间

+ **34 在排序数组中查找元素的第一个和最后一个位置**

  给定一个按照升序排列的整数数组 nums，和一个目标值 target。找出给定目标值在数组中的开始位置和结束位置。

  如果数组中不存在目标值 target，返回 [-1, -1]。

  ```
  输入：nums = [5,7,7,8,8,10], target = 8
  输出：[3,4]
  ```

  ```c++
  class Solution {
  public:
      vector<int> searchRange(vector<int>& nums, int target) {
  		if (nums.empty()) return vector<int>{-1, -1};
          int lower = lower_bound(nums, target);
          int upper = upper_bound(nums, target) - 1;
          if (lower == nums.size() || nums[lower] != target) {
              return vector<int>{-1, -1};
          }
          return vector<int>{lower, upper};
      }
      // 辅助函数 lower_bound
      int lower_bound(vector<int>& nums, int target) {
          int l = 0, r = nums.size(), mid,
          while (l < r) {
              mid = (l + r) / 2;
              if (nums[mid] >= target) {
                  r = mid;
              } else {
                  l = mid + 1;
              }
          }
          return l;
      }
      
      // 辅助函数 upper_bound
      int upper_bound(vector<int>& nums, int target) {
          int l = 0, r = nums.size(), mid,
          while (l < r) {
              mid = (l + r) / 2;
              if (nums[mid] > target) {
                  r = mid;
              } else {
                  l = mid + 1;
              }
          }
          return l;
      }
  };
  ```

## 4.4 旋转数组查找数字

+ **81搜索旋转排序数组 II**

  已知存在一个按非降序排列的整数数组 nums ，数组中的值不必互不相同。

  在传递给函数之前，nums 在预先未知的某个下标 k（0 <= k < nums.length）上进行了 旋转 ，使数组变为 [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]（下标 从 0 开始 计数）。例如， [0,1,2,4,4,4,5,6,6,7] 在下标 5 处经旋转后可能变为 [4,5,6,6,7,0,1,2,4,4] 。

  给你 旋转后 的数组 nums 和一个整数 target ，请你编写一个函数来判断给定的目标值是否存在于数组中。如果 nums 中存在这个目标值 target ，则返回 true ，否则返回 false。
  
  ```c++
  [1,0,1,1,1]
  0
  
  class Solution {
  public:
      bool search(vector<int>& nums, int target) {
      	int l = 0, r = nums.size() - 1, mid;
          while (l <= r) {
              /*
              l < r 会使 [1,0,1,1,1] 0 这个实例不通过
              */
              mid = (l + r) / 2;
              if (nums[mid] == target) return true;
              // 无法判断左边是增序还是右边是增序
              if (nums[mid] == nums[l]) {
                  ++l;
              // 右区间是增序
              } else if (nums[mid] <= nums[r]) {
                  if (target > nums[mid] && target <= nums[r]) {
                      l = mid + 1;
                  } else{
                      r = mid - 1;
                  }
              } else {
                  if (target < nums[mid] && target >= nums[l]) {
                      r = mid - 1;
                  } else {
                      l = mid + 1
                  }
              }
          }
          return false;
      }
  };
  ```

## 4.5 练习

+ **154 [寻找旋转排序数组中的最小值 II](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array-ii/)**

  已知一个长度为 n 的数组，预先按照升序排列，经由 1 到 n 次 旋转 后，得到输入数组。例如，原数组 nums = [0,1,4,4,5,6,7] 在变化后可能得到：
  若旋转 4 次，则可以得到 [4,5,6,7,0,1,4]
  若旋转 7 次，则可以得到 [0,1,4,4,5,6,7]
  注意，数组 [a[0], a[1], a[2], ..., a[n-1]] 旋转一次 的结果为数组 [a[n-1], a[0], a[1], a[2], ..., a[n-2]] 。

  给你一个可能存在 重复 元素值的数组 nums ，它原来是一个升序排列的数组，并按上述情形进行了多次旋转。请你找出并返回数组中的 最小元素 。

  ```
  输入：nums = [1,3,5]
  输出：1
  ```

  ```c++
  class Solution {
  public:
      int findMin(vector<int>& nums) {
          
          int l = 0, r = nums.size() - 1, mid;
          
          while (l < r) {
              mid = (l + r) / 2;
              if (nums[l] < nums[r]) {
                  return nums[l];
              } else if (nums[l] == nums[mid]) {
                  ++l;
              } else if (nums[mid] > nums[l]) {
                  l = mid + 1;
              } else {
                  r = mid;
              }
          }
          return nums[l];
      }
  };
  ```

+ **540 [有序数组中的单一元素](https://leetcode-cn.com/problems/single-element-in-a-sorted-array/)**

  给定一个只包含整数的有序数组，每个元素都会出现两次，唯有一个数只会出现一次，找出这个数。

  ```
  输入: nums = [1,1,2,3,3,4,4,8,8]
  输出: 2
  ```

  ```c++
  class Solution {
  public:
      int singleNonDuplicate(vector<int>& nums) {
  		int l = 0, r = nums.size(), mid;
          
          while (l < r) {
              mid = (l + r) / 2;
              if (nums[mid + 1] != nums[mid] &&
                  nums[mid] ! = nums[mid - 1]) return nums[mid];
              if (nums[mid] == nums[mid + 1]) {
                  if ((mid - l) % 2 == 1) {
                      r = mid - 1
                  } else {
                      l = mid + 2;
                  }
              }
              if (nums[mid - 1] = nums[mid]) {
                  if ((r - mid) % 2 == 1) {
                      l = mid + 1;
                  } else {
                      r = mid - 2;
                  }
              }
          }
          return nums[l];
      }
  };
  //执行用时：4 ms, 在所有 C++ 提交中击败了 95.78% 的用户 
  //内存消耗：10.8 MB, 在所有 C++ 提交中击败了17.08%的用户
  
  class Solution {
  public:
      int singleNonDuplicate(vector<int>& nums) {
          int l = 0, r = nums.size() - 1, mid;
          
          while (l < r) {
              mid = (l + r) / 2;
              if (nums[mid] == nums[mid + 1]) {
                  if ((mid - l) % 2 == 1) {
                      r = mid - 1;
                  } else {
                      l = mid + 2;
                  }
              } else if (nums[mid - 1] == nums[mid]) {
                  if ((r - mid) % 2 == 1) {
                      l = mid + 1;
                  } else {
                      r = mid - 2;
                  }
              } else {
                  return nums[mid];
              }
          }
          return nums[l];
      }
  };
  //内存消耗：10.8 MB, 在所有 C++ 提交中击败了35.82%的用户
  
  class Solution {
  public:
      int singleNonDuplicate(vector<int>& nums) {
          int lo = 0, hi = nums.size() - 1, mid;
  
          while (lo < hi) {
              mid = (hi + lo) / 2;
              if (mid % 2 == 1) mid--;
              if (nums[mid] == nums[mid + 1]) {
                  lo = mid + 2;
              } else {
                  hi = mid;
              }
          }
          return nums[lo];
      }
  };
  
  //执行用时：8 ms, 在所有 C++ 提交中击败了68.74%的用户
  //内存消耗：10.7 MB, 在所有 C++ 提交中击败了90.80%的用户
  ```

+ **4.[寻找两个正序数组的中位数](https://leetcode-cn.com/problems/median-of-two-sorted-arrays/)**

  给定两个大小分别为 `m` 和 `n` 的正序（从小到大）数组 `nums1` 和 `nums2`。请你找出并返回这两个正序数组的 **中位数** 

  需要对两个数组同时进行二分搜索

  ```
  输入：nums1 = [1,3], nums2 = [2]
  输出：2.00000
  解释：合并数组 = [1,2,3] ，中位数 2
  
  输入：nums1 = [1,2], nums2 = [3,4]
  输出：2.50000
  解释：合并数组 = [1,2,3,4] ，中位数 (2 + 3) / 2 = 2.5
  ```

  下面这个解法的巧妙点在于将原本求中位数转换为了求两个数组中的第K个最小值

  这里可以总结数组nums的中位数下标求法

  ```
  当有奇个数时，中位数为正中间的数
  left = int((nums.size() + 1) / 2) 
  right = = int((nums.size() + 2) / 2)
  此时 left = right
  当数组有偶数个数时，中位数为中间两个的平均值
  left = int((nums.size() + 1) / 2) 
  right = = int((nums.size() + 2) / 2)
  此时 left + 1 = right
  这里的left和right是数组的第几个数 如果转换成下标应该再减去一
  即中位数为 (nums[left] + nums[right]) / 2
  ```

  ```c++
  class Solution {
  public:
      
      double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
  		int n = nums1.size();
          int m = nums2.size();
          int left = (n + m + 1) / 2;
          int right = (n + m + 2) / 2;
          return (getKth(nums1, 0, n - 1, nums2, 0, m - 1, left) + getKth(nums2, 0, m - 1, right));
      }
      
      int getKth(vector<int>& nums1, int start1, int end1, 
                 vector<int>& nums2, int start2, int end2, 
                 int k) {
          len1 = end1 - start1 + 1;
          len2 = end2 - start2 + 1;
          if (len1 > len2) return getKth(nums2, start2, end2, nums1, start1, end1, k);
          if (len1 == 0) return nums2[start2 + k - 1];
          if (k == 1) return min(nums1[start1], nums2[start2]);
          
          int i = start1 + min(len1, k / 2) - 1;
          int j = start2 + min(len2, k / 2) - 1;
          if (nums1[i] < nums2[j]) {
              return getKth(nums1, i + 1, end1, nums2, start2, end2, k - (i - start + 1));
          } else {
              return getKth(nums1, start1, end1, nums2, j + 1, end2, k - (j - start + 1));
          }
      }
  };
  ```

# 5. 排序算法

常用的排序算法

+ **快速排序**

  我们采用左闭右闭的二分写法

  ```c++
  void quick_sort(vector<int> &nums, int l, int r) {
      if (l + 1 >= r) return; // 说明只有一个元素不需要排序了
      
      int first = l, last = r - 1, key = nums[first];
      while (first < last) {
          if (first < last && nums[last] >= key) --last;
          nums[first] = nums[last];
          if (first < last && nums[first] <= key) ++first;
          nums[last] = nums[first];
      }
      nums[first] = key;
      quick_sort(nums, l, first);
      quick_sort(nums, first + 1, r);
  }
  ```

  ```c++
  // 数据结构和语言算法
  template <class T>
  void quickShort(T a[], int n) {
      if (n <= 1) return;
      int max = indexOfMax(a, n);
      swap(a[n - 1], a[max]);
      quickShort(a, 0, n - 2);
  }
  
  void quickShort(T a[], int leftEnd, int rightEnd) {
      if (leftEnd >= rightEnd) return;
      
      int leftCursor = leftEnd,  rightCursor = rightEnd + 1;
      int pivot = a[leftEnd];
      while (true) {
          do {
              ++leftCursor;
          } while (a[leftCursor] < pivot);
          do {
              --rightCursor;
          } while (a[rightCursor] > pivot);
          if (leftCursor >= rightCursor) break;
          swap(a[leftCursor], a[rightCursor]);
      }
      a[leftEnd] = a[rightCursor];
      a[rightCursor] = pivot;
      quickShort(a, leftEnd, rightCursor - 1);
      quickShort(a, rightCursor + 1, rightEnd);
  }
  ```

+ **归并排序**

  ```c++
  void merge_sort(vector<int> &nums, int l, int r, vector<int> &temp) {
      if (l + 1 > r) return;
      
      int mid = (l + r) / 2;
      // 分开
      merge_sort(nums, l, mid, tmp);
      merge_sort(nums, mid, r, tmp);
      // 归并
      int p = l, q = m, i = l;
      // 代码不错
      while (p < m || q < r) {
          if (q >= r || (p < m && nums[p] < nums[q])) {
              tmp[i++] = nums[p++];
          } else {
              tmp[i++] = nums[q++];
          }
      }
      // tmp 复制回num
      for (i = l; i < r; i++) {
          nums[i] = tmp[i];
      }
  }
  ```

  ```c++
  // 数据结构与算法
  template <class T>
  void mergeSort(T *a, int n) {
      T *b = new T[n];
      int segmentSize = 1;
      while (segementSize < n) {
          mergePass(a, b, n, segmentSize);
          segmentSize += segmentSize;
          mergePass(b, a, n, segmentSize);
          segmentSize += segmentSize; 
      }
      delete[] b;
  }
  
  void mergePass(T *x, T *y, int n, int segmentSize) {
      int i = 0;
      while (i <= n - 2 * segmentSize) {
          merge(x, y, i, i + segmentSize - 1, i + 2 * segmentSize - 1);
          i += 2 * segmentSize;
      }
      // 如果不够两个满的数据段
      if (i + segmentSize < n) {
          // 剩有两个数据段
          merge(x, y, i, i + segmentSize - 1, n - 1);
      } else {
          // 剩下一个数据段 复制到y
          for (int j = i; j < n; ++j) {
              y[j] = x[j];
          }
      }
  }
  
  // 均为闭区间
  void merge(T c[], T d[], int startOfFirst, int endOfFirst, int endOfSecond) {
      int first = startOfFirst, second = endOfFirst, result = startOfFirst;
      while (first <= endOfFirst && second <= endofSecond) {
          if (c[first] <= c[second]) {
              d[result++] = c[first++];
          } else {
              d[result++] = c[second++];
          }
      }
      // 如果有段没复制完
      if (first > endOfFirst) {
          for (int q = second; q <= endofSecond; q++) {
              d[result++] = c[q];
          }
      } else {
          for (int p = first; p <= endofFirst; p++) {
              d[result++] = c[p];
          }
      }
      /* 
      while (first <= endOfFirst || second <= endOfSecond) {
      	if (second > endOfSecond || (first <= endOfFirst && c[first] <= c[second])) {
      		d[result++] = c[first++];
      	} else {
      		d[result++] = c[second++];
      	}
      }
      */
  }
  
  
  // 还是第一个两数组合并写的好
  
  ```

+ **插入排序**

  ```c++
  void insertion_sort(vector<int> &nums, int n) {
      for (int i = 0; i < n; ++i) {
          // 从小到大排列
          for (int j = i; j > 0 && nums[j] < nums[j - 1]; --j) {
              swap(nums[j], nums[j - 1]);
          }
      }
  }
  ```

+ **冒泡排序  (Bubble Sort)**

  ```c++
  template<class T>
  void bubble(T a[], int n) {
      for (int i = 0; i < n - 2; i++) {
          if (a[i] > a[i + 1]) swap(a[i], a[i + 2]);
      }
  }
  
  void bubbleSort(T a[], int n) {
      for (int i = n; i > 0; i++) {
          bubble(a, i);
      }
  }
  ```

+ **选择排序（Selection Sort）**

  ```c++
  template<class T>
  void selectionSort(T a[], int n) {
      for (int i = 0; i < n; i++) {
          mid = i;
          for (int j = i + 1; j < n; j++) {
              if (a[j] < a[mid]) mid = j;
          }
          swap(a[i], a[mid]);
      }
  }
  ```

## 5.2 快速选择

+ **215 [数组中的第K个最大元素](https://leetcode-cn.com/problems/kth-largest-element-in-an-array/)**

  给定整数数组 nums 和整数 k，请返回数组中第 k 个最大的元素。

  请注意，你需要找的是数组排序后的第 k 个最大的元素，而不是第 k 个不同的元素。


```
输入: [3,2,1,5,6,4] 和 k = 2
输出: 5
```

```c++
class Solution {
public:
    int findKthLargest(vector<int>& nums, int k) {
        int l = 0, r = nums.size() - 1, target = nums.size() - k;
        while (l < r) {
        	int mid = selectionSort(nums, l, r);
            if (mid == target) {
                return nums[mid];
            }
            if (mid < target) {
                r = mid - 1;
            } else {
                l = mid + 1;
            }
        }
        return nums[l];
    }
    
    int selectionSort(vector<int>& nums, int left, int right) {
        int first = l + 1, last = right;
        while (true) {
            while (first < r && nums[first] <= nums[l]) ++first;
            while (l < last && nums[last] >= nums[l]) --last;
            if (first >= last) break;
            swap(nums[first], nums[last]);
        }
        swap(nums[l], nums[last]);
        return last;
    }
};

//执行用时：64 ms, 在所有 C++ 提交中击败了17.63%的用户
//内存消耗：9.7 MB, 在所有 C++ 提交中击败了82.11%的用户
```

## 5.3 桶排序

+ **347 [前 K 个高频元素](https://leetcode-cn.com/problems/top-k-frequent-elements/)**

  给你一个整数数组 `nums` 和一个整数 `k` ，请你返回其中出现频率前 `k` 高的元素。你可以按 **任意顺序** 返回答案。

  ```
  输入: nums = [1,1,1,2,2,3], k = 2
  输出: [1,2]
  
  输入: nums = [1], k = 1
  输出: [1]
  
  提示：
   + 1 <= nums.length <= 105
   + k 的取值范围是 [1, 数组中不相同的元素的个数]
   + 题目数据保证答案唯一，换句话说，数组中前 k 个高频元素的集合是唯一的
  ```

  ```c++
  class Solution {
  public:
      vector<int> topKFrequent(vector<int>& nums, int k) {
          unordered_map<int, int> counts;
          int max_count = 0;
          // 确定桶的大小
          for (const int &num: nums) {
              max_count = max(max_count, ++counts[num]);
          }
          // 建立桶 为什么要加1
          vector<vector<int>> buckets(max_count + 1);
          for (const auto &p: counts) {
              buckets[p.second].push_back(p.first);
          }
          //
          vector<int> ans;
          for (int i = max_count; i >= 0 && ans.size() < k; i--) {
              for (const auto &num: buckets[i]) {
                  ans.push_back(num);
                  if (ans.size() == k) {
                      break;
                  }
              }
          }
          return ans;
      }
  };
  
  执行用时：12 ms, 在所有 C++ 提交中击败了83.51%的用户
  内存消耗：13.5 MB, 在所有 C++ 提交中击败了19.95%的用户
  ```

## 5.4 练习

+ **451 [根据字符出现频率排序](https://leetcode-cn.com/problems/sort-characters-by-frequency/)**

  给定一个字符串，请将字符串里的字符按照出现的频率降序排列。

  ```
  输入:
  "tree"
  
  输出:
  "eert"
  
  解释:
  'e'出现两次，'r'和't'都只出现一次。
  因此'e'必须出现在'r'和't'之前。此外，"eetr"也是一个有效的答案。
  
  输入:
  "cccaaa"
  
  输出:
  "cccaaa"
  
  解释:
  'c'和'a'都出现三次。此外，"aaaccc"也是有效的答案。
  注意"cacaca"是不正确的，因为相同的字母必须放在一起。
  ```

  ```c++
  class Solution {
  public:
      string frequencySort(string s) {
          // 用unordered_map试试
          map<char, int> counts;
          // 确定桶的大小
          int max_count = 0;
          for (const char &ch: s) {
              max_count = max(max_count, ++counts[ch]);
          }
          // 确定桶
         	vector<vector<char>> buckets;
          for (const auto &p: counts) {
              buckets[p.second].push_back(p.first);
          }
          //
          string res;
          for (int i = max_count; i >= 0 && res.size() < s.size(); --i) {
              for (const char &ch: buckets[i]) {
                  for (int j = 0; j < i; j++) {
                      res.push_back(ch);
                  }
                  if (res.size() == s.size()) break;
              }
          }
          return res;
      }
  };
  
  //执行用时：12 ms, 在所有 C++ 提交中击败了54.50%的用户
  //内存消耗：10.4 MB, 在所有 C++ 提交中击败了16.78%的用户
  ```

+ **75 [颜色分类](https://leetcode-cn.com/problems/sort-colors/) （中等）** 

  给定一个包含红色、白色和蓝色，一共 n 个元素的数组，**原地**对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列。

  此题中，我们使用整数 0、 1 和 2 分别表示红色、白色和蓝色。

  ```
  输入：nums = [2,0,2,1,1,0]
  输出：[0,0,1,1,2,2]
  
  输入：nums = [2,0,1]
  输出：[0,1,2]
  ```

  ```c++
  class Solution {
  public:
      void sortColors(vector<int>& nums) {
          sortValues(nums, 0);
      }
      
      void sortValues(vector<int>& nums, int l) {
          if (l == nums.size() - 1) return ;
          int minValue = *min_element(nums.cbegin() + l, nums.cend());
          int r = nums.size();
          int first = l, last = r - 1;
          while (true) {
              while (first < r - 1 && nums[first] <= minValue) ++first;
              while (last > l && nums[last] > minValue) --last;
              if (first >= last) break;
              swap(nums[first], nums[last]);
          }
          return sortValues(nums, first);
      }
  };
  
  // 自己写的太拉了
  //执行用时：4 ms, 在所有 C++ 提交中击败了47.44%的用户
  //内存消耗：8.1 MB, 在所有 C++ 提交中击败了49.00%的用户
  
  // 单指针
  class Solution {
  public:
      void sortColors(vector<int>& nums) {
      	int n = nums.size();
          int ptr = 0;
          for (int i = 0; i < n; i++) {
              if (nums[i] == 0) {
                  swap(nums[i], nums[ptr]);
                  ++ptr;
              }
          }
          for (int j = 0; j < n; j++) {
              if (nums[j] == 1) {
                  swap(nums[j], nums[ptr]);
                  ++ptr;
              }
          }
      }
  };
  // 时间换空间
  //执行用时：0 ms, 在所有 C++ 提交中击败了100.00%的用户
  //内存消耗：8.2 MB, 在所有 C++ 提交中击败了5.02%的用户
  
  // 双指针
  class Solution {
  public:
      void sortColors(vector<int>& nums) {
          int n = nums.size();
          int p0 = 0, p1 = 0;
          for (int i = 0; i < n; i++) {
              if (nums[i] == 1) {
                  swap(nums[i], nums[p1]);
                  ++p1;
              } else if (nums[i] == 0) {
                  swap(nums[i], nums[0]);
                  if (p0 < p1) {
                      swap(nums[i], nums[p1]);
                  }
                  ++p0;
                  ++p1;
              }
          }
      }
  };
  
  //执行用时：4 ms, 在所有 C++ 提交中击败了47.44%的用户
  //内存消耗：8.1 MB, 在所有 C++ 提交中击败了22.14%的用户
  ```

# 6. 一切皆可搜索

**深度优先搜索**和**广度优先搜索**是两种最常见的优先搜索方法，它们被广泛地运用在图和树等结构中进行搜索

## 6.1 深度优先搜索

+ **659 [岛屿的最大面积](https://leetcode-cn.com/problems/max-area-of-island/) (中等)**

  给你一个大小为 m x n 的二进制矩阵 grid 。

  岛屿 是由一些相邻的 1 (代表土地) 构成的组合，这里的「相邻」要求两个 1 必须在 水平或者竖直的四个方向上 相邻。你可以假设 grid 的四个边缘都被 0（代表水）包围着。

  岛屿的面积是岛上值为 1 的单元格的数目。

  计算并返回 grid 中最大的岛屿面积。如果没有岛屿，则返回面积为 0 。

  <img src="/Users/shuiguangshan/Pictures/Typora imgs/maxarea1-grid.jpg" alt="img" style="zoom:50%;" />

```
输入：grid = [[0,0,1,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,0,1,1,1,0,0,0],[0,1,1,0,1,0,0,0,0,0,0,0,0],[0,1,0,0,1,1,0,0,1,0,1,0,0],[0,1,0,0,1,1,0,0,1,1,1,0,0],[0,0,0,0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1,1,1,0,0,0],[0,0,0,0,0,0,0,1,1,0,0,0,0]]
输出：6
解释：答案不应该是 11 ，因为岛屿只能包含水平或垂直这四个方向上的 1 。
```

```c++
// 使用栈方法
class Solution {
public:
    vector<int> direction{-1, 0, 1, 0, -1};
    int maxAreaOfIsland(vector<vector<int>>& grid) {
        int m = grid.size(), n = m ? grid[0].size(): 0;
        int local_area, area = 0, x, y;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (grid[i][j]) {
                    grid[i][j] = 0;
                    local_area = 1;
                    stack<pair<int, int>> island;
                    island.push({i, j});
                    while(!island.empty()) {
                        auto [r, c] = island.top();
                        island.pop();
                        for (int k = 0; k < 4; ++k) {
                            x = direction[k], y = direction[k + 1];
                            if (x >= 0 && x < m &&
                                y >= 0 && y < n && grid[x][y] == 1) {
                                grid[x][y] = 0;
                                island.push({x, y});
                                ++local_area;
                            }
                        }
                    }
                    area = max(area, local_area);
                }
            }
        }
        return area;
    }
};

16 ms	26.1 MB
```

```c++
class Solution {
public:

    vector<int> direction{-1, 0, 1, 0, -1};
    int maxAreaOfIsland(vector<vector<int>>& grid) {
    	if (grid.empty() || grid[0].empty()) return 0;
        int max_area = 0;
        for (int i = 0; i < grid.size(); ++i) {
            for (int j = 0; j < grid[0].size(); ++j) {
                if (grid[i][j]) {
                    max_area = max(max_area, dfs(grid, i, j));
                }
            }
        }
        return max_area;
    }
    // 先搜索 然后在判断是不是越界了
    int dfs(vector<vector<int>>& grid, int r, int c) {
        if (grid[r][c] == 0) return 0;
        grid[r][c] = 0;
        int area = 1;
        int x, y;
        for (int k = 0; k < 4; ++k) {
            x = r + direction[k], y = c + direction[k + 1];
            if (x >= 0 && x < grid.size() && y >= 0 && y < grid[0].size()) {
                area += dfs(grid, x, y);
            }
        }
        return area;
    }
};
```

```c++
class Solution {
public:

    vector<int> direction{-1, 0, 1, 0, -1};
    int maxAreaOfIsland(vector<vector<int>>& grid) {
    	if (grid.empty() || grid[0].empty()) return 0;
        int max_area = 0;
        for (int i = 0; i < grid.size(); ++i) {
            for (int j = 0; j < grid[0].size(); ++j) {
                if (grid[i][j]) {
                    max_area = max(max_area, dfs(grid, i, j));
                }
            }
        }
        return max_area;
    }
    
    int dfs(vector<vector<int>>& grid, int r, int c) {
        if (r < 0 || r >= grid.size() ||
            c < 0 || c >= grid[0].size() || grid[r][c] == 0) return 0;
        grid[r][c] = 0;
        return 1 + dfs(grid, r - 1, c) + dfs(grid, r, c + 1) +
                   dfs(grid, r + 1, c) + dfs(grid, r, c - 1);
    }
};

//执行用时：12 ms, 在所有 C++ 提交中击败了93.97%的用户
//内存消耗：22.5 MB, 在所有 C++ 提交中击败了90.27%的用户
```

+ **547 [省份数量](https://leetcode-cn.com/problems/number-of-provinces/) (中等)**

  有 n 个城市，其中一些彼此相连，另一些没有相连。如果城市 a 与城市 b 直接相连，且城市 b 与城市 c 直接相连，那么城市 a 与城市 c 间接相连。

  省份 是一组直接或间接相连的城市，组内不含其他没有相连的城市。

  给你一个 n x n 的矩阵 isConnected ，其中 isConnected [i] [j] = 1 表示第 i 个城市和第 j 个城市直接相连，而 isConnected [i] [j] = 0 表示二者不直接相连。

  返回矩阵中 省份 的数量。

  ```
  输入：isConnected = [[1,1,0],[1,1,0],[0,0,1]]
  输出：2
  ```

  ```c++
  // 深度优先算法
  class Solution {
  public:
      int findCircleNum(vector<vector<int>>& isConnected) {
          int n = isConnected.size(), count = 0;
          vector<bool> visited(n, false);
          for (int i = 0; i < n; ++i) {
              if (!visited[i]) {
                  dfs(isConnected, i, visited);
                  ++count;
              }
          }
          return count;
      }
      
      void dfs(vector<vector<int>>& isConnected, int i, vector<bool> visited) {
          visited[i] = true;
          for (int k = 0; k < isConnected.size(); ++k) {
              if (isConnected[i][k] == 1 && !visited[k]) {
                  dfs(isConnected, k, visited);
              }
          }
      }
  };
  ```

  ```c++
  // 并查集的应用
  
  class UnionFind {
  public:
      int find(int x) {
          int root = x;
          
          while (father[root] != -1) {
              root = father[root];
          }
          
          while (father[x] != root) {
              int original_father = father[x];
              father[x] = root;
              x = original_father;
          }
          return root;
      }
      
      bool is_connected(int x, int y) {
          if (find(x) == find(y)) {
              return true;
          }
          return false;
      }
      
      void merge(int x, int y) {
          int root_x = find(x);
          int root_y = find(y);
          
          if (root_x != root_y) {
              father[root_x] = root_y;
              --num_of_sets;
          }
      }
      
      void add(int x){
          if (!father.count[x]) {
              father[x] = -1;
              ++num_of_sets;
          }
      }
      
      int get_num_of_sets() {
          return num_of_sets;
      }
     
  
  private:
      // 记录父节点
      unordered_map<int, int> father;
      // 记录集合的数量
      int num_of_sets = 0;
  }
  class Solution {
  public:
      int findCircleNum(vector<vector<int>>& isConnected) {
          UnionFind uf;
          for (int i = 0; i < isConnected.size(); ++i) {
              uf.add(i);
              for (int j = 0; j < i; ++j) {
                  if (isConnected[i][j]) {
                      uf.merge(i, j);
                  }
              }
          }
          return uf.get_num_of_sets();
      }
  };
  ```

+ **417 [太平洋大西洋水流问题](https://leetcode-cn.com/problems/pacific-atlantic-water-flow/)  （中等）**

  给定一个 m x n 的非负整数矩阵来表示一片大陆上各个单元格的高度。“太平洋”处于大陆的左边界和上边界，而“大西洋”处于大陆的右边界和下边界。

  规定水流只能按照上、下、左、右四个方向流动，且只能从高到低或者在同等高度上流动。

  请找出那些水流既可以流动到“太平洋”，又能流动到“大西洋”的陆地单元的坐标。

  ```
  给定下面的 5x5 矩阵:
  
    太平洋 ~   ~   ~   ~   ~ 
         ~  1   2   2   3  (5) *
         ~  3   2   3  (4) (4) *
         ~  2   4  (5)  3   1  *
         ~ (6) (7)  1   4   5  *
         ~ (5)  1   1   2   4  *
            *   *   *   *   * 大西洋
  
  返回:
  
  [[0, 4], [1, 3], [1, 4], [2, 2], [3, 0], [3, 1], [4, 0]] (上图中带括号的单元).
  ```

  ```c++
  class Solution {
  public:
      vector<int> direction{-1, 0, 1, 0, -1}; // 上 右 下 左
      vector<vector<int>> pacificAtlantic(vector<vector<int>>& heights) {
          if (heights.empty() || heights[0].empty()) return{};
          
          int m = heights.size(), n = heights[0].size();
          vector<vector<int>> ans;
          vector<vector<bool>> can_reach_p(m, vector<bool>(n, false));
          vector<vector<bool>> can_reach_a(m, vector<bool>(n, false));
          
          for (int i = 0; i < m; ++i) {
              dfs(heights, can_reach_p, i, 0);
              dfs(heights, can_reach_a, i, n - 1);
          }
          for (int i = 0; i < n; ++i) {
              dfs(heights, can_reach_p, 0, i);
              dfs(heights, can_reach_a, m - 1, i);
          }
          
          for (int i = 0; i < m; ++i) {
              for (int j = 0; j < n; ++j) {
                  if (can_reach_p[i][j] && can_reach_a[i][j]) {
                      ans.push_back(vector<int>{i, j});
                  }
              }
          }
          return ans;
      }
      void dfs(vector<vector<int>>& heights,  vector<vector<bool>>& can_reach, int r, int c) {
          if (can_reach[r][c]) return;
          can_reach[r][c] = true;
          int x, y;
          for (int k = 0; k < 4; ++k) {
              x = r + direction[k], y = c + direction[k + 1];
              if (x >= 0 && x < heights.size() &&
                  y >= 0 && y < heights[0].size() && 
                  heights[r][c] <= heights[x][y]) {
                  dfs(heights, can_reach, x, y);
              }
          }
      }
  };
  
  //执行用时：40 ms, 在所有 C++ 提交中击败了54.50%的用户
  //内存消耗：16.9 MB, 在所有 C++ 提交中击败了89.64%的用户
  
  ```

## 6.2 回溯法

+ **46 [全排列](https://leetcode-cn.com/problems/permutations/) (中等)**

  给定一个不含重复数字的数组 `nums` ，返回其 **所有可能的全排列** 。你可以 **按任意顺序** 返回答案。

  ```
  输入：nums = [1,2,3]
  输出：[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
  ```

  ```c++
  class Solution {
  public:
      vector<vector<int>> permute(vector<int>& nums) {
          vector<vector<int>> ans;
          backtracking(nums, 0, ans);
          return ans;
      }
      
      void backtracking(vector<int>& nums, int level, vector<vector<int>>& ans) {
          if (level == nums.size() - 1) {
              ans.push_back(nums);
              return;
          }
          
          for (int i = level; i < nums.size(); ++i) {
              swap(nums[i], nums[level]);
              backtracking(nums, level + 1, ans);
              swap(nums[i], nums[level]);
          }
      }
  };
  //执行用时：4 ms, 在所有 C++ 提交中击败了71.89%的用户
  //内存消耗：7.5 MB, 在所有 C++ 提交中击败了73.15%的用户
  ```

+ **77 [ 组合](https://leetcode-cn.com/problems/combinations/) (中等)**

  给定两个整数 `n` 和 `k`，返回范围 `[1, n]` 中所有可能的 `k` 个数的组合。

  你可以按 **任何顺序** 返回答案。

  ```
  输入：n = 4, k = 2
  输出：
  [
    [2,4],
    [3,4],
    [2,3],
    [1,2],
    [1,3],
    [1,4],
  ]
  ```

  ```c++
  class Solution {
  public:
      vector<vector<int>> combine(int n, int k) {
          vector<vector<int>> ans;
          vector<int> path;
          backtracking(ans, path, 1, n, k);
          return ans;
      }
      
      void backtracking(vector<vector<int>>& ans, vector<int>& path, int level, int& n, int& k) {
          if (path.size() == k) {
              ans.push_back(path);
              return;
          }
          for (int i = level; i <= n; ++i) {
              path.push_back(i);
              backtracking(ans, path, i + 1, n, k);
              path.pop_back();
          }
      }
  };
  
  //执行用时：28 ms, 在所有 C++ 提交中击败了24.03%的用户
  //内存消耗：8.7 MB, 在所有 C++ 提交中击败了98.68%的用户
  ```

+ **79 [单词搜索](https://leetcode-cn.com/problems/word-search/) (中等)**

  给定一个 m x n 二维字符网格 board 和一个字符串单词 word 。如果 word 存在于网格中，返回 true ；否则，返回 false 。

  单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。

  ```
  输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
  输出：true
  ```

  ```c++
  class Solution {
  public:
      bool exist(vector<vector<char>>& board, string word) {
      	if (board.empty() || board[0].empty()) return false;
          
          int m = board.size(), n = board[0].size();
          vector<vector<bool>> visited(m, vector<bool>(n, false));
          bool find = false;
          for (int i = 0; i < m; ++i) {
              for (int j = 0; j < n; ++j) {
                  backtracking(i, j, board, visited, word, find, 0);
              }
          }
          return find;
      }
      
      void backtracking(int i, int j, vector<vector<char>>& board, vector<vector<bool>>& visited, string& word, bool& find, int pos) {
          // 函数内剪枝
          if (i < 0 || i > board.size() || j < 0 || board[0].size()) return;
          if (visited[i][j] || find || board[i][j] != word[pos]) return;
          if (pos == word.size() - 1) {
              find = true;
              return;
          }
          // 探索
          visited[i][j] = true;
          backtracking(i + 1, j, board, visited, word, find, pos + 1);
          backtracking(i - 1, j, board, visited, word, find, pos + 1);
          backtracking(i, j + 1, board, visited, word, find, pos + 1);
          backtracking(i, j - 1, board, visited, word, find, pos + 1);
          // 回溯
          visited[i][j] = false;
      }
  };
  
  class Solution {
  public:
      bool exist(vector<vector<char>>& board, string word) {
          
      }
  };
  ```

+ **51 [ N 皇后](https://leetcode-cn.com/problems/n-queens/) （困难）**

  n 皇后问题 研究的是如何将 n 个皇后放置在 n×n 的棋盘上，并且使皇后彼此之间不能相互攻击(任意两个皇后不能位于同一行，同一列，同一斜线)。

  给你一个整数 n ，返回所有不同的 n 皇后问题 的解决方案。

  每一种解法包含一个不同的 n 皇后问题 的棋子放置方案，该方案中 'Q' 和 '.' 分别代表了皇后和空位。

  ```
  输入：n = 4
  输出：[[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]
  解释：如上图所示，4 皇后问题存在两个不同的解法。
  ```

  ```c++
  class Solution {
  public:
      vector<vector<string>> solveNQueens(int n) {
         	vector<vector<string>> ans;
          if (n == 0) return ans;
          
          vector<string> board(n, string(n, '.'));
          vector<bool> column(n, false), ldiag(2 * n - 1, false), rdiag(2 * n - 1, false);
          backtracking(ans, board, column, ldiag, rdiag, 0, n);
          return ans;
      }
      
      void backtracking(vector<vector<string>>& ans, string& board, vector<bool>& column, vector<bool>& ldiag, vector<bool>& rdiag, int row, int n) {
          if (row == n) {
              ans.push_back(board);
              return;
          }
          for (int i = 0; i < n; ++i) {
              // 这里的rdiag下标原书(row + i + 1)错了，虽然能够得到正确的答案可能是因为表格小
              if (column[i] || lidag[n - row + i - 1] || rdiag[row + i]) {
                  continue;
              }
              // 修改当前节点状态
              board[row][i] = 'Q';
              column[i] = lidag[n - row + i - 1] = rdiag[row + i] = true;
              
              // 递归子节点
              backtracking(ans, board, column, ldiag, rdiag, row + 1, n);
              
              // 复原当前节点状态
              board[row][i] = '.';
              column[i] = ldiag[n - row + i - 1] = rdiag[row + i] = false;     
          }
      }
  };
  //执行用时：0 ms, 在所有 C++ 提交中击败了100.00%的用户
  //内存消耗：6.9 MB, 在所有 C++ 提交中击败了96.94%的用户
  class Solution {
  public:
      vector<vector<string>> solveNQueens(int n) {
          
      }
  };
  ```

## 6.3 广度优先搜索

+ 934 [最短的桥](https://leetcode-cn.com/problems/shortest-bridge/) (中等)

  在给定的二维二进制数组 A 中，存在两座岛。（岛是由四面相连的 1 形成的一个最大组。）

  现在，我们可以将 0 变为 1，以使两座岛连接起来，变成一座岛。

  返回必须翻转的 0 的最小数目。（可以保证答案至少是 1 。）

  ```
  输入：A = [[0,1],[1,0]]
  输出：1
  ```

  ```c++
  class Solution {
  public:
      vector<int> direction{-1, 0, 1, 0, -1};
      int shortestBridge(vector<vector<int>>& grid) {
      	if (grid.empty() || grid[0].empty()) return 0;
          
          int m = grid.size(), n = grid[0].size();
          queue<pair<int, int>> points;
          bool flipped = false;
          // 寻找第一个岛屿 并把岛屿标记为2，记录里第一个岛屿最近的海
          for (int i = 0; i < m; ++i) {
              if (flipped) break;
              for (int j = 0; j < n; ++j) {
                  if (grid[i][j] == 1) {
                      dfs(grid, points, i, j);
                      flipped = true;
                      break;
                  }
              }
          }
          
          int level = 0;
          int n_points;
          while (!points.empty()) {
              n_points = points.size();
              ++level;
              while (n_points--) {
                  auto [r, c] = points.front();
                  points.pop();
                  for (int k = 0; k < 4; ++k) {
                      x = r + direction[k], y = c + direction[k + 1];
                      if (grid[x][y] == 2) {
                          continue;
                      }
                      if (grid[x][y] == 1) {
                          return level;
                      }
                      points.push({x, y});
                      grid[x][y] = 2;
                  }
              }
          }
      	return 0;  
      }
      
      void dfs(vector<vector<int>>& grid, queue<pair<int, int>>& points, int r, int c) {
          if (r < 0 || r >= grid.size() ||
              c < 0 || c >= grid[0].size() ||
              grid[r][c] == 2) {
              return;
          }
          if (grid[r][c] == 0) {
              points.push({r, c});
              return;
          }
          grid[r][c] = 2;
          dfs(grid, points, r - 1, c);
          dfs(grid, points, r, c + 1);
          dfs(grid, points, r + 1, c);
          dfs(grid, points, r, c - 1);
      }
  };
  ```

+ **126 [ 单词接龙 II](https://leetcode-cn.com/problems/word-ladder-ii/) (困难)**

  按字典 wordList 完成从单词 beginWord 到单词 endWord 转化，一个表示此过程的转换序列是形式上像beginWord -> s1 -> s2 -> ... -> sk 这样的单词序列，并满足：
  + 每对相邻的单词之间仅有单个字母不同。
  + 转换过程中的每个单词 si（1 <= i <= k）必须是字典 wordList 中的单词。注意，beginWord 不必是字典 wordList 中的单词。
  + sk == endWord

  给你两个单词 beginWord 和 endWord ，以及一个字典 wordList 。请你找出并返回所有从 beginWord 到 endWord 的 **最短转换序**列 ，如果不存在这样的转换序列，返回一个空列表。每个序列都应该以单词列表 [beginWord, s1, s2, ..., sk] 的形式返回。

  ```
  按字典 wordList 完成从单词 beginWord 到单词 endWord 转化，一个表示此过程的 转换序列 是形式上像 beginWord -> s1 -> s2 -> ... -> sk 这样的单词序列，并满足：
  
  每对相邻的单词之间仅有单个字母不同。
  转换过程中的每个单词 si（1 <= i <= k）必须是字典 wordList 中的单词。注意，beginWord 不必是字典 wordList 中的单词。
  sk == endWord
  给你两个单词 beginWord 和 endWord ，以及一个字典 wordList 。请你找出并返回所有从 beginWord 到 endWord 的 最短转换序列 ，如果不存在这样的转换序列，返回一个空列表。每个序列都应该以单词列表 [beginWord, s1, s2, ..., sk] 的形式返回。
  ```

  ```c++
  class Solution {
  public:
      vector<vector<string>> findLadders(string beginWord, string endWord, vector<string>& wordList) {
  		vector<vector<string>> ans;
          unordered_set<string> dict;
          for (const auto& word: wordList) {
              dict.push_back(word);
          }
          if (!dict.count(endWord)) return ans;
          dict.erase(beginWord);
          dict.erase(endWord);
          unordered_set<string> beginVisited{beginWord}, endVisited{endWord};
          unordered_map<string, vector<string>> nextLevel;
          bool found = false, reverse = false;
          while (!beginVisited.empty()) {
              unordered_set<string> q;
              for (const auto& word: beginVisited) {
                  string s = word;
                  for (int i = 0; i < s.size(); ++i) {
                      char ch = s[i];
                      for (int j = 0; j < 26; ++j) {
                          s[i] = j + 'a';
                          if (endVisited.count(s)) {
                              reverse ? nextLevel[s].push_back(word): nextLevel[word].push_back(s);
                              found = true;
                          }
                          if (dict.count(s)) {
                              reverse ? nextLevel[s].push_back(word): nextLevel[word].push_back(s);
                          }
                      }
                      s[i] = ch;
                  }
              }
              if (found) {
                  break;
              }
              for (const auto& w: q) {
                  dict.erase(w);
              }
              if (q.size() <= endVisited.size()) {
                  beginVisited = q;
              } else {
                  beginVisited = endVisited;
                  endVisited = q;
              }  
          }
          if (found) {
              vector<string> path{beginWord};
              backtracking(beginWord, endWord, nextLevel, path, ans);
          }
          return ans;
      }
      
      void backtracking(string beginWord, string endWord, unordered_map<string, vector<string>>& nextLevel,
                        vector<string>& path, vector<vector<string>>& ans) {
          if (beginWord == endWord) {
              ans.push_back(path);
              return;
          }
          for (const auto & word: nextLevel[beginWord]) {
              path.push_back(word);
              backtracking(word, endWord, nextLevel, path, ans);
              path.pop_back();
          }
          return;
      }
  };
  
  //执行用时：4 ms, 在所有 C++ 提交中击败了97.72%的用户
  //内存消耗：8.3 MB, 在所有 C++ 提交中击败了90.47%的用户
  ```

## 6.4 练习

+ **130 [被围绕的区域](https://leetcode-cn.com/problems/surrounded-regions/) (中等)**

  给你一个 `m x n` 的矩阵 `board` ，由若干字符 `'X'` 和 `'O'` ，找到所有被 `'X'` 围绕的区域，并将这些区域里所有的 `'O'` 用 `'X'` 填充。

  输入：board = [["X","X","X","X"],["X","O","O","X"],["X","X","O","X"],["X","O","X","X"]]
  输出：[["X","X","X","X"],["X","X","X","X"],["X","X","X","X"],["X","O","X","X"]]
  解释：被围绕的区间不会存在于边界上，换句话说，任何边界上的 'O' 都不会被填充为 'X'。 任何不在边界上，或不与边界上的 'O' 相连的 'O' 最终都会被填充为 'X'。如果两个元素在水平或垂直方向相邻，则称它们是“相连”的。

```c++
class Solution {
    public:
    vector<int> direction{-1, 0, 1, 0, -1};
    void solve(vector<vector<char>>& board) {
        int m = board.size(), n = board[0].size();
        if (m == 0) return;
        for (int i = 0; i < m; ++i) {
            backtracking(board, i, 0);
            backtracking(board, i, n - 1);
        }
        for (int j = 0; j < n; ++j) {
            backtracking(board, 0, j);
            backtracking(board, m - 1, j);
        }
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (board[i][j] == 'A') {
                    board[i][j] = 'O';
                } else if (board[i][j] == 'O') {
                    board[i][j] = 'X';
                }
            }
        }

    }
    void backtracking(vector<vector<char>>& board, int r, int c) {
        if ( r < 0 || r >= board.size() ||
            c < 0 || c >= board[0].size() ||
            board[r][c] != 'O') {
            return;
        }
        board[r][c] = 'A';
        int x, y;
        for (int k = 0; k < 4; ++k) {
            x = r + direction[k], y = c + direction[k + 1];
            backtracking(board, x, y);
        }
    }
};

//执行用时：12 ms, 在所有 C++ 提交中击败了80.87%的用户
//内存消耗：9.7 MB, 在所有 C++ 提交中击败了83.11%的用户
```

+  **257 [二叉树的所有路径](https://leetcode-cn.com/problems/binary-tree-paths/) (简单)**

  给你一个二叉树的根节点 `root` ，按 **任意顺序** ，返回所有从根节点到叶子节点的路径。

  **叶子节点** 是指没有子节点的节点。

  <img src="/Users/shuiguangshan/Documents/GitHub/Leetcode/pics/image-20211029171950395.png" alt="image-20211029171950395" style="zoom:40%;" />

```
输入：root = [1,2,3,null,5]
输出：["1->2->5","1->3"]
```

```c++
// 回溯法
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    vector<string> binaryTreePaths(TreeNode* root) {
        vector<string> paths;
        string path = to_string(root->val);
        if (root->left == nullptr && root->right == nullptr) {
            paths.push_back(path);
            return paths;
        }
        backtracking(root, path, paths);
        return paths;
    }
    
    void backtracking(TreeNode* node, string& path, vector<string>& paths) {
        if (node->left == nullptr && node->right == nullptr) {
            paths.push_back(path);
            return paths;
        }
        if (node->left != nullptr) {
            path += "->" + to_string(node->left->val);
            backtracking(node->left, path, paths);
            path.erase(path.size() - (2 + to_string(node->left->val).size()), path.size());
        }
        if (node->right != nullptr) {
            path += "->" + to_string(node->right->val);
            backtracking(node->right, path, paths);
            path.erase(path.size() - (2 + to_string(node->right->val).size()), path.size());
        }
    }
};

// 执行用时：4 ms, 在所有 C++ 提交中击败了74.41%的用户
// 内存消耗：11.9 MB, 在所有 C++ 提交中击败了92.05%的用户
```

+ **47 [全排列 II](https://leetcode-cn.com/problems/permutations-ii/) （中等）**

  给定一个可包含重复数字的序列 `nums` ，**按任意顺序** 返回所有不重复的全排列。

  ```
  输入：nums = [1,1,2]
  输出：
  [[1,1,2],
   [1,2,1],
   [2,1,1]]
  ```

  ```c++
  class Solution {
  public:
      vector<vector<int>> permuteUnique(vector<int>& nums) {
          vector<vector<int>> ans;
          vector<int> path;
          unordered_map<int, int> numCount;
          int n = nums.size();
  
          if (nums.size() <= 1) {
              ans.push_back(nums);
              return ans;
          }
          for (const auto & num: nums) {
              numCount[num] += 1;
          }
          backtracking(ans, numCount, path, n);
          return ans;
      }
  
      void backtracking(vector<vector<int>>& ans, unordered_map<int, int> &numCount, 
                        vector<int> path, int& n) {
          if (path.size() == n) {
              ans.push_back(path);
              return;
          }
          for (auto & k: numCount) {
              if (k.second != 0) {
                  path.push_back(k.first);
                  --k.second;
                  backtracking(ans, numCount, path, n);
                  path.pop_back();
                  ++k.second;
              }
          }
      }
  };
  // 执行用时：8 ms, 在所有 C++ 提交中击败了59.87%的用户
  // 内存消耗：9.9 MB, 在所有 C++ 提交中击败了17.60%的用户
  
  
  class Solution {
      vector<int> vis;
  public:
      vector<vector<int>> permuteUnique(vector<int>& nums) {
          vector<vector<int>> ans;
          vector<int> perm;
          sort(nums.begin(), nums.end());
          backtracking(ans, nums, idx, perm);
          return ans;
      }
      
      void backtracking(vector<vector<int>>& ans, vector<int>& nums, int idx, vector<int>& perm) {
          if (idx == nums.size()) {
              ans.push_back(prem);
              return ;
          }
          for (int i = 0; i < nums.size(); ++i) {
              if (vis[i] || (i > 0 && nums[i] == nums[i - 1] && !vis[i - 1])) {
                  continue;
              }
              perm.push_back(nums[i]);
              vis[i] = 1;
              backtracking(ans, nums, idx + 1, perm);
              vis[i] = 0;
              perm.pop_back();
          }
      }
  };
  
  // 执行用时：4 ms, 在所有 C++ 提交中击败了93.98%的用户
  // 内存消耗：8.1 MB, 在所有 C++ 提交中击败了98.14%的用户
  ```

+ **40 [组合总和 II](https://leetcode-cn.com/problems/combination-sum-ii/)(中等)**

  给定一个数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。

  candidates 中的每个数字在每个组合中只能使用一次。

  注意：解集不能包含重复的组合。 

  ```
  输入: candidates = [10,1,2,7,6,1,5], target = 8,
  输出:
  [
  [1,1,6],
  [1,2,5],
  [1,7],
  [2,6]
  ]
  ```

  ```c++
  class Solution {
      vector<int> vis;
  public:
      vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
      	vector<vector<int>> ans;
          vector<int> path;
          vis.resize(candidates.size());
          sort(candidates.begin(), candidates.end());
          backtracking(ans, candidates, path, 0, target);
          return ans;
      }
      
      void backtracking(vector<vector<int>>& ans, vector<int>& candidates, vector<int>& path, int level, 
                        int& target) {
          int sum = accumulate(path.begin(), path.end(), 0);
          if (sum == target) {
              ans.push_back(path);
              return;
          }
          
          for (int i = 0; i < candidates.size(); ++i) {
              if (vis[i] || (i > 0 && candidates[i] == candidates[i - 1] && !vis[i - 1]) ||
                  (sum + candidates[i]) > target) {
                  continue;
              }
              vis[i] = 1;
              path.push_back(candidates[i]);
              backtracking(ans, candidates, path, i + 1, target);
              path.pop_back();
              vis[i] = 0;
          }
      }
  };
  
  // 执行用时：4 ms, 在所有 C++ 提交中击败了88.24%的用户
  // 内存消耗：10.4 MB, 在所有 C++ 提交中击败了74.26%的用户
  ```

+ **37 [解数独](https://leetcode-cn.com/problems/sudoku-solver/) (困难)**

  编写一个程序，通过填充空格来解决数独问题。

  数独的解法需 遵循如下规则：

  数字 1-9 在每一行只能出现一次。
  数字 1-9 在每一列只能出现一次。
  数字 1-9 在每一个以粗实线分隔的 3x3 宫内只能出现一次。（请参考示例图）
  数独部分空格内已填入了数字，空白格用 '.' 表示

  ```
  输入：board = [["5","3",".",".","7",".",".",".","."],["6",".",".","1","9","5",".",".","."],[".","9","8",".",".",".",".","6","."],["8",".",".",".","6",".",".",".","3"],["4",".",".","8",".","3",".",".","1"],["7",".",".",".","2",".",".",".","6"],[".","6",".",".",".",".","2","8","."],[".",".",".","4","1","9",".",".","5"],[".",".",".",".","8",".",".","7","9"]]
  输出：[["5","3","4","6","7","8","9","1","2"],["6","7","2","1","9","5","3","4","8"],["1","9","8","3","4","2","5","6","7"],["8","5","9","7","6","1","4","2","3"],["4","2","6","8","5","3","7","9","1"],["7","1","3","9","2","4","8","5","6"],["9","6","1","5","3","7","2","8","4"],["2","8","7","4","1","9","6","3","5"],["3","4","5","2","8","6","1","7","9"]]
  ```

  ```c++
  class Solution {
  public:
  
      bitset<9> getPossibleStatus(int x, int y) {
          return ~(rows[x] | cols[y] | cells[x / 3][y / 3]);
      }
  
      vector<int> getNext(vector<vector<char>>& board) {
          // 找出回溯最少的位置进行填充
          // 标准就是，其横 竖 3x3宫格内填充的种类最多，那么留给当前空的数字就最少
          // 探索的空间就相应的减少
          vector<int> ret;
          int minCont = 10;
          for (int i = 0; i < board.size(); ++i) {
              for (int j = 0; j < board[0].size(); ++j) {
                  if (board[i][j] != '.') continue;
                  auto cur = getPossibleStatus(i, j);
                  if (cur.count() >= minCont) continue;
                  ret = {i, j};
                  minCont = cur.count();
              }
          }
          return ret;
      }
  
      void fillNum(int x, int y, int n, bool fillFlag) {
          rows[x][n] = (fillFlag) ? 1: 0;
          cols[y][n] = (fillFlag) ? 1: 0;
          cells[x/3][y/3][n] = (fillFlag) ? 1: 0;
      }
  
      bool dfs(vector<vector<char>>& board, int cnt) {
          if (cnt == 0) return true;
  
          // 在当前表格中找出回溯最少的位置进行填充
          auto next = getNext(board);
          auto bits = getPossibleStatus(next[0], next[1]);
          for (int i = 0; i < bits.size(); ++i) {
              if (!bits.test(i)) continue;
              // 找到当前位置 next[0] next[1] 可以填充的数字
              // 更新 3个状态表格
              fillNum(next[0], next[1], i, true);
              // 填充board
              board[next[0]][next[1]] = i + '1';
              // 递归子节点 满足条件就直接return 因为题目没有要求找出所有的情况
              if (dfs(board, cnt - 1)) return true;
              // 回溯
              board[next[0]][next[1]] = '.';
              fillNum(next[0], next[1], i, false);
          }
          return false;
      }
  
      void solveSudoku(vector<vector<char>>& board) {
          // 用于状态检查 初始化
          rows = vector<bitset<9>> (9, bitset<9>());
          cols = vector<bitset<9>> (9, bitset<9>());
          cells = vector<vector<bitset<9>>> (3, vector<bitset<9>>(3, bitset<9>()));
  
          // 记录总共需要填入多少个数，作为结束的判断
          int cnt = 0;
          for (int i = 0; i < board.size(); ++i) {
              for (int j = 0; j < board[0].size(); ++j) {
                  cnt += (board[i][j] == '.');
                  if (board[i][j] == '.') continue;
                  // 更新检查列的状态
                  int n = board[i][j] - '1';
                  // bitset 中的下标是从右向左 n - 0  所以这边的移位是反的
                  rows[i] |= (1 << n);
                  cols[j] |= (1 << n);
                  cells[i / 3][j / 3] |= (1 << n);
              }
          }
          dfs(board, cnt);
      }
  
  
  private:
      vector<bitset<9>> rows;
      vector<bitset<9>> cols;
      vector<vector<bitset<9>>> cells;
  
  };
  
  //执行用时：0 ms, 在所有 C++ 提交中击败了100.00%的用户
  //内存消耗：6.6 MB, 在所有 C++ 提交中击败了11.37%的用户
  ```

+ **310 [最小高度树](https://leetcode-cn.com/problems/minimum-height-trees/) (中等)**

  树是一个无向图，其中任何两个顶点只通过一条路径连接。 换句话说，一个任何没有简单环路的连通图都是一棵树。

  给你一棵包含 n 个节点的树，标记为 0 到 n - 1 。给定数字 n 和一个有 n - 1 条无向边的 edges 列表（每一个边都是一对标签），其中 edges[i] = [ai, bi] 表示树中节点 ai 和 bi 之间存在一条无向边。

  可选择树中任何一个节点作为根。当选择节点 x 作为根节点时，设结果树的高度为 h 。在所有可能的树中，具有最小高度的树（即，min(h)）被称为 最小高度树 。

  请你找到所有的 最小高度树 并按 任意顺序 返回它们的根节点标签列表。

  树的 高度 是指根节点和叶子节点之间最长向下路径上边的数量。

  ```
  输入：n = 4, edges = [[1,0],[1,2],[1,3]]
  输出：[1]
  解释：如图所示，当根是标签为 1 的节点时，树的高度是 1 ，这是唯一的最小高度树。
  ```

  ```c++
  class Solution {
  public:
  
  
      vector<int> findMinHeightTrees(int n, vector<vector<int>>& edges) {
          if (n == 1) return {0};
  
          // 每个节点对应的度
          vector<int> degree(n);
          // 邻接表
          map<int, vector<int>> adjMatrix;
          vector<int> ans;
          for (int i = 0; i < edges.size(); ++i) {
              int u = edges[i][0];
              int v = edges[i][1];
              ++degree[u];
              ++degree[v];
              adjMatrix[u].push_back(v);
              adjMatrix[v].push_back(u);
          }
          // 叶子结点
          queue<int> leaves;
          for (int i = 0; i < n; ++i) {
              if(degree[i] == 1) {
                  leaves.push(i);
              }
          }
          while(!leaves.empty()) {
              ans.clear();
              int sz = leaves.size();
              for (int i = 0; i < sz; ++i) {
                  int leaf = leaves.front();
                  leaves.pop();
                  ans.push_back(leaf);
                  degree[leaf]--;
                  // 与其相连的节点度减一
                  for (auto adjLeaf: adjMatrix[leaf]) {
                      degree[adjLeaf]--;
                      if (degree[adjLeaf] == 1) {
                          leaves.push(adjLeaf);
                      }
                  }
                  // 最后只会保留 度为0 的顶点 这也是我们要找的点
              }
          }
          return ans;
      }
  };
  
  // 执行用时：72 ms, 在所有 C++ 提交中击败了52.43%的用户
  // 内存消耗：30.9 MB, 在所有 C++ 提交中击败了29.07%的用户
  ```

# 7. DP

+ 确定状态 
+ 确定动作
+ 在当前状态下采取可能的所有动作 取最优就是状态转移方程
+ 动态规划也是通过递推的方式实现，所以边界条件也要确定清楚
+ 无后效性：当前的状态仅依赖于上一状态和动作



## 7.1 基本动态规划：一维

+ **70 [ 爬楼梯](https://leetcode-cn.com/problems/climbing-stairs/) (简单)**

  假设你正在爬楼梯。需要 *n* 阶你才能到达楼顶。

  每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？

  **注意：**给定 *n* 是一个正整数。

  ```
  输入： 2
  输出： 2
  解释： 有两种方法可以爬到楼顶。
  1.  1 阶 + 1 阶
  2.  2 阶
  ```

  ```c++
  class Solution {
  public:
      int climbStairs(int n) {
          if (n <= 2) {
              return n;
          }
          int pre2 = 1, pre1 = 2, cur;
          for (int i = 3; i <= n; ++i) {
              cur = pre2 + pre1;
              pre2 = pre1;
              pre1 = cur;
          }
          return cur;  
      }
  };
  
  // 执行用时：4 ms, 在所有 C++ 提交中击败了100.00%的用户
  // 内存消耗：5.8 MB, 在所有 C++ 提交中击败了73.18%的用户
  ```

+ **198 [打家劫舍](https://leetcode-cn.com/problems/house-robber/) (中等)**

  你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。

  给定一个代表每个房屋存放金额的非负整数数组，计算你 不触动警报装置的情况下 ，一夜之内能够偷窃到的最高金额

  ```
  输入：[1,2,3,1]
  输出：4
  解释：偷窃 1 号房屋 (金额 = 1) ，然后偷窃 3 号房屋 (金额 = 3)。
       偷窃到的最高金额 = 1 + 3 = 4 。
  ```

  ```c++
  class Solution {
  public:
      int rob(vector<int>& nums) {
          if (nums.empty()) return 0;
          int n = nums.size();
          if (n == 1) return nums[0];
          int pre1 = 0, pre2 = 0, cur;
          for (int i = 0; i < nums.size(); ++i) {
              cur = max(pre1, pre2 + nums[i]);
              pre2 = pre1;
              pre1 = cur;
          }
          return cur;
      }
  };
  
  // 执行用时：4 ms, 在所有 C++ 提交中击败了39.67%的用户
  // 内存消耗：7.3 MB, 在所有 C++ 提交中击败了99.76%的用户
  ```

+ 413 等差数列划分

  如果一个数列 至少有三个元素 ，并且任意两个相邻元素之差相同，则称该数列为等差数列。

  例如，[1,3,5,7,9]、[7,7,7,7] 和 [3,-1,-5,-9] 都是等差数列。
  给你一个整数数组 nums ，返回数组 nums 中所有为等差数组的 子数组 个数。

  **子数组** 是数组中的一个**连续序列。**

  ```
  输入：nums = [1,2,3,4]
  输出：3
  解释：nums 中有三个子等差数组：[1, 2, 3]、[2, 3, 4] 和 [1,2,3,4] 自身。
  ```

  ```c++
  class Solution {
  public:
      int numberOfArithmeticSlices(vector<int>& nums) {
          if (nums.empty() || nums.size() <= 2) return 0;
          int n = nums.size();
          vector<int> dp(n, 0);
          for (int i = 2; i < n; ++i) {
              dp[i] = (nums[i] - nums[i - 1] == nums[i - 1] - nums[i - 2]) ? dp[i - 1] + 1: 0;
          }
          return accumulate(dp.begin(), dp.end(), 0);
      }
  };
  
  // 执行用时：4 ms, 在所有 C++ 提交中击败了45.24%的用户
  // 内存消耗：7.2 MB, 在所有 C++ 提交中击败了40.85%的用户
  ```

## 7.2 基本动态规划: 二维

+ **64 [最小路径和](https://leetcode-cn.com/problems/minimum-path-sum/) (中等)**

  给定一个包含非负整数的 `*m* x *n*` 网格 `grid` ，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。

  **说明：**每次只能向下或者向右移动一步。

  ```
  输入：grid = [[1,3,1],[1,5,1],[4,2,1]]
  输出：7
  解释：因为路径 1→3→1→1→1 的总和最小。
  ```

  ```c++
  class Solution {
  public:
      int minPathSum(vector<vector<int>>& grid) {
  
          if (grid.empty() || grid[0].empty()) return 0;
          int m = grid.size(), n = grid[0].size();
  
          vector<vector<int>> ans(m, vector<int>(n, 99999));
          ans[0][0] = grid[0][0];
          for (int i = 0; i < m; ++i) {
              for (int j = 0; j < n; ++j) {
                  int x_left = i, y_left = j - 1;
                  int x_up = i - 1, y_up = j;
                  if (x_left >= 0 && x_left < m &&
                      y_left >= 0 && y_left < n) {
                      ans[i][j] = min(ans[i][j], ans[x_left][y_left] + grid[i][j]);
                  }
                  if (x_up >= 0 && x_up < m &&
                      y_up >= 0 && y_up < n) {
                      ans[i][j] = min(ans[i][j], ans[x_up][y_up]+ grid[i][j]);
                  }
              }
          }
          return ans[m - 1][n - 1];
      }
  };
  // 执行用时：8 ms, 在所有 C++ 提交中击败了81.61%的用户
  // 内存消耗：9.9 MB, 在所有 C++ 提交中击败了18.41%的用户
  ```

+ **542 [01 矩阵](https://leetcode-cn.com/problems/01-matrix/) (中等)**

  给定一个由 0 和 1 组成的矩阵 mat ，请输出一个大小相同的矩阵，其中每一个格子是 mat 中对应位置元素到最近的 0 的距离。

  两个相邻元素间的距离为 1 。

  ```
  输入：mat = [[0,0,0],[0,1,0],[0,0,0]]
  输出：[[0,0,0],[0,1,0],[0,0,0]]
  ```

  ```c++
  class Solution {
  public:
      vector<vector<int>> updateMatrix(vector<vector<int>>& mat) {
  		int m = mat.size(), n = mat[0].size();
          vector<vector<int>> dp(m, vector<int>(n, INT_MAX - 1));
          for (int i = 0; i < m; ++i) {
              for (int j = 0; j < n; ++j) {
                  if (mat[i][j] == 0) {
                      dp[i][j] = 0;
                  } else {
                      if (i > 0) {
                          dp[i][j] = min(dp[i][j], dp[i - 1][j] + 1);
                      }
                      if (j > 0) {
                          dp[i][j] = min(dp[i][j], dp[i][j - 1] + 1);
                      }
                  }
              }
          }
          for (int i = m - 1; i >= 0; ++i) {
              for (int j = n - 1; j >= 0; ++j) {
                  if (mat[i][j] == 1) {
                      if (i < m - 1) {
                          dp[i][j] = min(dp[i][j], dp[i + 1][j] + 1);
                      }
                      if (j < n - 1) {
                          dp[i][j] = min(dp[i][j], dp[i][j + 1] + 1);
                      }
                  }
              }
          }
          return dp;
      }
  };
  
  // 执行用时：64 ms, 在所有 C++ 提交中击败了78.61%的用户
  // 内存消耗：26.9 MB, 在所有 C++ 提交中击败了80.88%的用
  ```

+ **221 [最大正方形](https://leetcode-cn.com/problems/maximal-square/) (中等)**

  在一个由 `'0'` 和 `'1'` 组成的二维矩阵内，找到只包含 `'1'` 的最大正方形，并返回其面积。

  ```
  输入：matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
  输出：4
  ```

  ```c++
  class Solution {
  public:
      int maximalSquare(vector<vector<char>>& matrix) {
          if (matrix.empty() || matrix[0].empty()) return 0;
          int m = matrix.size(), n = matrix[0].size();
          vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));
          int max_side = 0;
          for (int i = 1; i < m + 1; ++i) {
              for (int j = 1; j < n + 1; ++j) {
                  if (matrix[i - 1][j - 1] == '1') {
                      dp[i][j] = min(dp[i-1][j-1], min(dp[i-1][j], dp[i][j-1])) + 1;
                  }
                  max_side = max(max_side, dp[i][j]);
              }
          }
          return max_side * max_side;
      }
  };
  
  // 执行用时：24 ms, 在所有 C++ 提交中击败了67.43%的用户
  // 内存消耗：11.6 MB, 在所有 C++ 提交中击败了46.41%的用户
  ```

通过补列的方式，可以将边界上的点都更新到（221），如果不补列的话就需要用更多的if条件进行更新（542）

## 7.4 分割类型题

+ **279 [完全平方数](https://leetcode-cn.com/problems/perfect-squares/)(中等)**

  给定正整数 n，找到若干个完全平方数（比如 1, 4, 9, 16, ...）使得它们的和等于 n。你需要让组成和的完全平方数的个数最少。

  给你一个整数 n ，返回和为 n 的完全平方数的 最少数量 。

  完全平方数是一个整数，其值等于另一个整数的平方；换句话说，其值等于一个整数自乘的积。例如，1、4、9 和 16 都是完全平方数，而 3 和 11 不是。

  ```
  输入：n = 12
  输出：3 
  解释：12 = 4 + 4 + 4
  ```

  ```c++
  class Solution {
  public:
      int numSquares(int n) {
  		vector<int> dp(n + 1, INT_MAX - 1);
          dp[0] = 0;
          for (int i = 0; i <= n; ++i) {
              for (int j = 0; j * j <= i; ++j) {
                  dp[i] = min(dp[i], dp[i - j * j] + 1);
              }
          }
          return dp[n];
      }
  };
  
  // 执行用时：224 ms, 在所有 C++ 提交中击败了15.48%的用户
  // 内存消耗：8.8 MB, 在所有 C++ 提交中击败了75.83%的用户
  ```

+ **91 [解码方法](https://leetcode-cn.com/problems/decode-ways/)(中等)**

  一条包含字母 `A-Z` 的消息通过以下映射进行了 **编码** ：

  ```
  'A' -> 1
  'B' -> 2
  ...
  'Z' -> 26
  ```

  要 解码 已编码的消息，所有数字必须基于上述映射的方法，反向映射回字母（可能有多种方法）。例如，"11106" 可以映射为：

  + "AAJF" ，将消息分组为 (1 1 10 6)
  + "KJF" ，将消息分组为 (11 10 6)
    注意，消息不能分组为  (1 11 06) ，因为 "06" 不能映射为 "F" ，这是由于 "6" 和 "06" 在映射中并不等价。

  给你一个只含数字的 非空 字符串 s ，请计算并返回 解码 方法的 总数 。

  题目数据保证答案肯定是一个 32 位 的整数。

  ```
  输入：s = "12"
  输出：2
  解释：它可以解码为 "AB"（1 2）或者 "L"（12）。
  ```

  ```c++
  class Solution {
  public:
      int numDecodings(string s) {
      	int n = s.length();
          if (n == 0) return 0;
          int pre = s[0] - '0';
          if (!pre) return 0;
          if (n == 1) return 1;
          // 到这里n 至少大于或等于2，
          vector<int> dp(n + 1, 1);
          for (int i = 2; i <= n; ++i) {
              int cur = s[i - 1] - '0';
              if ((pre == 0 || pre > 2) && cur == 0) {
                  return 0;
              }
              if ((pre == 1) || (pre == 2 && cur < 7)) {
                  if (cur) {
                      dp[i] = dp[i - 1] + dp[i - 2];
                  } else {
                      dp[i] = dp[i - 2];
                  }
              } else {
                  dp[i] = dp[i - 1];
              }
              pre = cur;
          }
          return dp[n];
      }
  };
  ```

+ **139 [单词拆分](https://leetcode-cn.com/problems/word-break/) (中等)**

  给你一个字符串 s 和一个字符串列表 wordDict 作为字典，判定 s 是否可以由空格拆分为一个或多个在字典中出现的单词。

  说明：拆分时可以重复使用字典中的单词。

  ```
  输入: s = "catsandog", wordDict = ["cats", "dog", "sand", "and", "cat"]
  输出: false
  ```

  ```c++
  class Solution {
  public:
      bool wordBreak(string s, vector<string>& wordDict) {
      	int n = s.length();
          vector<bool> dp(n + 1, false);
          dp[0] = true;
          for (int i = 1; i < n + 1; ++i) {
              for (const string & word: wordDict) {
                  int len = word.length();
                  if (i >= len && s.sutstr(i - len, len) == word) {
                      dp[i] = dp[i] || dp[i - len]
                  }
              }
          }
          return dp[n];
      }
  };
  // 执行用时：4 ms, 在所有 C++ 提交中击败了93.77%的用户
  // 内存消耗：7.2 MB, 在所有 C++ 提交中击败了99.53%的用户
  ```

## 7.5 子序列问题

+ **300 [最长递增子序列](https://leetcode-cn.com/problems/longest-increasing-subsequence/) (中等)**

  给你一个整数数组 nums ，找到其中最长严格递增子序列的长度。

  子序列是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。例如，[3,6,2,7] 是数组 [0,3,1,6,2,2,7] 的子序列。

  定义 $\textit{dp}[i]$ 为考虑前 i个元素，以第 i 个数字结尾的最长上升子序列的长度，**注意 $\textit{nums}[i]$ 必须被选取。**

  ```
  输入：nums = [10,9,2,5,3,7,101,18]
  输出：4
  解释：最长递增子序列是 [2,3,7,101]，因此长度为 4 。
  ```

  ```c++
  class Solution {
  public:
      int lengthOfLIS(vector<int>& nums) {
      	int max_length = 0, n = nums.size();
          if (n <= 0) return 0;
          vector<int> dp(n, 1);
          for (int i = 0; i < n; ++i) {
              for (int j = 0; j < i; ++j) {
                  if (nums[i] > nums[j]) {
                      dp[i] = max(dp[i], dp[j] + 1);
                  }
              }
              max_length = max(max_length, dp[i]);
          }
          return max_length;
      }
  };
  // 执行用时：264 ms, 在所有 C++ 提交中击败了51.06%的用户
  // 内存消耗：10.3 MB, 在所有 C++ 提交中击败了14.88%的用户
  
  
  class Solution {
  public:
      int lengthOfLIS(vector<int>& nums) {
          int len = 1, n = nums.size();
          if (n <= 0) return 0;
          vector<int> d(n + 1, 0);
          d[len] = nums[0];
          for (int i = 1; i < n; i++) {
              if (nums[i] > d[len]) {
                  d[++len] = nums[i];
              } else {
                  int l = 1, r = len, pos = 0;
                  while (l < r) {
                      int middle = (l + r) / 2;
                      if (d[middle] < nums[i]) {
                          pos = middle;
                          l = middle + 1;
                      } else {
                          r = middle;
                      }
                  }
                  d[pos + 1] = nums[i];
              }
          }
          return len;
      }
  };
  
  // 执行用时：8 ms, 在所有 C++ 提交中击败了92.04%的用户
  // 内存消耗：10.3 MB, 在所有 C++ 提交中击败了41.16%的用户
  ```

+ **1143 [最长公共子序列](https://leetcode-cn.com/problems/longest-common-subsequence/)(中等)**

  给定两个字符串 text1 和 text2，返回这两个字符串的最长 公共子序列 的长度。如果不存在 公共子序列 ，返回 0 。

  一个字符串的 子序列 是指这样一个新的字符串：它是由原字符串在不改变字符的相对顺序的情况下删除某些字符（也可以不删除任何字符）后组成的新字符串。

  例如，"ace" 是 "abcde" 的子序列，但 "aec" 不是 "abcde" 的子序列。
  两个字符串的 公共子序列 是这两个字符串所共同拥有的子序列。

  ```
  输入：text1 = "abcde", text2 = "ace" 
  输出：3  
  解释：最长公共子序列是 "ace" ，它的长度为 3 。
  ```

  ```c++
  class Solution {
  public:
      int longestCommonSubsequence(string text1, string text2) {
          int m = text1.length(), n = text2.length();
          vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));
          for (int i = 1; i <= m; ++i) {
              for (int j = 1; j <= n; ++j) {
                  if (text1[i - 1] == text2[j - 1]) {
                      dp[i][j] = dp[i - 1][j - 1] + 1;
                  } else {
                      dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
                  }
              }
          }
          return dp[m][n];
      }
  };
  
  // 执行用时：20 ms, 在所有 C++ 提交中击败了81.57%的用户
  // 内存消耗：12.8 MB, 在所有 C++ 提交中击败了49.65%的用户
  ```

## 7.6 背包问题

+ 0—1背包问题

  ```c++
  int knapsack(vector<int> weights, vector<int> values, int N, int W) {
      // 前i件物品体积不超过j的情况下所能达到的最大价值
      // 在体积约束j（状态变量）的条件下，放第i件物品时（决策变量x）, x=0可以不放，那么就等于放第i-1件
      // 物品时的最大价值， dp[i - 1][j]
      // 如果放物品x = 1, 那么 dp[i - 1][j - w] + v
      // dp[i][j] 为这两条路径的最大值。
      vector<vector<int>> dp(N + 1, vector<int>(W + 1, 0));
      // 这里为什么要是N+1 W+1,原因是，动态规划是一个递推公式，
      // 需要有初始状态，我们这里相当于构建了虚拟的初始状态
      for (int i = 1; i <= N; ++i) {
          int w = weights[i - 1], v = values[i - 1];
          for (int j = 1; j <= W; ++j) {
              // 如果当前状态变量 体积约束大于当前物品，才有可能再放入第i件物品
              if (j >= w) {
                  dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - w] + v);
              } else {
                  dp[i][j] = dp[i - 1][j];
              }
          }
      }
      return dp[N][W];
  }
  ```

  上述代码的时间复杂度和空间复杂度都是 $O(NW)$   可以看到，第i行的更新只依赖于第i-1行的更新，所以可以进行数组的压缩使复杂度将为 $O(W)$ 

  ```c++
  int knapsack(vector<int> weights, vector<int> values, int N, int W) {
  	vector<int> dp(W + 1, 0);
      for (int i = 1; i <= N; ++i) {
          int w = weights[i - 1], v = values[i - 1];
          for (int j = W; j >= w; --j) {
              dp[j] = max(dp[j], dp[j - w] + v);
          }
      }
      return dp[W];
  }
  ```

+ 完全背包问题

  假设我们遍历到物品 i = 2，且其体积为 w = 2，价值为 v = 3；对于背包容量 j = 5，最多只能装下 2 个该物品。那么我们的状态转移方程就变成了 $dp[2] [5] = max(dp[1] [5], dp[1] [3] + 3, dp[1] [1] + 6)$。如果采用这种方法，假设背包容量无穷大而物体的体积无穷小，我们这里的比较次数也会趋近于无穷大，远超 O„NW” 的时间复杂度。
  $$
  \begin{align}
   dp[2][5] &= max(dp[1] [5], dp[1] [3] + 3, dp[1] [1] + 6) \\\\
   &= max(dp[1] [5], max(dp[1][3] + 3, dp[1] [1] + 6)) \\\\
   &= max(dp[1] [5], max(dp[1][3], dp[1][1] + 3) + 3) \\\\
   &= max(dp[1] [5], dp[2][3] + 3)
   \end{align}
  $$
  

  ```c++
  int knapsack(vector<int> weights, vector<int> values, int N, int W) {
  	vector<vector<int>> dp(N + 1, vector<int>(W + 1, 0));
      for (int i = 1; i <= N; ++i) {
          int w = weights[i - 1], v = values[i - 1];
          for (int j = 1; j <= W; ++j) {
              if (j >= w) {
                  dp[i][j] = max(dp[i - 1][j], dp[i][j - w] + v);
              } else {
                  dp[i][j] = dp[i - 1][j];
              }
          }
      }
      return dp[N][W];
  }
  ```

  空间压缩

  ```c++
  int knapsack(vector<int> weights, vector<int> values, int N, int W) {
      vector<int> dp(W + 1, 0);
      for (int i = 1; i <= N; ++i) {
          int w = weights[i], v = values[i];
          for (int j = 1; j <= W; ++j) {
              if (j >= w) {
                  dp[j] = max(dp[j], dp[j - w] + v);
              } 
          }
      }
      return dp[W];
  }
  ```

+ **416 [分割等和子集](https://leetcode-cn.com/problems/partition-equal-subset-sum/) (中等)**

  给你一个 **只包含正整数** 的 **非空** 数组 `nums` 。请你判断是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。

  ```
  输入：nums = [1,5,11,5]
  输出：true
  解释：数组可以分割成 [1, 5, 5] 和 [11] 。
  ```

  ```c++
  class Solution {
  public:
      bool canPartition(vector<int>& nums) {
      	int sum = accumulate(nums.begin(), nums.end(), 0);
          if (sum % 2 == 1)  return false;
          vector<int> dp(sum / 2 + 1, 0);
          for (int i = 1; i <= nums.size(); ++i) {
              int w = nums[i - 1];
              for (int j = sum / 2; j >= w; --j) {
                  dp[j] = max(dp[j], dp[j - w] + w);
                  if (dp[j] == sum / 2) return true;
              }
          }
          return false;
      }
  };
  // 执行用时：84 ms, 在所有 C++ 提交中击败了91.15%的用户
  // 内存消耗：9.5 MB, 在所有 C++ 提交中击败了74.84%的用户
  ```

+ **474 [ 一和零](https://leetcode-cn.com/problems/ones-and-zeroes/) (中等)**

  给你一个二进制字符串数组 strs 和两个整数 m 和 n 。

  请你找出并返回 strs 的最大子集的长度，该子集中 最多 有 m 个 0 和 n 个 1 。

  如果 x 的所有元素也是 y 的元素，集合 x 是集合 y 的 子集 。

  ```
  输入：strs = ["10", "0001", "111001", "1", "0"], m = 5, n = 3
  输出：4
  解释：最多有 5 个 0 和 3 个 1 的最大子集是 {"10","0001","1","0"} ，因此答案是 4 。
  其他满足题意但较小的子集包括 {"0001","1"} 和 {"10","1","0"} 。{"111001"} 不满足题意，因为它含 4 个 1 ，大于 n 的值 3 。
  ```

  ```c++
  class Solution {
  public:
      int findMaxForm(vector<string>& strs, int m, int n) {
          vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));
          for (const string& str: strs) {
              auto [count0, count1] = count(str);
              // 0 - 1背包逆序遍历
              for (int i = m; i >= count0; --i) {
                  for (int j = n; j >= count1; --j) {
                      dp[i][j] = max(dp[i][j], dp[i - count0][j - count1] + 1);
                  }
              }
          }
          return dp[m][n];
      }
  
      pair<int, int> count(const string& str) {
          int count0 = 0, count1 = 0;
          for (int i = 0; i < str.length(); ++i) {
              if (str[i] == '0') ++count0;
              else ++count1;
          }
          return make_pair(count0, count1);
      }
  };
  // 执行用时：200 ms, 在所有 C++ 提交中击败了55.31%的用户
  // 内存消耗：9.4 MB, 在所有 C++ 提交中击败了89.55%的用户
  ```

+ **322 [零钱兑换](https://leetcode-cn.com/problems/coin-change/)(中等)**

  给你一个整数数组 coins ，表示不同面额的硬币；以及一个整数 amount ，表示总金额。

  计算并返回可以凑成总金额所需的 最少的硬币个数 。如果没有任何一种硬币组合能组成总金额，返回 -1 。

  你可以认为每种硬币的数量是无限的。

  ```
  输入：coins = [1, 2, 5], amount = 11
  输出：3 
  解释：11 = 5 + 5 + 1
  ```

  ```c++
  class Solution {
  public:
      int coinChange(vector<int>& coins, int amount) {
          if (coins.empty()) return -1;
          vector<int> dp(amount + 1,  amount + 1);
          dp[0] = 0;
          for (int i = 1; i <= coins.size(); ++i) {
              int v = coins[i - 1];
              for (int j = 1; j <= amount; ++j) {
                  if (j >= v) {
                      dp[j] = min(dp[j], dp[j - v] + 1);
                  }
              }
          }
          return dp[amount] == amount + 1 ? -1: dp[amount];
      }
  };
  // 执行用时：56 ms, 在所有 C++ 提交中击败了90.29%的用户
  // 内存消耗：14 MB, 在所有 C++ 提交中击败了86.61%的用户
  ```

## 7.7 字符串编辑

+ **72 [ 编辑距离](https://leetcode-cn.com/problems/edit-distance/) (困难)**

  给你两个单词 word1 和 word2，请你计算出将 word1 转换成 word2 所使用的最少操作数 。

  你可以对一个单词进行如下三种操作：

  + 插入一个字符
  + 删除一个字符
  + 替换一个字符

  ```
  输入：word1 = "horse", word2 = "ros"
  输出：3
  解释：
  horse -> rorse (将 'h' 替换为 'r')
  rorse -> rose (删除 'r')
  rose -> ros (删除 'e')
  ```

  ```c++
  // 状态 dp[i][j] 第一个字符串到位置i为止，第二个字符串到位置j为止，最多需要编辑几步。
  // 对于这个状态可以由上一阶段的哪些状态转变过来。 三个 dp[i - 1][j - 1]  dp[i][j - 1]  dp[i - 1][j]
  // 接下来在看这三个状态怎么转换到状态 dp[i][j] 
  // 分类讨论 当word1[i] == word2[j] 时 dp[i][j] = dp[i - 1][j - 1], 
  // word1[i] ！= word2[j] 时,dp[i][j] = dp[i - 1][j - 1] + 1 这个时候只需要采取替换操作，因为另外两个操作会使得步数更多
  // 而对于dp[i][j - 1]只需要插入i或者删除j即可，dp[i][j] = dp[i][j - 1] + 1 另一个同理
  // 考虑边界条件，dp[0][0] dp[i][0] dp[j][0] 
  class Solution {
  public:
      int minDistance(string word1, string word2) {
          int m = word1.length(), n = word2.length();
          vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));
          for (int i = 0; i <= m; ++i) {
              for (int j = 0; j <= n; ++j) {
                  if (j == 0) {
                      dp[i][j] = i;
                  } else if (i == 0) {
                      dp[i][j] = j;
                  } else {
                      dp[i][j] = min(
                          dp[i - 1][j - 1] + ((word1[i - 1] == word2[j - 1])? 0:1),
                          min(dp[i][j - 1] + 1, dp[i - 1][j] + 1)
                      );
                  }
              }
          }
          return dp[m][n];
      }
  };
  
  // 执行用时：20 ms, 在所有 C++ 提交中击败了20.18%的用户
  // 内存消耗：8.9 MB, 在所有 C++ 提交中击败了24.01%的用户
  ```


+ **650 [只有两个键的键盘](https://leetcode-cn.com/problems/2-keys-keyboard/)** 

  最初记事本上只有一个字符 'A' 。你每次可以对这个记事本进行两种操作：

  + Copy All（复制全部）：复制这个记事本中的所有字符（不允许仅复制部分字符）。

  + Paste（粘贴）：粘贴 上一次 复制的字符。

    给你一个数字 n ，你需要使用最少的操作次数，在记事本上输出 恰好 n 个 'A' 。返回能够打印出 n 个 'A' 的最少操作次数

  ```
  输入：3
  输出：3
  解释：
  最初, 只有一个字符 'A'。
  第 1 步, 使用 Copy All 操作。
  第 2 步, 使用 Paste 操作来获得 'AA'。
  第 3 步, 使用 Paste 操作来获得 'AAA'。
  
  输入：n = 1
  输出：0
  ```

  ```c++
  class Solution {
  public:
      int minSteps(int n) {
          vector<int> dp(n + 1);
          int h = sqrt(n);
          // 边界值 dp[0] = dp[1] = 0;所以更新从2开始
          for (int i = 2; i <= n; ++i) {
              // 每个最多是i
              dp[i] = i;
              // 除非有因子存在 没有必要从1开始，因为dp[i] = i 已经考虑了dp[9] = dp[1] + dp[9]
              for (int j = 2; j <= h; ++j) {
                  if (i % j == 0) {
                      // 为什么这里只有一个就行了，因为分开的dp还可以继续分 
                      // 例如 dp[12] = dp[2] + dp[6] = dp[2] + dp[2] + dp[3] = dp[3] + dp[4]
                      // 这里又体现了动态规划的本质，递归
                      dp[i] = dp[j] + dp[i / j];
                      
                      break;
                  }
              }
          }
          return dp[n];
      }
  };
  ```

+ **10 [正则表达式匹配](https://leetcode-cn.com/problems/regular-expression-matching/)**

  给你一个字符串 s 和一个字符规律 p，请你来实现一个支持 '.' 和 '*' 的正则表达式匹配。

  + '.' 匹配任意单个字符

  + '*' 匹配零个或多个前面的那一个元素
    所谓匹配，是要涵盖 整个 字符串 s的，而不是部分字符串。

  ```
  输入：s = "aa" p = "a"
  输出：false
  解释："a" 无法匹配 "aa" 整个字符串。
  
  输入：s = "aab" p = "c*a*b"
  输出：true
  解释：因为 '*' 表示零个或多个，这里 'c' 为 0 个, 'a' 被重复一次。因此可以匹配字符串 "aab"。
  ```

  ```c++
  class Solution {
  public:
      bool isMatch(string s, string p) {
          int n = s.size(), m = p.size();
          vector<vector<bool>> dp(n + 1, vector<bool>(m + 1, false));
          // 处理好边界值dp[i][0] = false  dp[0][0] = true dp[0][i] = false
          dp[0][0] = true;
          // 边界值没处理好 a*b*c*d*e*.... 少了下面这个
          for (int i = 1; i <= m; ++i) {
              if (p[i - 1] == '*') {
                  dp[0][i] = dp[0][i - 2];
              }
          }
          // 动态规划
          for (int i = 1; i <= n; ++i) {
              for (int j = 1; j <= m; ++j) {
                  // 先考虑简单的情况
                  // . 的情况
                  if (p[j - 1] == '.') {
                      dp[i][j] = dp[i - 1][j - 1];
                  // 字符的情况
                  } else if (p[j - 1] != '*') {  
                      dp[i][j] = dp[i - 1][j - 1] && s[i - 1] == p[j - 1];
                  // * 的情况 这个情况比较复杂
                  // *1: ...b  ...a* 
                  } else if (p[j - 2] != s[i - 1] && p[j - 2] != '.' ) {
                      dp[i][j] = dp[i][j - 2];
                  // *2: ...a  ...a*/....*
                  // dp[i][j - 1] * 匹配前面一个元素
                  // dp[i - 1][j] * 匹配前面n个元素 这个匹配体现在递推上面
                  // dp[i][j - 2] * 匹配0个前面的元素
                  } else {
                      dp[i][j] = dp[i][j - 1] || dp[i - 1][j] || dp[i][j - 2];
                  }
              }
          }
          return dp[n][m];
      }
  };
  
  // 执行用时：0 ms, 在所有 C++ 提交中击败了100.00%的用户
  // 内存消耗：7 MB, 在所有 C++ 提交中击败了44.91%的用户
  
  // 总结
  // 首先边界的处理一定要考虑仔细
  // 分类讨论从简单到复杂
  // 时刻理解好递推的思想 对于二维状态矩阵，一般不会涉及很深的递推 
  // [i][j], [i - 1][j], [i][j - 1], [i - 1][j - 1] 这是基础需要考虑的 本题的j - 2 是题目特色
  ```
  

## 7.8 股票交易

股票交易类问题通常可以用动态规划来解决。对于稍微复杂一些的股票交易类问题，比如需要冷却时间或者交易费用，则可以用通过动态规划实现的状态机来解决。

如何理解这里的状态机？

+  **121 [买卖股票的最佳时机](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/)**

  给定一个数组 prices ，它的第 i 个元素 prices[i] 表示一支给定股票第 i 天的价格。

  你只能选择 某一天 买入这只股票，并选择在 未来的某一个不同的日子 卖出该股票。设计一个算法来计算你所能获取的最大利润。

  返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回 0 。

  ```
  输入：[7,1,5,3,6,4]
  输出：5
  解释：在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。
       注意利润不能是 7-1 = 6, 因为卖出价格需要大于买入价格；同时，你不能在买入前卖出股票。
  ```

  ```c++
  // 状态 两个状态 第i天之前的最小价格 第i天的将股票卖出所能收获的最大收益
  // 动作 从第i天转到第i+1天
  // 指标函数 
  class Solution {
  public:
      int maxProfit(vector<int>& prices) {
          int sell = 0, buy = INT_MAX;
          for (int i = 0; i < prices.size(); i++) {
              buy = min(buy, prices[i]);
              sell = max(sell, prices[i] - buy);
          }
          return sell;
      }
  };
  ```

  











