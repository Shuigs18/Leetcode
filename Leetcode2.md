# 7. DP

## 7.9 练习

+ **583 [两个字符串的删除操作](https://leetcode-cn.com/problems/delete-operation-for-two-strings/)**

  给定两个单词 *word1* 和 *word2*，找到使得 *word1* 和 *word2* 相同所需的最小步数，每步可以删除任意一个字符串中的一个字符。

  ```
  输入: "sea", "eat"
  输出: 2
  解释: 第一步将"sea"变为"ea"，第二步将"eat"变为"ea"
  ```

  ```c++
  class Solution {
  public:
      int minDistance(string word1, string word2) {
          int n = word1.length(), m = word2.length();
          // dp数组 
          vector<vector<int>> dp(n + 1, vector<int>(m + 1, 0));
  
          // 边界问题 dp[0][i] = i dp[j][0] = j dp[0][0] = 0
  
          for (int i = 1; i <= n; ++i) {
              dp[i][0] = i;
              for (int j = 1; j <= m; ++j) {
                  dp[0][j] = j;
                  dp[i][j] = min(dp[i - 1][j] + 1, 
                                 min(dp[i][j - 1] + 1, 
                                     dp[i - 1][j - 1] + (word1[i - 1] == word2[j - 1])? 0: 2));
              }
          }
          return dp[n][m];
      }
  };
  
  // 执行用时：32 ms, 在所有 C++ 提交中击败了20.44%的用户
  // 内存消耗：11.9 MB, 在所有 C++ 提交中击败了62.43%的用户
  ```

### 进阶难度

+ **646 [最长数对链](https://leetcode-cn.com/problems/maximum-length-of-pair-chain/) (中等)**

  给出 n 个数对。 在每一个数对中，第一个数字总是比第二个数字小。

  现在，我们定义一种跟随关系，当且仅当 b < c 时，数对(c, d) 才可以跟在 (a, b) 后面。我们用这种形式来构造一个数对链。

  给定一个数对集合，找出能够形成的最长数对链的长度。你不需要用到所有的数对，你可以以任何顺序选择其中的一些数对来构造。

  ```
  输入：[[1,2], [2,3], [3,4]]
  输出：2
  解释：最长的数对链是 [1,2] -> [3,4]
  ```

  ```c++
  class Solution {
  public:
      int findLongestChain(vector<vector<int>>& pairs) {
          sort(pairs.begin(), pairs.end(), [](vector<int>& u, vector<int>& v) {
              return u[0] < v[0] || (u[0] == v[0] && u[1] < v[1]);
          });
          int n = pairs.size();
          // 以 第i个链结尾的最长链
          vector<int> dp(n + 1, 1);
          // 处理边界问题 dp[0] = 0;  dp[1] = 1
          dp[0] = 0;
          for (int i = 2; i <= n; ++i) {
              for (int j = i - 1; j >= 1; --j) {
                  if (pairs[i - 1][0] > pairs[j - 1][1]) {
                      dp[i] = max(dp[i], dp[j] + 1);
                  }
              }
          }
          return dp[n];
      }
  };
  // 执行用时：216 ms, 在所有 C++ 提交中击败了35.94%的用户
  // 内存消耗：22.3 MB, 在所有 C++ 提交中击败了48.90%的用户
  ```

+ **376 [摆动序列](https://leetcode-cn.com/problems/wiggle-subsequence/)(中等)**

  如果连续数字之间的差严格地在正数和负数之间交替，则数字序列称为 摆动序列 。第一个差（如果存在的话）可能是正数或负数。仅有一个元素或者含两个不等元素的序列也视作摆动序列。

  + 例如， [1, 7, 4, 9, 2, 5] 是一个 摆动序列 ，因为差值 (6, -3, 5, -7, 3) 是正负交替出现的。

  + 相反，[1, 4, 7, 2, 5] 和 [1, 7, 4, 5, 5] 不是摆动序列，第一个序列是因为它的前两个差值都是正数，第二个序列是因为它的最后一个差值为零。

  子序列 可以通过从原始序列中删除一些（也可以不删除）元素来获得，剩下的元素保持其原始顺序。

  给你一个整数数组 nums ，返回 nums 中作为 摆动序列 的 最长子序列的长度 。

  ```
  输入：nums = [1,7,4,9,2,5]
  输出：6
  解释：整个序列均为摆动序列，各元素之间的差值为 (6, -3, 5, -7, 3) 。
  ```

  ```c++
  class Solution {
  public:
      int wiggleMaxLength(vector<int>& nums) {
          int n = nums.size();
          if (n == 1 || (n == 2 && nums[0] != nums[1])) return n;
  
          // up[i]表示数组nums[0...i]  中最长的以上升趋势为结尾的子序列的长度
          // down[i]表示数组nums[0...i] 中最长的以下降趋势为结尾的子序列的长度
          
          vector<int> up(n), down(n);
          // 处理边界 up[0] = 1, down[0] = 1;
          up[0] = down[0] = 1;
          for (int i = 1; i < n; ++i) {
              if (nums[i] < nums[i - 1]) {
                  up[i] = up[i - 1];
                  down[i] = max(down[i - 1], up[i - 1] + 1);
              } else if (nums[i] > nums[i - 1]){
                  up[i] = max(up[i - 1], down[i - 1] + 1);
                  down[i] = down[i - 1];
              } else {
                  up[i] = up[i - 1];
                  down[i] = down[i - 1];
              }
          }
          return max(down[n - 1], up[n - 1]);
      }
  };
  // 执行用时：0 ms, 在所有 C++ 提交中击败了100%的用户
  // 内存消耗：7.1 MB, 在所有 C++ 提交中击败了14.37%的用户
  
  class Solution {
  public:
      int wiggleMaxLength(vector<int>& nums) {
          int n = nums.size();
          if (n == 1 || (n == 2 && nums[0] != nums[1])) return n;
  
          // up[i]表示数组nums[0...i]  中最长的以上升趋势为结尾的子序列的长度
          // down[i]表示数组nums[0...i] 中最长的以下降趋势为结尾的子序列的长度
          // 优化 内存 只利用两个变量
          // 时间复杂度O(n) 空间复杂度O(1)
          int up = 1, down = 1;
          for (int i = 1; i < n; ++i) {
              if (nums[i] < nums[i - 1]) {
                  down = max(down, up + 1);
              } else if (nums[i] > nums[i - 1]){
                  up = max(up, down + 1);
              } 
          }
          return max(down, up);
      }
  };
  
  // 执行用时：4 ms, 在所有 C++ 提交中击败了46.69%的用户
  // 内存消耗：6.9 MB, 在所有 C++ 提交中击败了68.37%的用户
  ```

+ **494 [目标和](https://leetcode-cn.com/problems/target-sum/)(中等)**

  给你一个整数数组 nums 和一个整数 target 。

  向数组中的每个整数前添加 '+' 或 '-' ，然后串联起所有整数，可以构造一个 表达式 ：

  例如，nums = [2, 1] ，可以在 2 之前添加 '+' ，在 1 之前添加 '-' ，然后串联起来得到表达式 "+2-1" 。
  返回可以通过上述方法构造的、运算结果等于 target 的不同 表达式 的数目。

  ```
  输入：nums = [1,1,1,1,1], target = 3
  输出：5
  解释：一共有 5 种方法让最终目标和为 3 。
  -1 + 1 + 1 + 1 + 1 = 3
  +1 - 1 + 1 + 1 + 1 = 3
  +1 + 1 - 1 + 1 + 1 = 3
  +1 + 1 + 1 - 1 + 1 = 3
  +1 + 1 + 1 + 1 - 1 = 3
  ```

  ```c++
  class Solution {
  public:
      int findTargetSumWays(vector<int>& nums, int target) {
          // 0-1背包问题？？？
          int sum = 0;
          int n = nums.size();
          for (int i = 0; i < n; ++i) {
              sum += nums[i];
          }
  
          if (abs(target) > abs(sum)) return 0;
  
          int range = 2 * sum + 1;
          // dp数组 dp[i][j] 用nums[0...i]构造不同表达式的和为j的数目
          vector<vector<int>> dp(n, vector<int>(range, 0));
          if (nums[0] == 0) {
              dp[0][sum] = 2;
          } else {
              dp[0][sum + nums[0]] = 1;
              dp[0][sum - nums[0]] = 1;
          }
          // 开始更新dp数组
          for (int i = 1; i < n; ++i) {
              for (int j = 0; j < range; ++j) {
                  if (j - nums[i] >= 0 && j + nums[i] < range) { // 加法减法都可以
                      dp[i][j] = dp[i - 1][j - nums[i]] + dp[i - 1][j + nums[i]];
                  } else if (j - nums[i] >= 0) { // 加法满足 减法不满足
                      dp[i][j] = dp[i - 1][j - nums[i]];
                  } else if (j + nums[i] < range) { // 减法满足，加法不满足
                      dp[i][j] = dp[i - 1][j + nums[i]];
                  }
              }
          }
          return dp[n - 1][sum + target];
      }
  };
  
  // 执行用时：68 ms, 在所有 C++ 提交中击败了40.81%的用户
  // 内存消耗：21.5 MB, 在所有 C++ 提交中击败了9.54%的用户
  ```

+ **714 [买卖股票的最佳时机含手续费](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/) (中等)**

  给定一个整数数组 prices，其中第 i 个元素代表了第 i 天的股票价格 ；整数 fee 代表了交易股票的手续费用。

  你可以无限次地完成交易，但是你每笔交易都需要付手续费。如果你已经购买了一个股票，在卖出它之前你就不能再继续购买股票了。

  返回获得利润的最大值。

  注意：这里的一笔交易指买入持有并卖出股票的整个过程，每笔交易你只需要为支付一次手续费。

  ```
  输入：prices = [1, 3, 2, 8, 4, 9], fee = 2
  输出：8
  解释：能够达到的最大利润:  
  在此处买入 prices[0] = 1
  在此处卖出 prices[3] = 8
  在此处买入 prices[4] = 4
  在此处卖出 prices[5] = 9
  总利润: ((8 - 1) - 2) + ((9 - 4) - 2) = 8
  ```

  ```c++
  class Solution {
  public:
      int maxProfit(vector<int>& prices, int fee) {
          int days = prices.size();
          if (days <= 1) return 0;
  
          // 状态矩阵
          // buy第i天进行买入操作所获得的最大利润
          vector<int> buy(days), s1(days), sell(days), s2(days);
          // 边界
          buy[0] = -prices[0];
          s2[0] = -prices[0];
          sell[0] = 0;
          s1[0] = 0;
          for (int i = 1; i < days; ++i) {
              buy[i] = max(sell[i - 1], s1[i - 1]) - prices[i];
              s1[i] = max(sell[i - 1], s1[i - 1]);
              sell[i] = max(buy[i - 1], s2[i - 1]) + prices[i] - fee;
              s2[i] = max(s2[i - 1], buy[i - 1]);
  
          }
          return max(*max_element(sell.begin(), sell.end()), *max_element(s1.begin(), s1.end()));
      }
  };
  
  // 执行用时：104 ms, 在所有 C++ 提交中击败了45.16%的用户
  // 内存消耗：62.9 MB, 在所有 C++ 提交中击败了40.68%的用户
  
  class Solution {
  public:
      int maxProfit(vector<int>& prices, int fee) {
          int days = prices.size();
          if (days <= 1) return 0;
  
          // 状态矩阵
          // buy第i天进行买入操作所获得的最大利润
          // 优化
          int buy, s1, sell, s2, max_sell;
          int cur_buy;
          // 边界
          buy = -prices[0];
          s2 = -prices[0];
          sell = 0;
          s1 = 0;
          max_sell = 0;
          for (int i = 1; i < days; ++i) {
              cur_buy = buy;
              buy = max(sell, s1) - prices[i];
              s1 = max(sell, s1);
              sell = max(cur_buy, s2) + prices[i] - fee;
              s2 = max(s2, cur_buy);
              max_sell = max(max_sell, max(s1, sell));
  
          }
          return max_sell;
      }
  };
  
  // 执行用时：84 ms, 在所有 C++ 提交中击败了70.42%的用户
  // 内存消耗：53.7 MB, 在所有 C++ 提交中击败了87.40%的用户
  ```


# 8. 分治法

## 8.1 

将原问题分为子问题，然后再将子问题进行合并处理。

典型的分治问题：归并排序。“分”即为把大数组平均分成两个小数组，通过递归实现，最终我们会得到多个长度为 1 的子数组；“治”即为把已经排好序的两个小数组合成为一个排好序的大数组，从长度为 1 的子数组开始，最终合成一个大数组。

+ **241 [ 为运算表达式设计优先级](https://leetcode-cn.com/problems/different-ways-to-add-parentheses/) (中等)**

  给定一个含有数字和运算符的字符串，为表达式添加括号，改变其运算优先级以求出不同的结果。你需要给出所有可能的组合的结果。有效的运算符号包含 +, - 以及 * 。

  ```
  输入: "2-1-1"
  输出: [0, 2]
  解释: 
  ((2-1)-1) = 0 
  (2-(1-1)) = 2
  ```

  ```c++
  class Solution {
  public:
      vector<int> diffWaysToCompute(string expression) {
          vector<int> ways;
          for (int i = 0; i < expression.length(); ++i) {
              char ch = expression[i];
              if (ch == '-' || ch == '+' || ch == '*') {
                  vector<int> left = diffWaysToCompute(expression.substr(0, i));
                  vector<int> right = diffWaysToCompute(expression.substr(i + 1));
                  for (const int& l: left) {
                      for (const int& r: right) {
                          switch(ch) {
                              case '-': ways.push_back(l - r); break;
                              case '+': ways.push_back(l + r); break;
                              case '*': ways.push_back(l * r); break;
                          }
                      }
                  }
              }
          }
          if (ways.empty()) ways.push_back(stoi(expression));
          return ways;
      }
  };
  
  // 执行用时：4 ms, 在所有 C++ 提交中击败了70.98%的用户
  // 内存消耗：11.1 MB, 在所有 C++ 提交中击败了64.76%的用户
  ```

## 8.2 练习

+ **932 [漂亮数组](https://leetcode-cn.com/problems/beautiful-array/)(中等)**

  对于某些固定的 N，如果数组 A 是整数 1, 2, ..., N 组成的排列，使得：

  对于每个 i < j，都不存在 k 满足 i < k < j 使得 A[k] * 2 = A[i] + A[j]。

  那么数组 A 是漂亮数组。

  给定 N，返回任意漂亮数组 A（保证存在一个）。

  ```
  输入：4
  输出：[2,1,4,3]
  ```

  ```c++
  class Solution {
  public:
      unordered_map<int, vector<int>> mp;
      
      vector<int> beautifulArray(int n) {
          return f(n);
      }
  
      vector<int> f(int N) {
          vector<int> ans(N);
          if (mp.find(N) != mp.end()) {
              return mp[N];
          }
          int t = 0;
          if (N != 1) {
              for (auto &num: f((N + 1) / 2)) {
                  ans[t++] = 2 * num - 1;
              }
              for (auto &num: f(N / 2)) {
                  ans[t++] = 2 * num;
              }
          } else {
              ans[0] = 1;
          }
          mp[N] = ans;
          return ans;
      }
  };
  
  // 执行用时：4 ms, 在所有 C++ 提交中击败了81.90%的用户
  // 内存消耗：8.2 MB, 在所有 C++ 提交中击败了68.11%的用户
  ```

  **题解：**

  对于N形成的漂亮数组，思考如何构造出来。

  首先将数组分为两个部分 [left, right]，如果left数组是一个漂亮数组，right也是一个漂亮数组，那么我们如何构造才能使[left, right]也是一个漂亮数组？这个时候利用奇偶性， $A[k] * 2 = A[i] + A[j]$ 。如果我们使得 left都是奇数， right都是偶数，这个条件就能成立，比如，如果 $i$ 和 $j$  都落在left中，由于left是漂亮数组，那么显然$A[k] * 2 \neq A[i] + A[j]$ ， 同理 $i$ 和 $j$  都落在right中，也满足。那么如果 $i$ 落在left 中 $j$ 落在right中，由于奇数 + 偶数 = 奇数，所以$A[k] * 2 \neq A[i] + A[j]$ 也一定满足。

  现在再思考，都是奇数的和都是偶数的漂亮数组如何构造？这个时候利用的就是另一性质，如果 $[a_1, a_2, a_3]$ 是一个漂亮数组，那么 $[k * a_1 +b, k * a_2 +b, k * a_3 +b]$ 也是一个漂亮数组。

  例如，对于N = 5，奇数有$(N + 1) / 2$  = 3个(1,3,5)，偶数有 N / 2  = 2个 (2, 4)。利用映射，我们可以将$[1,2,3] \rightarrow [1,3,5]， [1, 2] \rightarrow [2,4]$，这也就意味着，我们可以将N=3的漂亮数组映射到全是奇数的数组，只需要将其中的每个数进行 $2 * k - 1$ 映射，对于N=2也是同理。所以，

  + N = 3 漂亮数组 $[2, 1, 3] \rightarrow [3, 1, 5]$ 

  + N = 2 漂亮数组 $[1, 2] \rightarrow [2, 4]$

  再将其合并得到 N = 5的漂亮数组 $[3,1,5,2,4]$

  这样“分”与“合”都能够得到满足。

  

+ **312 [戳气球](https://leetcode-cn.com/problems/burst-balloons/) (困难)**

  有 n 个气球，编号为0 到 n - 1，每个气球上都标有一个数字，这些数字存在数组 nums 中。

  现在要求你戳破所有的气球。戳破第 i 个气球，你可以获得 nums[i - 1] * nums[i] * nums[i + 1] 枚硬币。 这里的 i - 1 和 i + 1 代表和 i 相邻的两个气球的序号。如果 i - 1或 i + 1 超出了数组的边界，那么就当它是一个数字为 1 的气球。

  求所能获得硬币的最大数量。

  ```
  输入：nums = [3,1,5,8]
  输出：167
  解释：
  nums = [3,1,5,8] --> [3,5,8] --> [3,8] --> [8] --> []
  coins =  3*1*5    +   3*5*8   +  1*3*8  + 1*8*1 = 167
  ```

  ```c++
  class Solution {
  public:
      int maxCoins(vector<int>& nums) {
          int n = nums.size();
          // 动态规划
          vector<vector<int>> dp(n + 2, vector<int>(n + 2, 0));
          // dp[i][j] 表示开区间（i，j）内戳爆气球的最大值。
          vector<int> val(n + 2, 1);
          for (int i = 0; i < n; ++i) {
              val[i + 1] = nums[i];
          }
          for (int i = n - 1; i >= 0; --i) {
              for (int j = i + 2; j <= n + 1; ++j) {
                  for (int k = i + 1; k < j; ++k) {
                      int sum = val[i] * val[k] * val[j];
                      sum += dp[i][k] + dp[k][j];
                      dp[i][j] = max(dp[i][j], sum);
                  }
              }
          }
          return dp[0][n + 1];
      }
  };
  
  // 执行用时：588 ms, 在所有 C++ 提交中击败了66.87%的用户
  // 内存消耗：10 MB, 在所有 C++ 提交中击败了20.44%的用户
  ```

# 9. 巧解数学问题

## 9.1 公倍数与公因数

最小公倍数与最大公因数

```c++
int gcd(int a, int b) {
    return b == 0? a: gcd(b, a % b);
}

int lcm(int a, int b) {
    return a * b / gcd(a, b);
}
```

扩展的欧几里得算法 不太懂

```c++
int xGCD(int a, int b, int &x, int &y) {
    if (!b) {
        x = 1, y = 0;
        return a;
    }
    int x1, y1, gcd = xGCD(a, b, x1, y1);
    x = y1, y = x1 - (a / b) * y1;
    return gcd;
}
```

## 9.2 质数

+ **204 [计数质数](https://leetcode-cn.com/problems/count-primes/) (中等)**

  ```
  输入：n = 10
  输出：4
  解释：小于 10 的质数一共有 4 个, 它们是 2, 3, 5, 7 。
  ```

  ```c++
  class Solution {
  public:
      int countPrimes(int n) {
          if (n <= 2) return 0;
  
          int count = n - 2; // 去掉不是质数的1
          vector<int> prime(n, true);
          for (int i = 2; i < n; ++i) {
              if (prime[i]) {
                  for (int j = 2 * i; j < n; j += i) {
                      if (prime[j]) {
                          prime[j] = false;
                          --count;
                      }
                  }
              }
          }
          return count;
      }
  };
  
  class Solution {
  public:
      int countPrimes(int n) {
          if (n <= 2) return 0;
          vector<int> prime(n, true);
  
          int i = 3, sqrtn = sqrt(n), count = n / 2; // 偶数一定不是质数 (N - 1 + 1) / 2
          while (i <= sqrtn) { // 最大质因数一定小于开方数
              for (int j = i * i; j < n; j += 2 * i) {
                  if (prime[j]) {
                      prime[j] = false;
                      --count;
                  }
              }
              do {
                  i += 2;
              } while(i <= sqrtn && !prime[i]);
          }
          return count;
      }
  };
  ```

## 9.3 数字处理

+ **504 [七进制数](https://leetcode-cn.com/problems/base-7/) (简单)**

  给定一个整数 `num`，将其转化为 **7 进制**，并以字符串形式输出。

  ```
  输入: num = 100
  输出: "202"
  ```

  ```c++
  class Solution {
  public:
      string convertToBase7(int num) {
          if (num == 0) return "0";
          bool is_negative = false;
          if (num < 0) {
              is_negative = true;
              num = -num;
          }
          string ans;
          while (num) {
              int a = num / 7, b = num % 7;
              ans = to_string(b) + ans;
              num = a;
          }
          return is_negative ? "-" + ans: ans;
      }
  };
  // 执行用时：0 ms, 在所有 C++ 提交中击败了100.00%的用户
  // 内存消耗：5.9 MB, 在所有 C++ 提交中击败了26.76%的用户
  ```

+ **172 [阶乘后的零](https://leetcode-cn.com/problems/factorial-trailing-zeroes/) (中等)**

  给定一个整数 `n` ，返回 `n!` 结果中尾随零的数量。

  提示 `n! = n * (n - 1) * (n - 2) * ... * 3 * 2 * 1`

  ```
  输入：n = 3
  输出：0
  解释：3! = 6 ，不含尾随 0
  
  输入：n = 5
  输出：1
  解释：5! = 120 ，有一个尾随 0
  ```

  ```c++
  class Solution {
  public:
      int trailingZeroes(int n) {
          // 只需要看有多少个5 就行
          return n == 0? 0: n / 5 + trailingZeroes(n / 5);
      }
  };
  
  // 执行用时：0 ms, 在所有 C++ 提交中击败了100.00%的用户
  // 内存消耗：5.9 MB, 在所有 C++ 提交中击败了24.02%的用户
  ```

+ **415 [字符串相加](https://leetcode-cn.com/problems/add-strings/)(简单)**

  给定两个字符串形式的非负整数 num1 和num2 ，计算它们的和并同样以字符串形式返回。

  你不能使用任何內建的用于处理大整数的库（比如 BigInteger）， 也不能直接将输入的字符串转换为整数形式。

  ```
  输入：num1 = "11", num2 = "123"
  输出："134"
  ```

  ```c++
  class Solution {
  public:
      string addStrings(string num1, string num2) {
          string output;
          reverse(num1.begin(), num1.end());
          reverse(num2.begin(), num2.end());
          int onelen = num1.length(), twolen = num2.length();
          if (onelen < twolen) {
              swap(num1, num2);
              swap(onelen, twolen);
          }
          int addbit = 0;
          for (int i = 0; i < twolen; ++i) {
              int cursum = num1[i] - '0' + num2[i] - '0' + addbit;
              output += to_string(cursum % 10);
              addbit = cursum / 10;
          }
  
          for (int i = twolen; i < onelen; ++i) {
              int cursum = num1[i] - '0' + addbit;
              output += to_string(cursum % 10);
              addbit = cursum / 10;
          }
          if (addbit) {
              output += '1';
          }
          reverse(output.begin(), output.end());
          return output;
      }
  };
  
  // 执行用时：8 ms, 在所有 C++ 提交中击败了27.17%的用户
  // 内存消耗：6.6 MB, 在所有 C++ 提交中击败了71.41%的用户
  ```

+ **326 [3的幂](https://leetcode-cn.com/problems/power-of-three/) (简单)**

  给定一个整数，写一个函数来判断它是否是 3 的幂次方。如果是，返回 `true` ；否则，返回 `false` 。

  整数 `n` 是 3 的幂次方需满足：存在整数 `x` 使得 `n == 3x`

  ```
  输入：n = 27
  输出：true
  ```

  ```c++
  class Solution {
      vector<int> orgin;
  public:
      Solution(vector<int>& nums): orgin(std::move(nums)) {}
      
      vector<int> reset() {
          return orgin;
      }
      
      vector<int> shuffle() {
          int n = orgin.size();
          vector<int> shuffled(orgin);
          for (int i = n - 1; i >= 0; --i) {
              swap(shuffled[i], shuffled[rand() % (i + 1)]);
          }
          return shuffled;
      }
  };
  
  /**
   * Your Solution object will be instantiated and called as such:
   * Solution* obj = new Solution(nums);
   * vector<int> param_1 = obj->reset();
   * vector<int> param_2 = obj->shuffle();
   */
  ```




# 11. C++ 数据结构

## 11.2 数组

+ **448 [找到所有数组中消失的数字](https://leetcode-cn.com/problems/find-all-numbers-disappeared-in-an-array/) (Easy)**

  给你一个含 n 个整数的数组 nums ，其中 nums[i] 在区间 [1, n] 内。请你找出所有在 [1, n] 范围内但没有出现在 nums 中的数字，并以数组的形式返回结果。

  ```
  输入：nums = [4,3,2,7,8,2,3,1]
  输出：[5,6]
  ```

  ```c++
  class Solution {
  public:
      vector<int> findDisappearedNumbers(vector<int>& nums) {
          vector<int> ans;
          // 原地修改数组
          for (int i = 0; i < nums.size(); ++i) {
              int pos = abs(nums[i]) - 1;
              if (nums[pos] > 0) nums[pos] = -nums[pos];
          }
          for (int i = 0; i < nums.size(); ++i) {
              if (nums[i] > 0) ans.push_back(i + 1);
          }
          return ans;
      }
  };
  // 执行用时：40 ms, 在所有 C++ 提交中击败了89.09%的用户
  // 内存消耗：32.9 MB, 在所有 C++ 提交中击败了58.95%的用户
  ```

+ **48 [旋转图像](https://leetcode-cn.com/problems/rotate-image/) (Medium)**

  给定一个 n × n 的二维矩阵 matrix 表示一个图像。请你将图像顺时针旋转 90 度。

  你必须在 原地 旋转图像，这意味着你需要直接修改输入的二维矩阵。请不要 使用另一个矩阵来旋转图像。

  ```
  输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]
  输出：[[7,4,1],[8,5,2],[9,6,3]]
  ```

  ```c++
  class Solution {
  public:
      void rotate(vector<vector<int>>& matrix) {
      	int n = matrix.size();
          for (int i = 0; i < n / 2; ++i) {
              for (int j = 0; j < (n + 1) / 2; ++j) {
                  int temp = matrix[i][j];
                  matrix[i][j] = matrix[n - j - 1][i];
                  matrix[n - j - 1][i] = matrix[n - i - 1][n - j - 1];
                  matrix[n - i - 1][n - j - 1] = matrix[j][n - i - 1];
                  matrix[j][n - i - 1] = temp;
              }
          }
      }
  };
  // 执行用时：0 ms, 在所有 C++ 提交中击败了100.00%的用户
  // 内存消耗：7 MB, 在所有 C++ 提交中击败了13.06%的用户
  ```

  **总结** 

  对于寻找矩阵中间位置的下标， N为数组size

  +  $N / 2$  **中间位置** 

    如果N为奇数，N / 2 为中间位置的下标

    如果N为偶数，N / 2 为右半部分的第一个位置的下标

  + $(N + 1) / 2$   **右边**

    如果N为奇数，(N + 1) / 2 为右半部分的第一个位置的下标

    如果N为偶数，(N + 1) / 2 为右半部分的第一个位置的下标

  +  $(N - 1) / 2$   **这个可以用作找中间位置**

    如果N为奇数，(N - 1) / 2 为中间位置的下标

    如果N为偶数，(N - 1) / 2 为左半部分的最后一个位置的下标

  

+ **240 [搜索二维矩阵 II](https://leetcode-cn.com/problems/search-a-2d-matrix-ii/) (Medium)**

  编写一个高效的算法来搜索 m x n 矩阵 matrix 中的一个目标值 target 。该矩阵具有以下特性：

  每行的元素从左到右升序排列。
  每列的元素从上到下升序排列。

  ```
  输入：matrix = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], target = 5
  输出：true
  ```

  ```c++
  class Solution {
  public:
      bool searchMatrix(vector<vector<int>>& matrix, int target) {
          int n = matrix.size();
          if (n == 0) return false;
          int m = matrix[0].size();
          int i = 0; j = m - 1;
          while (i < n && j >= 0) {
              if (matrix[i][j] == target) return true;
              else if (matrix[i][j] > target) --j;
              else ++i;
          }
          return false;
      }
  };
  // 执行用时：92 ms, 在所有 C++ 提交中击败了79.77%的用户
  // 内存消耗：14.6 MB, 在所有 C++ 提交中击败了20.34%的用户
  ```
  

## 11.3 栈和队列

+ **155 [最小栈](https://leetcode-cn.com/problems/min-stack/) (Easy)**

  设计一个支持 `push` ，`pop` ，`top` 操作，并能在常数时间内检索到最小元素的栈。

  - `push(x)` —— 将元素 x 推入栈中。
  - `pop()` —— 删除栈顶的元素。
  - `top()` —— 获取栈顶元素。
  - `getMin()` —— 检索栈中的最小元素

  ```
  输入：
  ["MinStack","push","push","push","getMin","pop","top","getMin"]
  [[],[-2],[0],[-3],[],[],[],[]]
  
  输出：
  [null,null,null,null,-3,null,0,-2]
  
  解释：
  MinStack minStack = new MinStack();
  minStack.push(-2);
  minStack.push(0);
  minStack.push(-3);
  minStack.getMin();   --> 返回 -3.
  minStack.pop();
  minStack.top();      --> 返回 0.
  minStack.getMin();   --> 返回 -2.
  ```

  ```c++
  class MinStack {
      // 私有部分
      stack<int> s, min_s;
  public:
      MinStack() { }
      
      void push(int val) {
          s.push(val);
          if (min_s.empty() || val <= min_s.top()) {
              min_s.push(val);
          }
      }
      
      void pop() {
          if (!min_s.empty() && s.top() == min_s.top()) {
              min_s.pop();
          }
          s.pop();
      }
      
      int top() {
          return s.top();
      }
      
      int getMin() {
          return min_s.top();
      }
  };
  
  /**
   * Your MinStack object will be instantiated and called as such:
   * MinStack* obj = new MinStack();
   * obj->push(val);
   * obj->pop();
   * int param_3 = obj->top();
   * int param_4 = obj->getMin();
   */
  
  // 执行用时：24 ms, 在所有 C++ 提交中击败了50.99%的用户
  // 内存消耗：16 MB, 在所有 C++ 提交中击败了42.53%的用户
  ```

+ **20 [有效的括号](https://leetcode-cn.com/problems/valid-parentheses/) (Easy)**

  给定一个只包括 `'('`，`')'`，`'{'`，`'}'`，`'['`，`']'` 的字符串 `s` ，判断字符串是否有效。

  有效字符串需满足：

  1. 左括号必须用相同类型的右括号闭合。
  2. 左括号必须以正确的顺序闭合。

  ```
  输入：s = "()"
  输出：true
  ```

  ```c++
  class Solution {
  public:
      bool isValid(string s) {
          int n = s.size();
          if (n % 2 == 1) return false;
  
          unordered_map<int, int> pairs = {
              {')', '('},
              {']', '['},
              {'}', '{'}
          };
  
          stack<char> stk;
          for (auto &ch: s) {
              if (pairs.count(ch)) {
                  if (stk.empty() || stk.top() != pairs[ch]) {
                      return false;
                  } else {
                      stk.pop();
                  }
              } else {
                  stk.push(ch);
              }
          }
          return stk.empty();
      }
  };
  // 执行用时：0 ms, 在所有 C++ 提交中击败了100.00%的用户
  // 内存消耗：6.2 MB, 在所有 C++ 提交中击败了42.85%的用户
  ```

  

## 11.4 单调栈

+ **739 [每日温度](https://leetcode-cn.com/problems/daily-temperatures/)(Medium)**

  请根据每日 `气温` 列表 `temperatures` ，请计算在每一天需要等几天才会有更高的温度。如果气温在这之后都不会升高，请在该位置用 `0` 来代替.

  ```
  输入: temperatures = [73,74,75,71,69,72,76,73]
  输出: [1,1,4,2,1,1,0,0]
  ```

  ```c++
  class Solution {
  public:
      vector<int> dailyTemperatures(vector<int>& temperatures) {
  		int n = temperatures.size();
  		vector<int> ans(n);
  		stack<int> highDaily;
  		for (int i = 0; i < n; ++i) {
              while (!highDaily.empty()) {
                  int day = highDaily.top();
                  if (tempreatures[i] <= temperatures[day]) break;
                  highDaily.pop();
                  ans[day] = i - day;
              }
              highDaily.push(i);
  		}
          return ans;
      }
  };
  
  // 执行用时：128 ms, 在所有 C++ 提交中击败了85.85%的用户
  // 内存消耗：86.9 MB, 在所有 C++ 提交中击败了17.94%的用户
  ```

## 11.5 优先队列



# 12. 字符串

## 12.1 字符串比较

# 13. 链表

## 13. 1 链表的基本操作

+ **206 [反转链表](https://leetcode-cn.com/problems/reverse-linked-list/) (Easy)**

  给你单链表的头节点 `head` ，请你反转链表，并返回反转后的链表。

  ```
  输入：head = [1,2,3,4,5]
  输出：[5,4,3,2,1]
  ```

  ```c++
  /**
   * Definition for singly-linked list.
   * struct ListNode {
   *     int val;
   *     ListNode *next;
   *     ListNode() : val(0), next(nullptr) {}
   *     ListNode(int x) : val(x), next(nullptr) {}
   *     ListNode(int x, ListNode *next) : val(x), next(next) {}
   * };
   */
  class Solution {
  public:
      ListNode* reverseList(ListNode* head) {
          ListNode *prev = nullptr, *next;
          while (head) {
              next = head->next;
              head->next = prev;
              prev = head;
              head = next;
          }
          return prev;
      }
  };
  // 执行用时：4 ms, 在所有 C++ 提交中击败了88.94%的用户
  // 内存消耗：8 MB, 在所有 C++ 提交中击败了90.32%的用户
  ```

## 13.2 链表的其他操作

+ **160 [相交链表](https://leetcode-cn.com/problems/intersection-of-two-linked-lists/) (Easy)**

  给你两个单链表的头节点 `headA` 和 `headB` ，请你找出并返回两个单链表相交的起始节点。如果两个链表不存在相交节点，返回 `null` 。

  图示两个链表在节点 `c1` 开始相交**：**

  <img src="/Users/shuiguangshan/Pictures/Typora imgs/160_statement.png" alt="img" style="zoom:50%;" />

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
      ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
          ListNode *l1 = headA, *l2 = headB;
          while (l1 != l2) {
              l1 = (l1)? l1->next: headB;
              l2 = (l2)? l2->next: headA;
          }
          return l1;
      }
  };
  ```

+ **234 [回文链表](https://leetcode-cn.com/problems/palindrome-linked-list/) (Easy)**

  给你一个单链表的头节点 `head` ，请你判断该链表是否为回文链表。如果是，返回 `true` ；否则，返回 `false` 。

  ```
  输入：head = [1,2,2,1]
  输出：true
  ```

  ```c++
  /**
   * Definition for singly-linked list.
   * struct ListNode {
   *     int val;
   *     ListNode *next;
   *     ListNode() : val(0), next(nullptr) {}
   *     ListNode(int x) : val(x), next(nullptr) {}
   *     ListNode(int x, ListNode *next) : val(x), next(next) {}
   * };
   */
  class Solution {
  public:
      bool isPalindrome(ListNode* head) {
          if (!head || !head->next) {
              return true;
          }
          ListNode *slow = head, *fast = head;
          while (fast->next && fast->next->next) {
              slow = slow->next;
              fast = fast->next->next;
          }
          slow->next = reverseList(slow->next);
          slow = slow->next;
  
          while(slow) {
              if (head->val != slow->val) return false;
              head = head->next;
              slow = slow->next;
          }
          return true;
      }
  
      ListNode* reverseList(ListNode* head) {
          ListNode *prev = nullptr, *next;
          while (head) {
              next = head->next;
              head->next = prev;
              prev = head;
              head = next;
          }
          return prev;
      }
  };
  // 执行用时：184 ms, 在所有 C++ 提交中击败了70.25%的用户
  // 内存消耗：115.2 MB, 在所有 C++ 提交中击败了57.49%的用户
  ```

  

# 14. 树

## 14.1 树的递归

+ **104 [二叉树的最大深度](https://leetcode-cn.com/problems/maximum-depth-of-binary-tree/) (easy)**

  给定一个二叉树，找出其最大深度。

  二叉树的深度为根节点到最远叶子节点的最长路径上的节点数。

  **说明:** 叶子节点是指没有子节点的节点。

  **示例：**
  给定二叉树 `[3,9,20,null,null,15,7]`

  ```
      3
     / \
    9  20
      /  \
     15   7
  ```

  ```c++
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
      int maxDepth(TreeNode* root) {
  		return root? 1 + max(maxDepth(root->left), maxDepth(root->right)): 0;
      }
  };
  
  // 执行用时：8 ms, 在所有 C++ 提交中击败了70.34%的用户
  // 内存消耗：18.3 MB, 在所有 C++ 提交中击败了94.66%的用户
  ```

+ **110 [平衡二叉树](https://leetcode-cn.com/problems/balanced-binary-tree/)(Easy)**

  给定一个二叉树，判断它是否是高度平衡的二叉树。

  本题中，一棵高度平衡二叉树定义为:

  一个二叉树*每个节点* 的左右两个子树的高度差的绝对值不超过 1 。

  ```
  输入：root = [3,9,20,null,null,15,7]
  输出：true
  ```

  ```c++
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
      int maxDepth(TreeNode* root) {
          return root? 1 + max(maxDepth(root->left), maxDepth(root->right)):0;
      }
      bool isBalanced(TreeNode* root) {
          if (!root) return true;
          int left = maxDepth(root->left), right = maxDepth(root->right);
          if (abs(left - right) >= 2) return false;
          return isBalanced(root->left) && isBalanced(root->right);
      }
  };
  // 执行用时：12 ms, 在所有 C++ 提交中击败了63.17%的用户
  // 内存消耗：20.5 MB, 在所有 C++ 提交中击败了21.45%的用户
  
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
      int helper(TreeNode* root) {
          if (!root) return 0;
          int left = helper(root->left), right = helper(root->right);
          // 这里left==-1和right==-1的作用是如果有子树的高度差大于了1，那么整个部分都是-1
          // 后面的1 + Max（left， right）都不需要计算了
          if (left == -1 || right == -1 || abs(left - right) > 1) {
              return -1;
          }
          return 1 + max(left, right);
      }
  
      bool isBalanced(TreeNode* root) {
          return helper(root) != -1;
      }
  };
  // 执行用时：4 ms, 在所有 C++ 提交中击败了98.68%的用户
  // 内存消耗：20.5 MB, 在所有 C++ 提交中击败了25.54%的用户
  ```

+ **543 [二叉树的直径](https://leetcode-cn.com/problems/diameter-of-binary-tree/)(Easy)**

  给定一棵二叉树，你需要计算它的直径长度。一棵二叉树的直径长度是任意两个结点路径长度中的最大值。这条路径可能穿过也可能不穿过根结点

  ```
            1
           / \
          2   3
         / \     
        4   5   
  返回 3, 它的长度是路径 [4,2,1,3] 或者 [5,2,1,3]
  ```

  ```c++
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
      int maxDepth(TreeNode* root) {
          return root? 1 + max(maxDepth(root->left), maxDepth(root->right)): 0;
      }
      int diameterOfBinaryTree(TreeNode* root) {
          int length;
          if (!root) length = 0;
          else length = maxDepth(root->left) + maxDepth(root->right);
          return root? max(length, max(diameterOfBinaryTree(root->left), diameterOfBinaryTree(root->right))): length;
      }
  };
  // 执行用时：16 ms, 在所有 C++ 提交中击败了21.23%的用户
  // 内存消耗：19.8 MB, 在所有 C++ 提交中击败了43.35%的用户
  
  class Solution {
  public:
      int helper(TreeNode* root, int& diameter) {
          if (!root) return 0;
          int l = helper(root->left, diameter), r = helper(root->right, diameter);
          diameter = max(l + r, diameter);
          return max(l, r) + 1;
      }
      int diameterOfBinaryTree(TreeNode* root) {
          int diameter = 0;
          helper(root, diameter);
          return diameter;
      }
  };
  // 执行用时：4 ms, 在所有 C++ 提交中击败了95.77%的用户
  // 内存消耗：19.7 MB, 在所有 C++ 提交中击败了71.09%的用户
  ```

+ 437 [路径总和 III](https://leetcode-cn.com/problems/path-sum-iii/)(Medium)

  给定一个二叉树的根节点 `root` ，和一个整数 `targetSum` ，求该二叉树里节点值之和等于 `targetSum` 的 **路径** 的数目。

  **路径** 不需要从根节点开始，也不需要在叶子节点结束，但是路径方向必须是向下的（只能从父节点到子节点）。

  <img src="/Users/shuiguangshan/Pictures/Typora imgs/pathsum3-1-tree.jpg" alt="img" style="zoom:50%;" />

  

  ```
  输入：root = [10,5,-3,3,2,null,11,3,-2,null,1], targetSum = 8
  输出：3
  解释：和等于 8 的路径有 3 条，如图所示。
  ```

  ```c++
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
      int pathSum(TreeNode* root, int targetSum) {
          return root? pathSumWithRoot(root, targetSum) +
                       pathSum(root->left, targetSum) + 
                       pathSum(root->right, targetSum): 0;
      }
  
      int pathSumWithRoot(TreeNode* root, int sum) {
          if (!root) return 0;
          int count = root->val == sum ? 1: 0;
          count += pathSumWithRoot(root->left, sum - root->val);
          count += pathSumWithRoot(root->right, sum - root->val);
          return count;
      }
  };
  // 执行用时：36 ms, 在所有 C++ 提交中击败了25.00%的用户
  // 内存消耗：15 MB, 在所有 C++ 提交中击败了98.77%的用户	
  ```

+ **101 [对称二叉树](https://leetcode-cn.com/problems/symmetric-tree/)(Easy)**

  给定一个二叉树，检查它是否是镜像对称的。

  例如，二叉树 `[1,2,2,3,4,4,3]` 是对称的。

  ```
      1
     / \
    2   2
   / \ / \
  3  4 4  3
  ```

  但是下面这个 `[1,2,2,null,3,null,3]` 则不是镜像对称的:

  ```
  但是下面这个 [1,2,2,null,3,null,3] 则不是镜像对称的:
  
      1
     / \
    2   2
     \   \
     3    3
  ```

  ```c++
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
      
      bool helper(TreeNode* left, TreeNode* right) {
          if (left==nullptr && right==nullptr) return true;
          if (left==nullptr || right==nullptr) return false;
          
          bool flag = (left->val == right->val)? true: false;
          flag = flag && helper(left->left, right->right) && helper(left->right, right->left);
          return flag;
      }
  
      bool isSymmetric(TreeNode* root) {
          return root? helper(root->left, root->right): false;
      }
  };
  // 执行用时：8 ms, 在所有 C++ 提交中击败了39.14%的用户
  // 内存消耗：16 MB, 在所有 C++ 提交中击败了51.42%的用户
  
  class Solution {
  public:
      
      bool helper(TreeNode* left, TreeNode* right) {
          if (left==nullptr && right==nullptr) return true;
          if (left==nullptr || right==nullptr) return false;
          if (left->val != right->val) return false;
          return helper(left->left, right->right) && helper(left->right, right->left);
      }
  
      bool isSymmetric(TreeNode* root) {
          return root? helper(root->left, root->right): false;
      }
  };
  
  // 执行用时：4 ms, 在所有 C++ 提交中击败了80.91%的用户
  // 内存消耗：16 MB, 在所有 C++ 提交中击败了56.24%的用户
  
  class Solution {
  public:
      bool isSymmetric(TreeNode* root) {
          return root? isSymmetric(root->left, root->right): true;
      }
      bool isSymmetric(TreeNode* left, TreeNode* right) {
          if (!left && !right) return true;
          if (!left || !right) return false;
          if (left->val != right->val) return false;
          return isSymmetric(left->left, right->right) && 
                 isSymmetric(left->right, right->left);
      }
  };
  // 执行用时：12 ms, 在所有 C++ 提交中击败了8.72%的用户
  // 内存消耗：15.9 MB, 在所有 C++ 提交中击败了80.26%的用户
  ```

+ **1110 [删点成林](https://leetcode-cn.com/problems/delete-nodes-and-return-forest/) (Medium)**

  给出二叉树的根节点 `root`，树上每个节点都有一个不同的值。

  如果节点值在 `to_delete` 中出现，我们就把该节点从树上删去，最后得到一个森林（一些不相交的树构成的集合）。

  返回森林中的每棵树。你可以按任意顺序组织答案。

  <img src="/Users/shuiguangshan/Pictures/Typora imgs/screen-shot-2019-07-01-at-53836-pm.png" alt="img" style="zoom:50%;" />

  ```
  输入：root = [1,2,3,4,5,6,7], to_delete = [3,5]
  输出：[[1,2,null,4],[6],[7]]
  ```

  ```c++
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
  
      vector<TreeNode*> delNodes(TreeNode* root, vector<int>& to_delete) {
  		vector<TreeNode*> forest;
          unordered_set<int> dict(to_delete.begin(), to_delete.end());
          root = helper(root, dict, forest);
          if (root) forest.push_back(root);
          return forest;
      }
  
      TreeNode* helper(TreeNode* root, unordered_set<int>& dict, vector<TreeNode*>& forest) {
  		if (!root) {
              return root;
          }
          // 这个的用意在于后面的如果删掉后应该置为空 否则会影响后面的判断
          root->left = helper(root->left, dict, forest);
          root->right = helper(root->right, dict, forest);
          if (dict.count(root->val)) {
              if (root->left) forest.push_back(root->left);
              if (root->right) foret.push_back(root->right);
              root = NULL;
          }
          return root;
      }
  };
  // 执行用时：20 ms, 在所有 C++ 提交中击败了76.21%的用户
  // 内存消耗：24.9 MB, 在所有 C++ 提交中击败了26.53%的用户
  
  ```

## 14.2 层次遍历---广度优先搜索

+ **637 [二叉树的层平均值](https://leetcode-cn.com/problems/average-of-levels-in-binary-tree/) (Easy) (广度优先搜索)**

  给定一个非空二叉树, 返回一个由每层节点平均值组成的数组。

  ```
  输入：
      3
     / \
    9  20
      /  \
     15   7
  输出：[3, 14.5, 11]
  解释：
  第 0 层的平均值是 3 ,  第1层是 14.5 , 第2层是 11 。因此返回 [3, 14.5, 11] 。
  ```

  ```c++
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
      vector<double> averageOfLevels(TreeNode* root) {
          vector<double> ans;
          if (!root) return ans;
          queue<TreeNode*> q;
          q.push(root);
          while(!q.empty()) {
              int count = q.size();
              double sum = 0;
              for (int i = 0; i < count; ++i) {
                  TreeNode* node = q.front();
                  q.pop();
                  if (node->left) q.push(node->left);
                  if (node->right) q.push(node->right);
                  sum += node->val;
              }
              ans.push_back(sum / count);
          }
          return ans;
      }
  };
  ```

## 14.3 前中后序遍历

```
    1
  /  \
  2   3
/  \   \
4   5   6
```

+ **前序遍历**

  父节点——左节点——右节点   $[1,2,4,5,3,6]$

  ```c++
  void preorder(TreeNode* root) {
      visit(root);
      preorder(root->left);
      preorder(root->right);
  }
  ```

+ **中序遍历**

  左节点——父节点——右节点  $[4, 2, 5, 1,3, 6]$

  ```c++
  void inorder(TreeNode* root) {
      inorder(root->left);
      visit(root);
      inorder(root->right);
  }
  ```

+ **后序遍历**

  左节点——右节点——父节点 $[4, 5, 2, 6, 3, 1]$ 

  ```c++
  void postorder(TreeNode* root) {
      postorder(root->left);
      postorder(root->right);
      vist(root);
  }
  ```

+ **105 [从前序与中序遍历序列构造二叉树](https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/) (Medium)**

  给定一棵树的前序遍历 `preorder` 与中序遍历 `inorder`。请构造二叉树并返回其根节点。

  <img src="/Users/shuiguangshan/Pictures/Typora imgs/tree.jpg" alt="img" style="zoom:50%;" />

  ```
  Input: preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
  Output: [3,9,20,null,null,15,7]
  ```

  ```c++ 
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
      TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
          if (preorder.empty()) return nullptr;
          unordered_map<int, int> hash;
          for (int i = 0; i < inorder.size(); ++i) {
              hash[inorder[i]] = i;
          }
          return buildTreeHelper(preorder, hash, 0, preorder.size() - 1, 0);
      }
  
      TreeNode* buildTreeHelper(vector<int>& preorder, unordered_map<int,int>& hash, 
                                int s0, int e0, int s1) {
          if (s0 > e0) return nullptr;
  
          int mid = preorder[s1], index = hash[mid], leftLen = index - s0 - 1;
  
          TreeNode* node = new TreeNode(mid);
          node->left = buildTreeHelper(preorder, hash, s0, index - 1, s1 + 1);
          node->right = buildTreeHelper(preorder, hash, index + 1, e0, s1 + 1 + leftLen + 1);
          return node;
      }
  };
  // 执行用时：20 ms, 在所有 C++ 提交中击败了61.70%的用户
  // 内存消耗：25.7 MB, 在所有 C++ 提交中击败了40.94%的用户
  ```

+ **144 [二叉树的前序遍历](https://leetcode-cn.com/problems/binary-tree-preorder-traversal/)(Easy)**

  给你二叉树的根节点 `root` ，返回它节点值的 **前序** 遍历。

  ```
  输入：root = [1,null,2,3]
  输出：[1,2,3]
  ```

  ```c++
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
          vector<int> preorderTraversal(TreeNode* root) {
              vector<int> ans;
              if (!root) return ans;
              stack<TreeNode*> s;
              s.push(root);
              while (!s.empty()) {
                  TreeNode* node = s.top();
                  s.pop();
                  ans.push_back(node->val);
                  if (node->right) {
                      s.push(node->right);
                  }
                  if (node->left) {
                      s.push(node->left);
                  }
              }
              return ans;
          }
      };
  // 执行用时：4 ms, 在所有 C++ 提交中击败了43.66%的用户
  // 内存消耗：8.3 MB, 在所有 C++ 提交中击败了23.77%的用户
  ```

## 14.5 二叉查找树(BST)

```c++
template <class T>
class BST {
    struct Node {
        T data;
        Node* left;
        Node* right;
    };
    
    Node* root;
    
    Node* makeEmpty(Node* t) {
        if (!t) return NULL;
        makeEmpty(t->left);
        makeEmpty(t->right);
        delete t;
        return NULL;
    }
    
    Node* insert(Node* t, T x) {
        if (t == NULL) {
            t->data = x;
            t->left = NULL;
            t->right = NULL;
        }
        if (x < t->data) {
            t->left = insert(t->left, x);
        } else {
            t->right = insert(t->right, x);
        }
        return t;
    }
    
    Node* find(Node* t, T x) {
        // 返回x的节点，如果树中没有的话，返回空节点
        if (!t) return NULL;
        if (x < t->data) return find(t->left, x);
        if (x > t->data) return find(t->right, x);
        return t;
    }
    
    Node* findMin(Node* t) {
        if (!t) return NULL;
        if (t->right) return findMin(t->right);
        return t;
        // if (t == NULL || t->right == NULL) return t;
        // return findMin(t->right);
    }
    
    Node* findMax(Node* t) {
        if (!t) return NULL;
        if (t->left) return findMax(t->left);
        return t;
    }
    
    Node* remove(Node* t, T x) {
        // 删除节点方法
        // 1. 找到该节点
        // 2. 如果左子树和右子树都非空，找到右边最小的节点，用该节顶替掉需要删除的节点
        // 2.2 递归调用删除右边的最小节点
        // 3. 如果左子树非空，则用左子树直接顶上来，右子树同理
        Node* temp;
        if (!t) return NULL;
        else if (x < t->data) return remove(t->left, x);
        else if (x > t->data) return remove(t->right, x);
        else if (t->left && t -> right) {
            temp = findMin(t->right);
            t->data = temp->data;
            delete temp;
            t->right = remove(t->right, t->data);
        } else {
            temp = t;
            if (t->left == NULL) t = t->right;
            else if (t->right == NULL) t = t->left;
            delete temp;
        }
        return t;
    }

public:
    // 对于这个类 在整个程序中是可以访问的。
    // 构造函数和析构函数
    BST(): root(NULL) {}
    ~BST() {
        root = makeEmpty(root);
    }
    void insert(T x) {
        insert(root, x);
    }
    void remove(T x) {
        remove(root, x);
    }
}
```

+ **[恢复二叉搜索树](https://leetcode-cn.com/problems/recover-binary-search-tree/) (Medium)**

给你二叉搜索树的根节点 root ，该树中的两个节点被错误地交换。请在不改变其结构的情况下，恢复这棵树。

进阶：使用 O(n) 空间复杂度的解法很容易实现。你能想出一个只使用常数空间的解决方案吗？

<img src="/Users/shuiguangshan/Pictures/Typora imgs/recover1.jpg" alt="img" style="zoom:50%;" />



```
输入：root = [1,3,null,null,2]
输出：[3,1,null,null,2]
解释：3 不能是 1 左孩子，因为 3 > 1 。交换 1 和 3 使二叉搜索树有效。
```

```c++
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
    void recoverTree(TreeNode* root) {

    }
};
```

