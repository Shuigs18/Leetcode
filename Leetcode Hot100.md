# Leetcode Hot100

## **2 [两数相加](https://leetcode-cn.com/problems/add-two-numbers/) (中等)**

给你两个 非空 的链表，表示两个非负的整数。它们每位数字都是按照 逆序 的方式存储的，并且每个节点只能存储 一位 数字。

请你将两个数相加，并以相同形式返回一个表示和的链表。

你可以假设除了数字 0 之外，这两个数都不会以 0 开头。

```
输入：l1 = [0], l2 = [0]
输出：[0]
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
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {

        if (!l1) return l2;
        if (!l2) return l1;
        ListNode *head = new ListNode(), *cur = head;
        int addBit = 0;
        while (l1 || l2) {
            cur->val = ((l1?l1->val: 0) + (l2?l2->val: 0)+ addBit) % 10;
            addBit = ((l1?l1->val: 0) + (l2?l2->val: 0)+ addBit) / 10;
            l1 = l1? l1->next: nullptr;
            l2 = l2? l2->next: nullptr;
            if (l1 || l2) {
                cur->next = new ListNode();
                cur = cur->next;
            }
        }
        if (addBit) {
            cur->next = new ListNode(1);
        }
        return head;
    }
};
// 执行用时：28 ms, 在所有 C++ 提交中击败了67.90%的用户
// 内存消耗：69.3 MB, 在所有 C++ 提交中击败了88.18%的用户
```

## **3 [无重复字符的最长子串](https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/)(Meidum)**

给定一个字符串 `s` ，请你找出其中不含有重复字符的 **最长子串** 的长度。

```
输入: s = "abcabcbb"
输出: 3 
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
```

```c++
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        unordered_set<char> occ;
        int r = -1;
        int n = s.size();

        int maxLength = 0;

        for (int i = 0; i < n; ++i) {
            if (i != 0) {
                occ.erase(s[i - 1]);
            }
            while (r + 1 < n && !occ.count(s[r + 1])) {
                occ.insert(s[r + 1]);
                ++r;
            }
            maxLength = max(maxLength, r - i + 1);
        }
        return maxLength;
    }
};
// 执行用时：28 ms, 在所有 C++ 提交中击败了44.34%的用户
// 内存消耗：10.7 MB, 在所有 C++ 提交中击败了23.04%的用户

class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        unordered_set<char> occ;
        int left = 0;
        int n = s.size();

        int maxLength = 0;

        for (int i = 0; i < n; ++i) {
            while (occ.count(s[i])) {
                occ.erase(s[left]);
                ++left;
            }
            occ.insert(s[i]);
            maxLength = max(maxLength, i - left + 1);
        }
        return maxLength;
    }
};
// 执行用时：24 ms, 在所有 C++ 提交中击败了52.38%的用户
// 内存消耗：10.6 MB, 在所有 C++ 提交中击败了37.04%的用户
```

## **5 [最长回文子串](https://leetcode-cn.com/problems/longest-palindromic-substring/) (Medium)** 

给你一个字符串 `s`，找到 `s` 中最长的回文子串。

```
输入：s = "babad"
输出："bab"
解释："aba" 同样是符合题意的答案。
```

```c++
class Solution {
public:
    string longestPalindrome(string s) {
        // dp动态规划解决 dp[i][j] 表示i...j是否为回文字串
        int n = s.size();
        if (n <= 1) return s;

        vector<vector<int>> dp(n, vector<int>(n));
        int maxLen = 1;
        int begin = 0;

        for (int i = 0; i < n; ++i) dp[i][i] = true;

        for (int L = 2; L <= n; ++L) {
            for (int i = 0; i < n; ++i) {
                // j - i + 1 = L
                int j = L + i - 1;
                if (j >= n) break;

                if (s[i] != s[j]) {
                    dp[i][j] = false;
                } else {
                    if (j - i < 3) {
                        dp[i][j] = true;
                    } else {
                        dp[i][j] = dp[i + 1][j - 1];
                    }
                }
                
                if (dp[i][j] && maxLen < j - i + 1) {
                    maxLen = j - i + 1;
                    begin = i;
                }
            }
        }
        return s.substr(begin, maxLen);
    }
};
// 执行用时：608 ms, 在所有 C++ 提交中击败了22.92%的用户
// 内存消耗：378.1 MB, 在所有 C++ 提交中击败了7.54%的用户
```

## **11 [盛最多水的容器](https://leetcode-cn.com/problems/container-with-most-water/) (Meidum)**

给你 n 个非负整数 a1，a2，...，an，每个数代表坐标中的一个点 (i, ai) 。在坐标内画 n 条垂直线，垂直线 i 的两个端点分别为 (i, ai) 和 (i, 0) 。找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。

说明：你不能倾斜容器

<img src="/Users/shuiguangshan/Pictures/Typora imgs/question_11.jpg" alt="img" style="zoom:50%;" />

```
输入：[1,8,6,2,5,4,8,3,7]
输出：49
解释：图中垂直线代表输入数组 [1,8,6,2,5,4,8,3,7]。在此情况下，容器能够容纳水（表示为蓝色部分）的最大值为 49。
```

```c++
// 暴力解法 超出时间限制
class Solution {
public:
    int maxArea(vector<int>& height) {
        int n = height.size();

        int maxWater = 0;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i != j) {
                    maxWater = max(maxWater, min(height[i], height[j]) * abs(i - j));
                }
            }
        }
        return maxWater;

    }
};

// 双指针 
// 证明见 https://leetcode-cn.com/problems/container-with-most-water/solution/sheng-zui-duo-shui-de-rong-qi-by-leetcode-solution/
class Solution {
public:
    int maxArea(vector<int>& height) {
        int n = height.size();
        int l = 0, r = n - 1;
        int ans = 0;

        while (l < r) {
            int area = min(height[l], height[r]) * abs(l - r);
            if (height[l] <= height[r]) ++l;
            else --r;
            ans = max(ans, area);
        }
        return ans;
    }
};

// 执行用时：88 ms, 在所有 C++ 提交中击败了19.44%的用户
// 内存消耗：57.5 MB, 在所有 C++ 提交中击败了90.81%的用户
```

## **15 [三数之和](https://leetcode-cn.com/problems/3sum/) (Medium)**

给你一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？请你找出所有和为 0 且不重复的三元组。

注意：答案中不可以包含重复的三元组

```
输入：nums = [-1,0,1,2,-1,-4]
输出：[[-1,-1,2],[-1,0,1]]
```

```c++
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        int n = nums.size();
        vector<vector<int>> ans;
        if (n < 3) return ans;

        sort(nums.begin(), nums.end());

        for (int first = 0; first < n; ++first) {
            if (first != 0 && nums[first] == nums[first - 1]) continue;
            int third = n - 1;
            for (int second = first + 1; second < n; ++second) {
                if (second != first + 1 && nums[second] == nums[second - 1]) continue;
                while (second < third && nums[first] + nums[second] + nums[third] > 0) {
                    --third;
                }
                if (second == third) break;
                if (nums[first] + nums[second] + nums[third] == 0) {
                    ans.push_back({nums[first], nums[second], nums[third]});
                }
            }
        }
        return ans;
    }
};

// 执行用时：92 ms, 在所有 C++ 提交中击败了37.73%的用户
// 内存消耗：19.4 MB, 在所有 C++ 提交中击败了92.19%的用户
```

## **17 [电话号码的字母组合](https://leetcode-cn.com/problems/letter-combinations-of-a-phone-number/) (Meidum)** 

给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合。答案可以按 任意顺序 返回。

给出数字到字母的映射如下（与电话按键相同）。注意 1 不对应任何字母。

```
输入：digits = "23"
输出：["ad","ae","af","bd","be","bf","cd","ce","cf"]
```

```c++
class Solution {
public:
    vector<string> letterCombinations(string digits) {
        vector<string> combinations;
        if (digits.empty()) return combinations;

        string combination;

        unordered_map<char, string> phoneMap = {
            {'2', "abc"},
            {'3', "def"},
            {'4', "ghi"},
            {'5', "jkl"},
            {'6', "mno"},
            {'7', "pqrs"},
            {'8', "tuv"},
            {'9', "wxyz"}
        };
        backtracking(combinations, digits, phoneMap, 0, combination);
        return combinations;
    }

    void backtracking(vector<string> &combinations, string digits, 
                      unordered_map<char, string> &phoneMap, int index, string combination) {
        if (index == digits.size()) {
            combinations.push_back(combination);
            return ;
        }
        char digit = digits[index];
        const string &letters = phoneMap.at(digit);
        for (auto &ch: letters) {
            combination.push_back(ch);
            backtracking(combinations, digits, phoneMap, index + 1, combination);
            combination.pop_back();
        }

    }
};

// 执行用时：4 ms, 在所有 C++ 提交中击败了33.64%的用户
// 内存消耗：6.3 MB, 在所有 C++ 提交中击败了79.31%的用户
```

## 19 [删除链表的倒数第 N 个结点](https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list/) (Medium)

给你一个链表，删除链表的倒数第 `n` 个结点，并且返回链表的头结点

```
输入：head = [1,2,3,4,5], n = 2
输出：[1,2,3,5]
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
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        // 快慢指针
        ListNode *fast = head, *slow, *dummy = new ListNode(0, head);
        slow = dummy;

        // 找到待删除节点的前驱节点
        for (int i = 0; i < n; ++i) {
            fast = fast->next;
        }

        while (fast) {
            fast = fast->next;
            slow = slow->next;
        }

        // 删除节点 此时slow为待删除节点的前驱节点
        ListNode *temp = slow->next;
        slow->next = temp->next;
        // 释放内存 如果要求的话
        delete temp;

        return dummy->next;

    }
};

// 执行用时：4 ms, 在所有 C++ 提交中击败了76.03%的用户
// 内存消耗：10.4 MB, 在所有 C++ 提交中击败了73.44%的用户
```

## **22 [括号生成](https://leetcode-cn.com/problems/generate-parentheses/) (Medium)**

数字 `n` 代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且 **有效的** 括号组合。

有效括号组合需满足：左括号必须以正确的顺序闭合

```
输入：n = 3
输出：["((()))","(()())","(())()","()(())","()()()"]
```

```c++
class Solution {
    // 智能指针 自动释放所指向对象
    shared_ptr<vector<string>> cache[100] = {nullptr};
public:
    shared_ptr<vector<string>> generate(int n) {
        if (cache[n] != nullptr) return cache[n];
        if (n == 0) {
            cache[0] = shared_ptr<vector<string>>(new vector<string>{""});
            return cache[0];
        }
        shared_ptr<vector<string>> result(new vector<string>);
        for (int i = 0; i != n; ++i) {
            shared_ptr<vector<string>> lefts = generate(i);
            shared_ptr<vector<string>> rights = generate(n - i - 1);
            for (auto &left: *lefts) {
                for (auto &right: *rights) {
                    result->push_back("(" + left + ")" + right);
                }
            }
        }
        cache[n] = result;
        return cache[n];
    }
    vector<string> generateParenthesis(int n) {
        // 合理序列 c = (a)b 不断枚举a b
        return *generate(n);
    }
};

// 执行用时：0 ms, 在所有 C++ 提交中击败了100.00%的用户
// 内存消耗：7.1 MB, 在所有 C++ 提交中击败了97.44%的用户
```

## **23 [合并K个升序链表](https://leetcode-cn.com/problems/merge-k-sorted-lists/) (Medium)**

给你一个链表数组，每个链表都已经按升序排列。

请你将所有链表合并到一个升序链表中，返回合并后的链表。

```
输入：lists = [[1,4,5],[1,3,4],[2,6]]
输出：[1,1,2,3,4,4,5,6]
解释：链表数组如下：
[
  1->4->5,
  1->3->4,
  2->6
]
将它们合并到一个有序链表中得到。
1->1->2->3->4->4->5->6
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

    ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
        if (!list1 || !list2) return list1 ? list1: list2;

        ListNode head, *tail = &head, *list1Ptr = list1, *list2Ptr = list2;

        while (list1Ptr && list2Ptr) {
            if (list1Ptr->val <= list2Ptr->val) {
                tail->next = list1Ptr;
                list1Ptr = list1Ptr->next;
            } else {
                tail->next = list2Ptr;
                list2Ptr = list2Ptr->next;
            }
            tail = tail->next;
        }

        tail->next = list1Ptr? list1Ptr : list2Ptr;

        return head.next;
    }


    ListNode* merge(vector<ListNode*> &lists, int l, int r) {
        if (l == r) return lists[l];
        if (l > r) return nullptr;

        int mid = (l + r) >> 1;
        return mergeTwoLists(merge(lists, l, mid), merge(lists, mid + 1, r));
    }

    ListNode* mergeKLists(vector<ListNode*>& lists) {
        return merge(lists, 0, lists.size() - 1);
    }
};

// 执行用时：20 ms, 在所有 C++ 提交中击败了83.14%的用户
// 内存消耗：12.7 MB, 在所有 C++ 提交中击败了86.78%的用户
```

## **31 [下一个排列](https://leetcode-cn.com/problems/next-permutation/) (Medium)**

实现获取 下一个排列 的函数，算法需要将给定数字序列重新排列成字典序中下一个更大的排列（即，组合出下一个更大的整数）。

如果不存在下一个更大的排列，则将数字重新排列成最小的排列（即升序排列）。

必须 原地 修改，只允许使用额外常数空间

```
输入：nums = [1,2,3]
输出：[1,3,2]
```

```c++
class Solution {
public:
    void nextPermutation(vector<int>& nums) {
        // 从后向前找，找到第一个“较小”数 较小要满足尽可能的靠右
        // 所以找到第一个nums[i] < nums[i + 1] 的数
        int i = nums.size() - 2;
        while (i >= 0 && nums[i] >= nums[i + 1]) {
            --i;
        }
        if (i >= 0) {
            // 从后向前找，找到第一个大于nums[i]的值 这样才能使得交换后
            // 整体的数能够变大
            int j = nums.size() - 1;
            while (j >= 0 && nums[j] <= nums[i]) {
                --j;
            }
            swap(nums[i], nums[j]);
        }
        reverse(nums.begin() + i + 1, nums.end());
    }
};
// 执行用时：4 ms, 在所有 C++ 提交中击败了72.34%的用户
// 内存消耗：11.7 MB, 在所有 C++ 提交中击败了68.38%的用户
```

## **32 [最长有效括号](https://leetcode-cn.com/problems/longest-valid-parentheses/) (Medium)**

给你一个只包含 `'('` 和 `')'` 的字符串，找出最长有效（格式正确且连续）括号子串的长度

```
输入：s = "(()"
输出：2
解释：最长有效括号子串是 "()"
```

```c++
class Solution {
public:
    int longestValidParentheses(string s) {
        // 动态规划解决
        //
        int n = s.size();
        vector<int> dp(n + 1);
        // 边界dp[0] = dp[1] = 0;

        for (int i = 2; i <= n; ++i) {
            if (s[i - 1] == '(') continue;

            if (s[i - 2] == '(') {
                dp[i] = dp[i - 2] + 2;
            } else {
                if (i - 1 - dp[i - 1] - 1 >= 0 && s[i - 1 - dp[i - 1] - 1] == '(') {
                    dp[i] = dp[i - 1] + 2 + dp[i - dp[i - 1] - 2];
                }
            }
        }
        return *max_element(dp.begin(), dp.end());
    }
};
// 执行用时：0 ms, 在所有 C++ 提交中击败了100.00%的用户
// 内存消耗：7 MB, 在所有 C++ 提交中击败了79.17%的用户
```

## **33 [搜索旋转排序数组](https://leetcode-cn.com/problems/search-in-rotated-sorted-array/) (Medium)**

整数数组 `nums` 按升序排列，数组中的值 **互不相同** 

在传递给函数之前，nums 在预先未知的某个下标 k（0 <= k < nums.length）上进行了 旋转，使数组变为 [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]（下标 从 0 开始 计数）。例如， [0,1,2,4,5,6,7] 在下标 3 处经旋转后可能变为 [4,5,6,7,0,1,2] .

给你 **旋转后** 的数组 `nums` 和一个整数 `target` ，如果 `nums` 中存在这个目标值 `target` ，则返回它的下标，否则返回 `-1` 。

```
输入：nums = [4,5,6,7,0,1,2], target = 0
输出：4
```

```c++
class Solution {
public:


    int search(vector<int>& nums, int target) {
        int n = nums.size();
        if (n == 1) return (nums[0] == target)? 0: -1;

        int l = 0, r = nums.size() - 1;

        while (l <= r) {
            int mid = (l + r) >> 1;

            if (target == nums[mid]) return mid;

            // 这里是大于等于而不是大于 因为mid是可能和l相等的，
            if (nums[l] <= nums[mid]) { // 说明左区间是排好序的
                if (target >= nums[l] && target < nums[mid]) {
                    r = mid - 1;
                } else {
                    l = mid + 1;
                }
            }  else { // 右区间是排好序的
                if (target > nums[mid] && target <= nums[r]) {
                    l = mid + 1;
                } else {
                    r = mid - 1;
                }
            }
        }
        return -1;
    }
};
// 执行用时：4 ms, 在所有 C++ 提交中击败了69.37%的用户
// 内存消耗：10.8 MB, 在所有 C++ 提交中击败了43.97%的用户
```

## **39 [组合总和](https://leetcode-cn.com/problems/combination-sum/)(Medium)**

给定一个无重复元素的正整数数组 candidates 和一个正整数 target ，找出 candidates 中所有可以使数字和为目标数 target 的唯一组合。

candidates 中的数字可以无限制重复被选取。如果至少一个所选数字数量不同，则两种组合是唯一的。 

对于给定的输入，保证和为 target 的唯一组合数少于 150 个。

```
输入: candidates = [2,3,6,7], target = 7
输出: [[7],[2,2,3]]
```

```c++
class Solution {
public:
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        vector<int> combination;
        vector<vector<int>> combinations;
        backtracking(combinations, combination, target, candidates, 0);
        return combinations;
    }

    void backtracking(vector<vector<int>> &combinations, vector<int> &combination,
                      int target, vector<int> &candidates, int index) {
        if (index == candidates.size()) return;

        if (target == 0) {
            combinations.push_back(combination);
            return ;
        }

        // 跳过
        backtracking(combinations, combination, target, candidates, index + 1);
        
        // 不跳过，选择当前的值
        if (target - candidates[index] >= 0) {
            combination.push_back(candidates[index]);
            // 可以重复选所以这里的index没有加1
            backtracking(combinations, combination, 
                         target - candidates[index], candidates, index);
            combination.pop_back();
        }

    }
};

// 执行用时：4 ms, 在所有 C++ 提交中击败了93.17%的用户
// 内存消耗：16.5 MB, 在所有 C++ 提交中击败了23.30%的用户
```

## **42 [接雨水](https://leetcode-cn.com/problems/trapping-rain-water/) (Hard)**

给定 `n` 个非负整数表示每个宽度为 `1` 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水

<img src="/Users/shuiguangshan/Pictures/Typora imgs/rainwatertrap.png" alt="img"  />

```
输入：height = [0,1,0,2,1,0,1,3,2,1,2,1]
输出：6
解释：上面是由数组 [0,1,0,2,1,0,1,3,2,1,2,1] 表示的高度图，在这种情况下，可以接 6 个单位的雨水（蓝色部分表示雨水）。 
```

```c++
// 方法一 动态规划法
class Solution {
public:
    int trap(vector<int>& height) {
        // 动态规划 构建两个数组，左边最大的数组和右边最大的数组
        int n = height.size();
        vector<int> leftMax(n);
        vector<int> rightMax(n);

        // 边界
        leftMax[0] = height[0];
        for (int i = 1; i < n; ++i) {
            leftMax[i] = max(leftMax[i - 1], height[i]);
        }
        rightMax[n - 1] = height[n - 1];
        for (int i = n - 2; i >= 0; --i) {
            rightMax[i] = max(rightMax[i + 1], height[i]);
        }

        //某处i能够接到的雨水量=左边最高和右边最高的最小值-height[i]
        int water = 0;
        for (int i = 0; i < n; ++i) {
            water += min(leftMax[i], rightMax[i]) - height[i];
        }

        return water;
    }
};
// 执行用时：8 ms, 在所有 C++ 提交中击败了78.20%的用户
// 内存消耗：15.6 MB, 在所有 C++ 提交中击败了32.71%的用户

// 还有很多方法  之后再补充看
```

## **49 [字母异位词分组](https://leetcode-cn.com/problems/group-anagrams/) (Medium)**

给你一个字符串数组，请你将 字母异位词 组合在一起。可以按任意顺序返回结果列表。

字母异位词 是由重新排列源单词的字母得到的一个新单词，所有源单词中的字母都恰好只用一次。

```
输入: strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
输出: [["bat"],["nat","tan"],["ate","eat","tea"]]
```

```c++
// 暴力解题 超出时间限制
class Solution {
public:
    vector<vector<string>> groupAnagrams(vector<string>& strs) {
        vector<vector<string>> ans;
        int n = strs.size();
        if (n == 0) return ans;

        vector<bool> visit(n);

        for (int i = 0; i < n; ++i) {
            if (visit[i]) continue;
            vector<string> temp;
            temp.push_back(strs[i]);
            visit[i] = true;
            for (int j = i + 1; j < n; ++j) {
                if (check(strs[i], strs[j])) {
                    temp.push_back(strs[j]);
                    visit[j] = true;
                }
            }
            ans.push_back(temp);
        }
        return ans;
    }

    bool check(string s1, string s2) {
        unordered_map<char, int> count;
        for (auto &ch: s1) {
            ++count[ch];
        }
        for (auto &ch: s2) {
            // 说明出现了新的单词
            if(!count[ch]) return false;
            --count[ch];
        }
        // 如果单词没用完也不行
        for (auto &ch: s1) {
            if (count[ch]) return false;
        }
        return true;
    }

};

// 方法一 排序
class Solution {
public:
    vector<vector<string>> groupAnagrams(vector<string>& strs) {
        // 互为字母异位词的字符串排过序以后一定是相同字符串

        unordered_map<string, vector<string>> mp;

        for (int i = 0; i < strs.size(); ++i) {
            string s = strs[i];
            sort(s.begin(), s.end());
            mp[s].push_back(strs[i]);
        }

        vector<vector<string>> ans;

        for (auto it = mp.begin(); it != mp.end(); ++it) {
            ans.push_back(it->second);
        }
        
        return ans;
    }
};
// 执行用时：28 ms, 在所有 C++ 提交中击败了81.97%的用户
// 内存消耗：19 MB, 在所有 C++ 提交中击败了72.95%的用户

// 其余方法见lc题解
```

## **55 [跳跃游戏](https://leetcode-cn.com/problems/jump-game/) (Medium)**

给定一个非负整数数组 `nums` ，你最初位于数组的 **第一个下标** 。

数组中的每个元素代表你在该位置可以跳跃的最大长度。

判断你是否能够到达最后一个下标

```
输入：nums = [2,3,1,1,4]
输出：true
解释：可以先跳 1 步，从下标 0 到达下标 1, 然后再从下标 1 跳 3 步到达最后一个下标。
```

```c++
// 动态规划超出时间限制。。。。
class Solution {
public:
    bool canJump(vector<int>& nums) {
        // 动态规划
        int n = nums.size();

        // dp[i] 能否到达位置i
        vector<bool> dp(n, false);
        // 边界 
        dp[0] = true;

        for (int i = 1; i < n; ++i) {
            // j跳跃到i
            for (int j = 0; j < i; ++j) {
                bool jump = (abs(i - j) <= nums[j]);
                dp[i] = dp[i] || (dp[j] && jump);
            }
        }
        return dp[n - 1];
    }
};

class Solution {
public:
    bool canJump(vector<int>& nums) {
        // 。。。。 只能向左跳
        // 这样的话题目就简单了，只需要向左跳的最大位置，遍历一遍就结束了
        int n = nums.size();
        int maxDistance = 0 + nums[0];
        for (int i = 1; i < n - 1; ++i) {
            if (maxDistance >= i) {
                maxDistance = max(maxDistance, i + nums[i]);
            }
        }
        return (maxDistance >= n - 1)? true: false;
    }
};

// 执行用时：60 ms, 在所有 C++ 提交中击败了33.12%的用户
// 内存消耗：47 MB, 在所有 C++ 提交中击败了93.65%的用户

class Solution {
public:
    bool canJump(vector<int>& nums) {
        // 。。。。 只能向左跳
        // 这样的话题目就简单了，只需要向左跳的最大位置，遍历一遍就结束了

        int n = nums.size();

        int maxDistance = 0;
        for (int i = 0; i < n - 1; ++i) {
            if (maxDistance >= i) {
                maxDistance = max(maxDistance, i + nums[i]);
            }
        }

        return (maxDistance >= n - 1)? true: false;
    }
};

// 执行用时：60 ms, 在所有 C++ 提交中击败了33.12%的用户
// 内存消耗：47.1 MB, 在所有 C++ 提交中击败了81.27%的用户
```

## **56 [合并区间](https://leetcode-cn.com/problems/merge-intervals/) (Medium)**

以数组 intervals 表示若干个区间的集合，其中单个区间为 intervals[i] = [starti, endi] 。请你合并所有重叠的区间，并返回一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间.

```
输入：intervals = [[1,3],[2,6],[8,10],[15,18]]
输出：[[1,6],[8,10],[15,18]]
解释：区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].
```

```c++
class Solution {
public:

    vector<vector<int>> merge(vector<vector<int>>& intervals) {
        int n = intervals.size();
        sort(intervals.begin(), intervals.end(), [](vector<int> &u, vector<int> &v) {
            return (u[0] < v[0]);
        });
        vector<vector<int>> ans;

        ans.push_back(intervals[0]);

        for (int i = 1; i < n; ++i) {
            vector<int> top = ans.back();
            if (top[1] >= intervals[i][0]) {
                ans.pop_back();
                ans.push_back(vector<int>{top[0], max(top[1], intervals[i][1])});
            } else {
                ans.push_back(intervals[i]);
            }
        }
        return ans;
    }
};
// 执行用时：20 ms, 在所有 C++ 提交中击败了58.14%的用户
// 内存消耗：15 MB, 在所有 C++ 提交中击败了12.51%的用户
```

## **62 [不同路径](https://leetcode-cn.com/problems/unique-paths/)(Medium)**

一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为 “Start” ）。

机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish” ）。

问总共有多少条不同的路径？

![img](/Users/shuiguangshan/Documents/GitHub/Leetcode/pics/robot_maze.png)

```
输入：m = 3, n = 7
输出：28
```

```c++
class Solution {
public:
    int uniquePaths(int m, int n) {
        // 动态规划
        vector<vector<int>> dp(m, vector<int>(n, 1));
        // 边界 dp[0][i] = 1, dp[i][0] = 1;

        for (int i = 1; i < m; ++i) {
            for (int j = 1; j < n; ++j) {
                dp[i][j] = dp[i][j - 1] + dp[i - 1][j];
            }
        }
        return dp[m - 1][n - 1];
    }
};
// 执行用时：0 ms, 在所有 C++ 提交中击败了100.00%的用户
// 内存消耗：6.4 MB, 在所有 C++ 提交中击败了23.65%的用户

// 内存优化
class Solution {
public:
    int uniquePaths(int m, int n) {
        // 动态规划
        // 内存优化
        vector<int> dp(n, 1);
        for (int i = 1; i < m; ++i) {
            for (int j = 1; j < n; ++j) {
                dp[j] = dp[j] + dp[j - 1];
            }
        }
        return dp[n - 1];
    }
};

// 执行用时：0 ms, 在所有 C++ 提交中击败了100.00%的用户
// 内存消耗：6 MB, 在所有 C++ 提交中击败了71.69%的用户
```

## **78 [子集](https://leetcode-cn.com/problems/subsets/) (Medium)**

给你一个整数数组 `nums` ，数组中的元素 **互不相同** 。返回该数组所有可能的子集（幂集）。

解集 **不能** 包含重复的子集。你可以按 **任意顺序** 返回解集

```
输入：nums = [1,2,3]
输出：[[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
```

```c++
class Solution {
public:
    vector<vector<int>> subsets(vector<int>& nums) {
        // 回溯法 每个可以选或者不选
        // 并且每个不能重复选
        vector<int> subset;
        vector<vector<int>> ans;
        backtracking(ans, subset, nums, 0);
        return ans;
    }

    void backtracking(vector<vector<int>> &ans, vector<int> &subset, vector<int> &nums, int index) {
        if (index == nums.size()) {
            ans.push_back(subset);
            return;
        }
        // 跳过
        backtracking(ans, subset, nums, index + 1);

        // 选择当前数
        subset.push_back(nums[index]);
        backtracking(ans, subset, nums, index + 1);
        subset.pop_back();
    }
};

// 执行用时：4 ms, 在所有 C++ 提交中击败了49.88%的用户
// 内存消耗：10.5 MB, 在所有 C++ 提交中击败了10.76%的用户

class Solution {
public:
    vector<int> t;
    vector<vector<int>> ans;

    void dfs(int cur, vector<int>& nums) {
        if (cur == nums.size()) {
            ans.push_back(t);
            return;
        }
        t.push_back(nums[cur]);
        dfs(cur + 1, nums);
        t.pop_back();
        dfs(cur + 1, nums);
    }

    vector<vector<int>> subsets(vector<int>& nums) {
        dfs(0, nums);
        return ans;
    }
};

// 执行用时：0 ms, 在所有 C++ 提交中击败了100.00%的用户
// 内存消耗：7.1 MB, 在所有 C++ 提交中击败了34.12%的用户
```

## 84 [柱状图中最大的矩形](https://leetcode-cn.com/problems/largest-rectangle-in-histogram/) (Hard)

给定 *n* 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。

求在该柱状图中，能够勾勒出来的矩形的最大面积。

<img src="/Users/shuiguangshan/Documents/GitHub/Leetcode/pics/histogram.png" alt="img" style="zoom:50%;" />

```
输入：heights = [2,1,5,6,2,3]
输出：10
解释：最大的矩形为图中红色区域，面积为 10
```

```c++
class Solution {
public:
    int largestRectangleArea(vector<int>& heights) {
        // 核心枚举可能的高度，找到每个高度对应的宽即可解决问题
        // 优化点在于如何提高找到宽度的效率
        // 这里使用单调栈
        int n = heights.size();
        // 两个数组
        vector<int> left(n), right(n);

        // 构建单调栈
        stack<int> mono_stack;

        // 生成left数组
        for (int i = 0; i < n; ++i) {
            while (!mono_stack.empty() && heights[i] <= heights[mono_stack.top()]) {
                mono_stack.pop();
            }
            left[i] = (mono_stack.empty()? -1: mono_stack.top());
            mono_stack.push(i);
        }

        // 生成right数组
        // 别忘记重新使 mono_stack 为空
        mono_stack = stack<int>();
        for (int i = n - 1; i >= 0; --i) {
            while (!mono_stack.empty() && heights[i] <= heights[mono_stack.top()]) {
                mono_stack.pop();
            }
            right[i] = (mono_stack.empty()? n: mono_stack.top());
            mono_stack.push(i);
        }

        int ans = 0;
        for (int i = 0; i < n; ++i) {
            ans = max(ans, heights[i] * (right[i] - left[i] - 1));
        }
        return ans;

    }
};
// 执行用时：128 ms, 在所有 C++ 提交中击败了22.26%的用户
// 内存消耗：64.9 MB, 在所有 C++ 提交中击败了16.17%的用户
```

