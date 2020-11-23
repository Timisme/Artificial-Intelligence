## AI mid-term 

Agent's Performance Measure: 

:::info
Rationality:
1. criterion of success 
2. prior knowledge 
3. the actions the agent can perform 
4. Agents percepts sequence to date
:::

Simple Reflex agents 

This agent function only succeeds when the environment is fully observable

Model Based Reflex agents 

:::info
Model-based reflex agent
這種agent對世界環境有其猜測，認為環境符合某個model （也包括agent參與後如何改變世界的考量），maintain internal states，所以他可以在partial observable的條件下運作。

model通常是依賴percerpt history。
:::

Goal Based agent 

:::info
這種agent把未來考慮進去，所謂未來就是action是否朝向maximize goal的方向前進？
:::

utility based agent

:::info
goal可能可以通過不同的途徑抵達，所以我們需要找到一個能讓agent最滿足的途徑，透過utility function來評估。
:::

    A rational utility-based agent chooses the action that maximizes the expected utility of the action outcomes
    
--- 

# CH3 solving Problem by searching 

## Uninformed Search 

### BFS (Breadth First Search)

:::info
廣度優先搜尋
即廣義的Level-Order Traversal
會用到Recursive Algorithm 
:::

    計算量太大：假設每個node會產生b個子node，且solution在depth(level)d，則會產生b+b^2+b^3+...+b^d個nodes
   
    It is extremely inefficient and is not ideal for large data structures. Once the algorithm finds a path that reaches the end node it is guaranteed that this is the shortest possible path. This is because of the queue structure that the algorithm uses.
    
### Uinformed cost search

:::info
沒什麼好講?
:::
### Depth first search (may not find the best solution)
:::info
graph traversal algorithm 
:::

    優點: 比BFS較快
    缺點：找的到goal，但不是optimal solution(可能Path cost很大)
## Informed Search (Heuristics)

### Greedy best first search 

:::info
利用evaluation function f(n) for each node，已經知道某些資訊(estimate of cost from n to the goal)，每個node expand的原則就是從max or min f(n)的node expand下去
:::

    greedy best 不一定會找到optimal解，因為它就是只看目前狀態與goal之間的cost做決定，沒有考慮之前走過的路徑。
1. 可能會stuck in loops
2. time 有減少很多
3. space：keeps all nodes in memory 
4. optimal：No

### A* Search 

:::info
精隨在於 選擇一個node並繼續expand的條件由原本僅考慮現在state至goal之間的cost，改為考慮cost to reach the node g(n) & the cost from the node to the goal h(n)，所以f(n) = g(n) + h(n)

等同於uniform-cost-search，但f(n)改為以上式子
:::


----

# CH4 Beyond Classical Searching 

### Local Search Algo

    概念：Path does not matter. Objective function does

:::info
1. very little memory usage(不用紀錄所有走過的path) - 省記憶體，速度快
2. 僅從current node 往neighbors下去尋找(條件就是max objective func)
3. find reasonal solutions in continuoius spaces
:::

    heuristic：the path cost to the goal is known，and we are making decisions based on these given information to reach our desired outcome. 

### Hill Climbing Search 

    Greedy local search : 只找那最棒的鄰居，常常效果不錯

1. 問題: 卡local min & max 
2. 很快就找到local max or min，等在短時間內就可以找到local已經OK

:::info
1. Stochastic hill -> 隨便選neighbor(比較不會卡Local)
2. first choice -> 找到第一個比current state好的就選(省時間)
3. random restart hill climbing -> 不同的initial state下去找
:::

## 模擬退火法

:::info
隨便找鄰居，如果比較好直接取代，如果沒有的話會以固定機率取代。
如何決定機率? --> 溫度
:::

----

# ch13 量化不確定性

    agent 面對的問題中，很多時候是不能100%確定狀態的，或者當前狀態不能直接imply結果，其中的不確定性需要用機率評估

    Decision Theory = probability theroy + utility theory
:::info
每個agent做出action的條件還是最佳化utility func，只不過utility以機率的方式描述(因為環境不確定性等)
當進行決策時，agent需要針對當前的條件，推斷事件發生的條件機率。
:::


    P(X|y) = P(X, y) // P(y), P(X|y) = a(P(X,y))
    就算不知道P(y)，也可以算出P(X|y)，只要我們有data!
    
## Bayes' Rule

    P(x|y) = P(y|x) * P(x) // P(y)
    Sequence labeling -> P(labels | words) = P(words | labels) * P(labels) // P(words)
    
:::info
問題: 給定一組字下，想知道全部可能的Labels下，發生機率最高的那組label違和。此問題可以利用 p(x|y) ~ p(y|x)*p(x)

即 P(labels | words) ~ P(words | labels) * P(labels)
:::

==如何從多個evidence，推測單一原因?==

#### 這時候就會需要假設變數間條件獨立：

    P(label|word_a, word_b) = a(P(word_a, word_b|label)) * P(label)，其中P(word_a, word_b | label) 的求法，需假設word_a 與 word_b 之間獨立，即可將式改為 P(word_a|label) * P(word_b|label)
    
:::info
補充: 假設word_a 與 Label獨立，則P(label | word_a, word_b)可簡化為 P(label | word_b) 
:::

故在假設變數間獨立的情況下 (Naive Bayes model)

![](https://i.imgur.com/e78zpyQ.png)

---

# CH14 Probability Reasoning (機率推論)

#### Bayesian Network

:::info
價值: 用graph的方式表達變數間條件獨立的關係
:::

![](https://i.imgur.com/BHnwh5K.png =50%x)

用法:
1. 表達joint proba - 用於建構與理解conditional proba 計算
![](https://i.imgur.com/4hvMKA5.png =50%x)

2. ==解決chain rule大量計算的問題== 
![](https://i.imgur.com/kFSA57i.png)

3. 每個變數與其非child node，在其parent node條件下，皆獨立。
Network 建構的方法有很多種 取決於方法間機率取得準確性與容易度決定


### Markov blanket 

![](https://i.imgur.com/tEmnOxk.png =50%x)

:::info
給定某node的parents, childs, 及 childs parents, 該node與其他node有cond independent的關係。
:::

### markov assumption -> 某一變數僅其相鄰的變數有關係，其他在相鄰變數給定情況下，無關。

## ==若變數為連續，如何使用bayesian nets?==

    一個變數發生的條件機率，可以利用gaussian distribution來Model。以參數化母數u, sigma^2
    
:::info
如果node & parent node皆為連續，可用高斯分布Model他們的機率關係

如果node是binary & parent node為連續，可將parent變數分布取積分變成累積pdf，則取一個threshold就可以決定current node的0,1(Probit distribution)，也可以用logit distribution。

:::

### exact inference 

![](https://i.imgur.com/noLT0Tv.png =50%x)


## The variable elimination algorithm

    解決機率重複計算的問題

## Approximate inference

    若遇到multiply connected network 用近似的方式推估某query variable的條件機率。

:::info
當條件機率計算非常複雜時，可以用randomized sampling algo，如Monte Carlo algo的方式得到query variable的條件機率
:::

## Monte carlo 的方式

:::info
Direct sampling

根據貝氏網路順序抽樣
為何要有貝氏網路? -> 方便計算joint probability!!!
==但是當variable很多，joint probability很難算==
但如果sample很多次(n足夠大)，則抽樣Joint prob的機率會逼近真實P(x1,x2,...,xn)的聯合機率(estimate is consistent)。
:::

![](https://i.imgur.com/nW9xFjE.png)

### Rejectio sampling ??

    缺點，當evidence出現的機率很低的時候，sampling會需要花很多時間才能完成
    
### Likelihood weighting 

    not every event is equal，每個事件組合在抽樣出現的機率會經過weight修正

## ==Markov Chain Monte Carlo algo==

    不是一直重新抽樣，而是從之前抽樣的結果稍微改變而產生新的sample。

:::info
重點概念就是，從前一次的抽樣結果，稍微改一下內容，產生新的抽樣結果
:::

## Gibbs sampling in Bayesian networks

:::info
核心概念: 一次變一個variable，從上次結果變成下一個sample，是與direct sample不同的地方。==可節省inference推論所需時間==

針對X,Y,Z, E這四個變數，假設query為P(x|y,z)，則抽樣一次後將y,z的觀察值固定，僅改變non evidence的變數抽樣結果(改變X,E)，分別討論:

1. X 抽樣: 給定X的markov blanket(child, parents, childs parents)的值下，去抽樣X變數。
2. 
::: 

    MCMC的重點在於，sampling的依據是根據目前的狀態下，再去做sampling，得到query prob，而非重頭到尾重新sample，對於下棋來說，要到某個固定盤面的機率很低，若用direct很難遇到這樣得盤面。
    
-----

# CH15 根據時間的機率推論。

    當機率隨時間會產生改變(變數隨時間變化)，就會牽涉時間上的機率推論。

:::info
定義問題: 

Transition model ： 給定之前的state，下一個state的機率。
問題來了，P(Xt|X0.....Xt-1)中，條件的部分需要假設，才方便計算。
這個假設就會利用markov assumption -> 假設時間t的state只跟t-1的state有關。

first order Markov process, transition model -> P(Xt|Xt-1)
並假設time series是平穩 (transition model 和絕對t無關)。
:::

==transition & sensor model 假設:==

![](https://i.imgur.com/IzQJ8y6.png =80%x)


![](https://i.imgur.com/gAQ3KuA.png =80%x)

## ==Filtering==

:::info
(state esimation) filtering的定義: 給定t0 至t的觀察(evidence)，預估時間t的狀態。P(Xt | e1:t) -> ==一直從頭疊代==

prediction: P(Xt+k | e1:t)
:::

![](https://i.imgur.com/KYdCS06.png =50%x)
![](https://i.imgur.com/ctKSI05.png)
![](https://i.imgur.com/HJVwPCM.png)


## Smoothing: P(Xk | e1:t) for (1<k<t)

:::info
同樣利用recursive方式用backward(從t到k+1)，從後面推回來。
:::

![](https://i.imgur.com/pqPSgy5.png)

稱為forward, backward algo

## Most likely explanation : given sequence e1:t, 計算X1:t的機率。

:::info
利用smoothing的方式，給定e1...et，可以推測第k天的state(1<k<t)，則依照這個概念可以計算P(X1:t | e1:t)
:::

![](https://i.imgur.com/1gsj4ji.png)

## 以上的計算法為 Viterbi algorithm 

:::info
Viterbi 三大概念
1. recursive
2. transition model 
3. sensor model
:::

----

# Hidden Markov Models 

    1. hidden : 只能看到evidence，只能推估states。
    2. markov : state 只和附近states有關
    3. state is single discrete random variable
    
 
:::info 
agent除了可用filter預估目前位置外，也能利用smoothing方法加上viterbi 去找出最佳可能路徑，估算目前距離。則應過越多序列後，儘管error rate高，準確度也能到不錯的水準
:::
 
![](https://i.imgur.com/XQHnY5V.png =50%x)

## Kalman Filters 演算法

==假設states為連續變數，則HMM不適用，需使用kalman filters來判斷hidden state==

:::info
假設transition model及sensor model 皆follow linear Gaussian distribution 

下一個連續變數state會是目前連續變數state的高斯函數值+高斯noise
:::

    為何選擇高斯函數? -> closed under standard 貝氏網路
    高斯函數中變數做計算完後仍然為高斯分布


![](https://i.imgur.com/g2kOW0j.png)

    kalman filtering 的限制: 變數的模型僅限制於線性關係，若實際狀況為極度非線性，則不適用。
    
    
## 若有多個objects

---

# Training Tips

:::info
影響訓練accu表現的因素
1. loss function (cross entropy vs MSE)
2. mini - batch (訓練更快，更容易找到最佳解)
3. activation function (ReLU解決sigmoid遇到的gradient vanishing問題)
4. adaptive learning rate (adagrad針對梯度下降的幅度做修正)
5. Momentum (避免卡在local min or saddle pnt, or plateau)
6. model structure 
:::

:::info
影響測試accu因素，及解決
1. Early Stopping
2. Regularization (weight decay)
3. Dropout
4. simpler network structure
5. data augmentation 
:::

# CNN 

### Why CNN? 

    沒有cnn的話 FC的參數太多。
    將圖片特徵擷取，且不針對位置而分類。
    sumsampling (resize)，不用整張圖，後依然可以分類。

:::info
Convolution layer類似dropout功能，僅針對特定pixel進行neuron計算
，減少參數使用，也減少overfitting。
:::

# RNN 

:::info
RNN 在訓練時因為neuron之間將input連乘的關係 gradient經常有巨大變動，training效果不好。但是LSTM的output gate機制能解決gradient vanishing的問題。
:::