---
layout: post
title: 【Datawhalw】【CS224W】图神经网络（五）
date: 2023-02-13
tags: [Datawhale,CS224W,图神经网络]
comments: true
toc: true
author: 乌墨_rfr
---
## 一、Deepwalk

### 1.1 预备知识

机器学习，深度学习基础，语言模型，word2vec



### 1.2 Deepwalk介绍

DeepWalk的思想类似word2vec，使用**图中节点与节点的共现关系**来学习节点的向量表示。那么关键的问题就是如何来描述节点与节点的共现关系，DeepWalk给出的方法是使用随机游走(RandomWalk)的方式在图中进行节点采样。

$Deepwalk$将$graph$的每一个节点编码为一个$D$维向量(Embedding)(无监督学习)

Embedding中隐式包含了$Graph$中的社群，连接，结构信息，可用于后续节点分类等下游任务（监督学习）

![image-20230222142856728](https://raw.githubusercontent.com/wumorfr/photo/master/image-20230222142856728.png)

Deepwalk通过套用随机游走(Random walk generation)将图像转化为D维向量

![image-20230222153354939](https://raw.githubusercontent.com/wumorfr/photo/master/image-20230222153354939.png)

### 1.3 Embedding 

我们需要将图转化为计算机了解的向量，所以我们需要使用embedding

![image-20230222143330826](https://raw.githubusercontent.com/wumorfr/photo/master/image-20230222143330826.png)



### 1.4 word2Vec 词向量，词嵌入

word2vec通过语料库中的句子序列来描述词与词的共现关系，进而学习到词语的向量表示。

主要有两种方法

CBOW：用边缘词去预测中心词

Skip-gram：用输入的中心词去预测周围词

![image-20230222143852663](https://raw.githubusercontent.com/wumorfr/photo/master/image-20230222143852663.png)



### 1.5 random Walk随机游走

RandomWalk是一种**可重复访问已访问节点的深度优先遍历**算法。给定当前访问起始节点，从其邻居中随机采样节点作为下一个访问节点，重复此过程，直到访问序列长度满足预设条件。

获取足够数量的节点访问序列后，使用skip-gram model 进行向量学习。

![img](https://pic1.zhimg.com/80/v2-fdcb0babe4a168df85243b548fd86d30_720w.webp)

具体见[[Datawhale]\[CS224W]图机器学习(四)](https://blog.csdn.net/weixin_45856170/article/details/129070015)

### 1.6 DeepWalk 核心代码

DeepWalk算法主要包括两个步骤，第一步为随机游走采样节点序列，第二步为使用skip-gram modelword2vec学习表达向量。

①构建同构网络，从网络中的每个节点开始分别进行Random Walk 采样，得到局部相关联的训练数据； ②对采样数据进行SkipGram训练，将离散的网络节点表示成向量化，最大化节点共现，使用Hierarchical Softmax来做超大规模分类的分类器

#### Random Walk

我们可以通过并行的方式加速路径采样，在采用多进程进行加速时，相比于开一个进程池让每次外层循环启动一个进程，我们采用固定为每个进程分配指定数量的`num_walks`的方式，这样可以最大限度减少进程频繁创建与销毁的时间开销。

`deepwalk_walk`方法对应上一节伪代码中第6行，`_simulate_walks`对应伪代码中第3行开始的外层循环。最后的`Parallel`为多进程并行时的任务分配操作。

```python
def deepwalk_walk(self, walk_length, start_node):

    walk = [start_node]

    while len(walk) < walk_length:
        cur = walk[-1]
        cur_nbrs = list(self.G.neighbors(cur))
        if len(cur_nbrs) > 0:
            walk.append(random.choice(cur_nbrs))
        else:
            break
    return walk

def _simulate_walks(self, nodes, num_walks, walk_length,):
    walks = []
    for _ in range(num_walks):
        random.shuffle(nodes)
        for v in nodes:           
            walks.append(self.deepwalk_walk(alk_length=walk_length, start_node=v))
    return walks

results = Parallel(n_jobs=workers, verbose=verbose, )(
    delayed(self._simulate_walks)(nodes, num, walk_length) for num in
    partition_num(num_walks, workers))

walks = list(itertools.chain(*results))
```

#### Word2vec

这里就偷个懒直接用`gensim`里的Word2Vec了。

```python
from gensim.models import Word2Vec
w2v_model = Word2Vec(walks,sg=1,hs=1)
```

#### DeepWalk应用

```python
G = nx.read_edgelist('../data/wiki/Wiki_edgelist.txt',create_using=nx.DiGraph(),nodetype=None,data=[('weight',int)])

model = DeepWalk(G,walk_length=10,num_walks=80,workers=1)
model.train(window_size=5,iter=3)
embeddings = model.get_embeddings()

evaluate_embeddings(embeddings)
plot_embeddings(embeddings)
```

### 1.7 DeepWalk优缺点

1. 优点
   - 首个将深度学习和自然语言处理的思想用于图机器学习
   - 在系数标注节点分类场景下，嵌入性能卓越
2. 缺点
   - 均匀随机游走，没有偏向的游走方向（Node2Vec）
   - 需要大量随机游走序列训练
   - 基于随机游走，管中窥豹。距离较远的两个节点无法相互影响。看不到全图信息。（图神经网络）
   - 无监督，仅编码图的连接信息，没有利用节点的属性特征
   - 没有真正用到神经网络和深度学习





## 二、Node2Vec

### 2.1 图嵌入

将数据转化为D维连续稠密的向量，包含了原来的节点的多种信息

![image-20230222153010863](https://raw.githubusercontent.com/wumorfr/photo/master/image-20230222153010863.png)

![image-20230222153329769](https://raw.githubusercontent.com/wumorfr/photo/master/image-20230222153329769.png)

### 2.2 Node2Vec

#### 优化目标

设 $f(u)$是将顶点 $u$ 映射为embedding向量的映射函数,对于图中每个顶点$u$，定义 $N_s(u)$ 为通过采样策略 $S$采样出的顶点 $u$的近邻顶点集合。

node2vec优化的目标是给定每个顶点条件下，令其近邻顶点（**如何定义近邻顶点很重要**）出现的概率最大。

$max_f\sum_{u\in V}log Pr(N_s(U)|f(u))$

为了将上述最优化问题可解，文章提出两个假设：

- 条件独立性假设

假设给定源顶点下，其近邻顶点出现的概率与近邻集合中其余顶点无关。 $Pr(N_s(u)|f(u))=\prod_{n_i|f(u)}Pr(n_i|f(u))$

- 特征空间对称性假设

这里是说一个顶点作为源顶点和作为近邻顶点的时候**共享同一套embedding向量**。(对比LINE中的2阶相似度，一个顶点作为源点和近邻点的时候是拥有不同的embedding向量的) 在这个假设下，上述条件概率公式可表示为 $Pr(n_i|f(u))={expf(n_i)\cdot  f(u)}\over{\sum_{v\in V}expf(v)\cdot f(u)}$

根据以上两个假设条件，最终的目标函数表示为![image-20230222163604126](https://raw.githubusercontent.com/wumorfr/photo/master/image-20230222163604126.png)

由于归一化因子 ![image-20230222163615301](https://raw.githubusercontent.com/wumorfr/photo/master/image-20230222163615301.png)的计算代价高，所以采用负采样技术优化。

#### 顶点序列采样策略

node2vec依然采用随机游走的方式获取顶点的近邻序列，不同的是node2vec采用的是一种有偏的随机游走。

给定当前顶点 v ，访问下一个顶点 x 的概率为

![img](https://pic2.zhimg.com/80/v2-84cc0b66ec34043f82649f0d799997e1_720w.webp)

$\pi_{vx}$ 是顶点 v 和顶点 x 之间的未归一化转移概率， Z 是归一化常数。



node2vec引入两个超参数 p 和 q 来控制随机游走的策略，假设当前随机游走经过边 (t,v) 到达顶点 �v设 ![image-20230222163711950](https://raw.githubusercontent.com/wumorfr/photo/master/image-20230222163711950.png)，$\omega_{vx}$ 是顶点 v 和 x 之间的边权，

![img](https://pic3.zhimg.com/80/v2-0d170e5c120681823ed6880411a0478e_720w.webp)

$d_{tx}$为顶点 t 和顶点 x 之间的最短路径距离。

下面讨论超参数 p 和 q 对游走策略的影响

- Return parameter,p

参数p控制重复访问刚刚访问过的顶点的概率。 注意到p仅作用于 $d_{tx}$=0 的情况，而$d_{tx}$=0 表示顶点 x 就是访问当前顶点 v 之前刚刚访问过的顶点。 那么若 p 较高，则访问刚刚访问过的顶点的概率会变低，反之变高。

- In-out papameter,q

q 控制着游走是向外还是向内，若 q>1 ，随机游走倾向于访问和 t 接近的顶点(偏向BFS)。若 q<1 ，倾向于访问远离 t 的顶点(偏向DFS)。

下面的图描述的是当从 t 访问到 v 时，决定下一个访问顶点时每个顶点对应的 $\alpha$ 。

设定p,q进行有偏随机游走

![image-20230222153738097](https://raw.githubusercontent.com/wumorfr/photo/master/image-20230222153738097.png)

q节点小时更愿意进行BFS,p节点小时更愿意进行DFS

![image-20230222153919325](https://raw.githubusercontent.com/wumorfr/photo/master/image-20230222153919325.png)

### 2.3 理论上的实现

![image-20230222154012450](https://raw.githubusercontent.com/wumorfr/photo/master/image-20230222154012450.png)

### 2.4 Node2Vec代码实现

#### 学习算法

采样完顶点序列后，剩下的步骤就和deepwalk一样了，用word2vec去学习顶点的embedding向量。 值得注意的是node2vecWalk中不再是随机抽取邻接点，而是按概率抽取，node2vec采用了Alias算法进行顶点采样。

[Alias Method:时间复杂度O(1)的离散采样方法125 赞同 · 23 评论文章![img](https://pic2.zhimg.com/v2-775c1610ad114ca7d2107348c70345cd_180x120.jpg)](https://zhuanlan.zhihu.com/p/54867139)

#### node2vec核心代码

![img](https://pic4.zhimg.com/80/v2-dae49695db78fda4ff11284e932a7c43_720w.webp)

#### node2vecWalk

通过上面的伪代码可以看到，node2vec和deepwalk非常类似，主要区别在于顶点序列的采样策略不同，所以这里我们主要关注**node2vecWalk**的实现。

由于采样时需要考虑前面2步访问过的顶点，所以当访问序列中只有1个顶点时，直接使用当前顶点和邻居顶点之间的边权作为采样依据。 当序列多余2个顶点时，使用文章提到的有偏采样。

```python
def node2vec_walk(self, walk_length, start_node):
    G = self.G    
    alias_nodes = self.alias_nodes    
    alias_edges = self.alias_edges
    walk = [start_node]
    while len(walk) < walk_length:        
        cur = walk[-1]        
        cur_nbrs = list(G.neighbors(cur))        
        if len(cur_nbrs) > 0:            
            if len(walk) == 1:                
                walk.append(cur_nbrs[alias_sample(alias_nodes[cur][0], alias_nodes[cur][1])])            
            else:                
                prev = walk[-2]                
                edge = (prev, cur)                
                next_node = cur_nbrs[alias_sample(alias_edges[edge][0],alias_edges[edge][1])]                
                walk.append(next_node)        
        else:            
            break
    return walk
```

#### 构造采样表

`preprocess_transition_probs`分别生成`alias_nodes`和`alias_edges`，`alias_nodes`存储着在每个顶点时决定下一次访问其邻接点时需要的alias表（**不考虑当前顶点之前访问的顶点**）。`alias_edges`存储着在前一个访问顶点为 t ，当前顶点为 v 时决定下一次访问哪个邻接点时需要的alias表。

`get_alias_edge`方法返回的是在上一次访问顶点 t ，当前访问顶点为 v 时到下一个顶点 x 的未归一化转移概率 ![image-20230222164222830](https://raw.githubusercontent.com/wumorfr/photo/master/image-20230222164222830.png)

```python
def get_alias_edge(self, t, v):
    G = self.G    
    p = self.p    
    q = self.q
    unnormalized_probs = []    
    for x in G.neighbors(v):        
        weight = G[v][x].get('weight', 1.0)# w_vx        
        if x == t:# d_tx == 0            
            unnormalized_probs.append(weight/p)        
        elif G.has_edge(x, t):# d_tx == 1            
            unnormalized_probs.append(weight)        
        else:# d_tx == 2            
            unnormalized_probs.append(weight/q)    
    norm_const = sum(unnormalized_probs)    
    normalized_probs = [float(u_prob)/norm_const for u_prob in unnormalized_probs]
    return create_alias_table(normalized_probs)

def preprocess_transition_probs(self):
    G = self.G
    alias_nodes = {}    
    for node in G.nodes():        
        unnormalized_probs = [G[node][nbr].get('weight', 1.0) for nbr in G.neighbors(node)]        
        norm_const = sum(unnormalized_probs)        
        normalized_probs = [float(u_prob)/norm_const for u_prob in unnormalized_probs]                 
        alias_nodes[node] = create_alias_table(normalized_probs)
    alias_edges = {}
    for edge in G.edges():        
        alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
    self.alias_nodes = alias_nodes    
    self.alias_edges = alias_edges
    return
```

#### node2vec应用

使用node2vec在wiki数据集上进行节点分类任务和可视化任务。 wiki数据集包含 2,405 个网页和17,981条网页之间的链接关系，以及每个网页的所属类别。 通过简单的超参搜索，这里使用p=0.25,q=4的设置。

```python
G = nx.read_edgelist('../data/wiki/Wiki_edgelist.txt',create_using=nx.DiGraph(),nodetype=None,data=[('weight',int)])

model = Node2Vec(G,walk_length=10,num_walks=80,p=0.25,q=4,workers=1)
model.train(window_size=5,iter=3)    
embeddings = model.get_embeddings()

evaluate_embeddings(embeddings)
plot_embeddings(embeddings)
```

### 2.5 Node2Vec与DeepWalk相比的优点与特点

1. 优点
   - Node2Vec解决图嵌入问题，将图中的每个节点映射为一个向量（嵌入）
   - 向量（嵌入）包含了节点的语义信息（相邻社群和功能角色）
   - 语义相似的节点，向量（嵌入）的距离也近。
   - 向量（嵌入）用于后续的分类，聚类，link Predictin，推荐等任务。
2. 特点
   - 在DeepWalk完全随机游走的基础上，Node2Vec增加p,q参数，实现有偏随机游走。不同的p,q组合，对应了不同的探索范围和节点语义。
   - DFS深度优先探索，相邻的节点，向量（嵌入）距离相近
   - BFS广度优先探索，相同功能角色的节点，向量（嵌入）距离相近。
   - DeepWalk时Node2Vec在p=1,q=1的特例

### 2.6 Node2Vec的优缺点

1. 优点
   - 首次通过调节p,q值，实现了有偏随机游走，探索节点社群，功能等不同属性
   - 首次将节点分类用于Link Prediction
   - 可解释性，可扩展性好，性能卓越
2. 缺点
   - 需要大量随机游走序列训练
   - 距离较远的两个界定啊无法直接相互影响。看不到全图信息。（图神经网络）
   - 无监督，仅编码图的连接信息，没有利用节点的属性特征（图卷积）
   - 没有真正用到神经网络和深度学习	



# 参考文献

[1]  [斯坦福CS224W图机器学习、图神经网络、知识图谱【同济子豪兄】](https://www.bilibili.com/video/BV1pR4y1S7GA?vd_source=872fc2755b4c0ffb1be2bc7240a69fed)

[2] [【Graph Embedding】DeepWalk：算法原理，实现和应用](https://zhuanlan.zhihu.com/p/56380812)

[3] [【Graph Embedding】node2vec：算法原理，实现和应用](https://zhuanlan.zhihu.com/p/56542707)
