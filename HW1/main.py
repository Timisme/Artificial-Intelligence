#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from functions import func


# # 載入 Input.txt 資料

# In[2]:


with open('input.txt','r') as f:
    contents = f.readlines()
    f.close()


# In[3]:


intervals = []
for content in contents:
    interval = []
    for interval_id in range(2):
        interval.append(float(content.split('\n')[0].split(',')[interval_id]))
    intervals.append(interval)


# In[4]:


x_interval = intervals[0]
y_interval = intervals[1]


# # Q1 暴力法求function 的全域最佳解

# In[10]:


x_range = list(range(int(x_interval[0]),int(x_interval[1])+1,1))
y_range = list(range(int(y_interval[0]),int(y_interval[1])+1,1))


# In[17]:


z_list = []
for x in x_range:
    for y in y_range:
        if (x == x_range[0]) & (y==y_range[0]):
            z = func(x,y)
        elif func(x,y)<z:
            z = func(x,y)
            if z < min(z_list):
                x_optim, y_optim = x, y 
        z_list.append(z)


# In[18]:


print('the global minimum of the loss func is {:4f}\nthe optimum x: {} , y: {}'.format(z,x_optim,y_optim))


# # Q2 模擬退火法求函數全域最佳解

# In[6]:


def clip(x, xinterval=True):
    """ Force x to be in the interval."""
    if xinterval == True:
        a, b = x_interval
    else:
        a, b = y_interval 
    return max(min(x, b), a)


# In[7]:


def random_start():
    """ Random point in the interval."""
    a, b = x_interval
    c, d = y_interval
    return np.random.randint(a,b),np.random.randint(c,d)


# In[8]:


def random_delta():
    rand_value = np.random.random()
    if rand_value > (2/3):
        return 0 
    elif rand_value < (1/3):
        return 1
    else:
        return -1


# In[30]:


def simulated_annealing(init_T, cooling_rate, Reheat_prob, T_threshold, max_steps):
        x, y = random_start()
        cost = func(x,y)
        states, costs = [x,y],[cost]
        T = init_T
        step=0
        while step in range(max_steps):
            new_x, new_y = clip(x + random_delta(),True), clip(y + random_delta(),False)
            while (new_x == x) & (new_y == y):
                new_x, new_y = clip(x + random_delta(),True), clip(y + random_delta(),False)
            new_cost = func(new_x,new_y)
            step += 1
            if T < T_threshold:
                break
            elif costs.count(min(costs)) > 200:
                break      
            elif new_cost < cost:
                x, y, cost = new_x, new_y, new_cost
                states.append([x,y])
                costs.append(cost)
                T = cooling_rate * T
            elif np.exp(- (new_cost - cost) /T)> np.random.random():
                x, y, cost = new_x, new_y, new_cost
                if np.random.random() < Reheat_prob:
                    T = T / cooling_rate
            elif np.random.random() < Reheat_prob:
                T = T / cooling_rate #升溫
            print('current step:{:4.0f}, current state: X: {:5.1f}, Y: {:5.1f}, current T:{:7.3f}, current cost: {:7.3f}'.format(step,x,y,T,cost),end='\r')
        return print('\nx_optimum: {}, y_optimum: {}\nminimum : {:.3f}'.format(x,y,cost),end='\r')
        


# ## Running Simulated Annealing Algorithm 

# In[33]:


simulated_annealing(init_T=10,cooling_rate=0.998, Reheat_prob=0.2, T_threshold=0.05, max_steps=10000)


# -----

# ## 以下是讓模擬退火過程跑10次測試找出global minimum的機率

# In[24]:


"""
ans_list = []
for i in range(10):
    ans = simulated_annealing(init_T=10,cooling_rate=0.998, Reheat_prob=0.2, T_threshold=0.05, max_steps=10000)
    print('epoch:{}'.format(i+1))
    ans_list.append(ans)
for element in set(ans_list):
    print('{} : {} times'.format(element,ans_list.count(element)))
"""


# In[ ]:




