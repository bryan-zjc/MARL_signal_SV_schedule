# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 10:13:23 2025

@author: poc
"""

from __future__ import absolute_import
from __future__ import print_function

import os
import sys  # 导入sys模块
# sys.setrecursionlimit(10**5)  # 将默认的递归深度修改为3000
import numpy as np
import random
import pandas as pd
from numpy import nan
import copy
import xml.etree.ElementTree as ET


def extract_boundary_edges(net_file):
    # 解析 .net.xml 文件
    tree = ET.parse(net_file)
    root = tree.getroot()

    # 记录所有 edge 的连接情况
    all_edges = set()
    from_edges = set()  # 有下游的 edge
    to_edges = set()    # 有上游的 edge

    # 遍历所有的 edge
    for edge in root.findall('edge'):
        edge_id = edge.attrib['id']
        # 跳过内部 edge
        if 'function' in edge.attrib and edge.attrib['function'] == 'internal':
            continue
        all_edges.add(edge_id)

    # 遍历所有的 connection，找到所有有连接的 edge
    for connection in root.findall('connection'):
        from_edge = connection.attrib['from']
        to_edge = connection.attrib['to']

        from_edges.add(from_edge)  # 有下游连接的 edge
        to_edges.add(to_edge)      # 有上游连接的 edge

    # 流入的 edge：没有上游连接
    inflow_edges = all_edges - to_edges

    # 流出的 edge：没有下游连接
    outflow_edges = all_edges - from_edges

    return inflow_edges, outflow_edges


# 输入 net.xml 文件路径
net_file = '5x5.net.xml'

# 提取流入和流出的 edge
incoming_link, outcoming_link = extract_boundary_edges(net_file)
incoming_link = list(incoming_link)
outcoming_link = list(outcoming_link)
# 输出结果
print("Inflow edges:")
for edge in incoming_link:
    print(edge)

print("Outflow edges:")
for edge in outcoming_link:
    print(edge)


df = 1.0
df_text = int(df*100)

# 每个 OD 对的基本需求（总需求均分）
demand_per_od = (800*df)/(len(incoming_link)-1)

# RV 渗透率
p = 0.2
p_text = int(p*100)

# 新增参数 beta（β）用于 RV 需求的不均衡，β=1 表示均衡；β>1 表示不均衡，且最大的 OD 流量是最小的 β 倍
beta = 2.5  # 示例值，可调整
beta_text = int(beta*100)


# 生成 trip 文件，同时对 RV 流量引入 β 不均衡
def generate_tripfile(incoming_link, outcoming_link, demand_per_od, beta):
    random.seed(37)  # 使随机结果可复现

    # 先收集所有合法的 OD 对
    flows = []
    for m in range(len(incoming_link)):
        for n in range(len(outcoming_link)):
            if m != n:
                flows.append((incoming_link[m], outcoming_link[n]))
    N = len(flows)
    
    # 对 RV 流量乘子设置：
    # 希望乘子序列线性分布，从 m_min 到 m_max，
    # 且 (m_min + m_max)/2 = 1，从而保证总需求不变，同时 m_max/m_min = beta
    m_min = 2 / (1 + beta)
    m_max = 2 * beta / (1 + beta)
    multipliers = np.linspace(m_min, m_max, N)
    multipliers = list(multipliers)
    # 随机打乱乘子顺序，避免与 OD 对顺序产生直接关联
    random.shuffle(multipliers)
    
    # 修改文件名以体现 β 值
    filename = f'beta/trip_beta{beta_text}.trip.xml'
    with open(filename, "w") as routes:
        print("""<routes>
    <vType id="NV" accel="3" decel="5" tau="1.0" speedFactor="1.0" speedDev="0.0" sigma="0.5" length="5" minGap="2" maxSpeed="60" guiShape="passenger" color="0,1,0"/>
    <vType id="RV" accel="3" decel="5" tau="1.0" speedFactor="1.0" speedDev="0.0" sigma="0.5" length="5" minGap="2" maxSpeed="60" guiShape="passenger" color="1,0,0"/>""", file=routes)
        
        number = 0
        # 遍历所有 OD 对，分别生成 NV 与 RV 的 flow
        for (from_link, to_link) in flows:
            nv_demand = demand_per_od * (1 - p)
            # 对应当前 OD 对的 RV 流量乘子
            multiplier = multipliers.pop()
            rv_demand = demand_per_od * p * multiplier
            print('''<flow id="OD%i_NV" begin="0" end="3600" vehsPerHour="%s" type="NV" from="%s" to="%s"/>''' %
                  (number, nv_demand, from_link, to_link), file=routes)
            print('''<flow id="OD%i_RV" begin="0" end="3600" vehsPerHour="%s" type="RV" from="%s" to="%s"/>''' %
                  (number, rv_demand, from_link, to_link), file=routes)
            number += 1

        print("</routes>", file=routes)


generate_tripfile(incoming_link, outcoming_link, demand_per_od, beta)

#duarouter --route-files=trip_beta150.trip.xml --net-file=5x5.net.xml --output-file=Demand_beta150.rou.xml --randomize-flows true --departspeed 10 --weights.random-factor 1.5 --seed 30