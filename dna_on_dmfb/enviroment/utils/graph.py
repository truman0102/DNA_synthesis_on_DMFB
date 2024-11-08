from scipy.stats import linregress
from collections import Counter
import math
import networkx as nx
import numpy as np
import random
from dna_on_dmfb.enviroment.constants import dNTPs, ddNTPs

def get_graph(situation: dict) -> nx.Graph:
    def count_prefix(s1, s2):
        prefix = ""
        for i in range(min(len(s1), len(s2))):
            if s1[i] != s2[i]:
                break
            prefix += s1[i]
        return prefix
    G = nx.DiGraph()
    dNTPs = ['dATP', 'dTTP', 'dCTP', 'dGTP']
    ddNTPs = ['d'+i for i in dNTPs]
    zone_parent = {}
    depth = {}
    for i in dNTPs:
        G.add_node(i)
    for i in ddNTPs:
        G.add_node(i)
    root = situation.get("products", {})
    # w_max = len(root['value'])
    root['prefix'] = ""
    G.add_edge("primer", root['value'], weight=1)
    G.add_edge("eluent", root['value'], weight=1)
    G.add_node("buffer")
    if len(root) == 0:
        return
    nodes = [root]
    depth[root['value']] = 1
    while len(nodes) > 0:
        for i, node in enumerate(nodes):
            G.add_edge("agarose", node['value'], weight=1)
            nodes.pop(i)
            counter = Counter(node['value'][len(node['prefix']):][:-1])
            for k, v in counter.items():
                G.add_edge("d"+k+"TP", node['value'], weight=v)
            G.add_edge("dd"+node['value'][-1]+"TP", node['value'], weight=1)
            children = node.get("children",[])
            for child in children:
                p = count_prefix(node['value'], child['value'])
                # G.add_edge(node['value'], child['value'], weight=len(p)/len(children)/2)
                G.add_edge(node['value'], child['value'], weight=1)
                child['prefix'] = p
                nodes.append(child)
                zone_parent[child['value']] = node['value']
                depth[child['value']] = depth[node['value']] + 1
    # for i, edge in enumerate(G.edges()):
    #     if all([n in zones for n in edge]):
    #         G.edges[edge]['weight'] += w_max/n_zones
    # for i, edge in enumerate(G.edges()):
    #     G.edges[edge]['weight'] /= w_max
    return G, zone_parent, max(depth.values())

def sort_indices_by_line_regression(points):
    # 提取x和y坐标
    x_coords, y_coords = zip(*points)

    # 进行线性回归，得到回归直线的参数
    slope, intercept, r_value, p_value, std_err = linregress(x_coords, y_coords)
    
    # 计算每个点在回归直线上的投影位置
    projections = []
    for x, y in points:
        # 直线方程：y = slope * x + intercept
        # 找到从点到直线的垂直投影的x坐标
        x_proj = (x + slope * y - slope * intercept) / (1 + slope**2)
        projections.append(x_proj)  # 仅保留x坐标用于排序
    
    # 对投影的x坐标进行排序
    sorted_indices = sorted(range(len(projections)), key=lambda i: projections[i])
    
    return sorted_indices

def simulated_annealing_layout_v2(G, fixed_positions, area, min_distance, max_attempts, initial_temp, final_temp, alpha):
    """
    使用模拟退火算法对图进行布局
    G: NetworkX的有向图
    fixed_positions: 字典，key为节点名，value为固定的位置(tuple)
    area: (x1, y1, x2, y2)，可移动节点的位置范围
    min_distance: 节点之间的最小间距
    max_attempts: 更新节点位置时的最大尝试次数
    initial_temp: 初始温度
    final_temp: 终止温度
    alpha: 降温系数
    """
    nodes = list(G.nodes())
    fixed_nodes = set(fixed_positions.keys())
    non_fixed_nodes = [node for node in nodes if node not in fixed_nodes]

    positions = {}
    positions.update(fixed_positions)
    
    x1, y1, x2, y2 = area

    # 初始化非固定节点的位置，确保最小间距
    for node in non_fixed_nodes:
        for attempt in range(max_attempts):
            x = random.uniform(x1, x2)
            y = random.uniform(y1, y2)
            # 检查是否满足最小间距
            too_close = False
            for pos in positions.values():
                if math.hypot(pos[0] - x, pos[1] - y) < min_distance:
                    too_close = True
                    break
            if not too_close:
                positions[node] = (x, y)
                break
        else:
            # 如果无法找到满足条件的位置，保持原位置不变（这里可以随机赋值或设为默认值）
            positions[node] = (random.uniform(x1, x2), random.uniform(y1, y2))

    # 定义能量函数，计算所有边的加权平方距离之和
    def energy(G, positions):
        total_energy = 0.0
        for (u, v) in G.edges():
            pos_u = positions[u]
            pos_v = positions[v]
            weight = G.get_edge_data(u, v).get('weight', 1.0)
            distance = math.hypot(pos_u[0] - pos_v[0], pos_u[1] - pos_v[1])
            total_energy += weight * distance ** 2
        return total_energy

    temp = initial_temp

    while temp > final_temp:
        # 对每个非固定节点尝试更新位置
        for node in non_fixed_nodes:
            current_pos = positions[node]
            old_pos = current_pos
            # 在限定次数内尝试找到满足最小间距的新位置
            for attempt in range(max_attempts):
                # 生成新的位置，做一个小范围的随机移动
                delta = min((x2 - x1), (y2 - y1)) * 0.1  # 移动步长，可以调整
                new_x = current_pos[0] + random.uniform(-delta, delta)
                new_y = current_pos[1] + random.uniform(-delta, delta)
                # 确保新位置在指定区域内
                new_x = min(max(new_x, x1), x2)
                new_y = min(max(new_y, y1), y2)
                new_pos = (new_x, new_y)
                # 检查是否满足最小间距
                too_close = False
                for other_node, pos in positions.items():
                    if other_node != node:
                        if math.hypot(pos[0] - new_x, pos[1] - new_y) < min_distance:
                            too_close = True
                            break
                if not too_close:
                    # 找到满足条件的新位置
                    break
            else:
                # 未找到满足条件的位置，保持原位置不变
                new_pos = old_pos
            # 计算能量差
            positions[node] = new_pos
            new_energy = energy(G, positions)
            positions[node] = old_pos
            old_energy = energy(G, positions)
            delta_e = new_energy - old_energy
            # 决定是否接受新位置
            if delta_e < 0:
                positions[node] = new_pos
            else:
                probability = math.exp(-delta_e / temp)
                if random.random() < probability:
                    positions[node] = new_pos
                else:
                    positions[node] = old_pos
        # 降温
        temp *= alpha

    return positions

def adjust_layout(situation: dict, adjust_area, min_distance=5):
    def get_center(l):
            return (
                (l[0]+l[2])/2,
                (l[1]+l[3])/2
            )
    graph, zone_parent, height = get_graph(situation)
    fixed_positions = {}
    for k, v in situation['elements'].items():
        fixed_positions[k] = get_center(v)
    pos = nx.spring_layout(graph, pos=fixed_positions, fixed=fixed_positions.keys())
    dNTP_sort = sort_indices_by_line_regression([pos[i] for i in dNTPs])
    ddNTP_sort = sort_indices_by_line_regression([pos[i] for i in ddNTPs])
    entrances_dNTPs = situation['dNTPs']['fixed']
    entrances_ddNTPs = situation['ddNTPs']['fixed']
    for i, node_id in enumerate(dNTP_sort):
        fixed_positions[dNTPs[node_id]] = get_center(entrances_dNTPs[i])
    for i, node_id in enumerate(ddNTP_sort):
        fixed_positions[ddNTPs[node_id]] = get_center(entrances_ddNTPs[i])
    pos = simulated_annealing_layout_v2(graph, fixed_positions, adjust_area, min_distance, 1000, 1.0, 0.001, 0.995)
    return graph, pos, zone_parent, height