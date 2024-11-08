import random

class Node:
    def __init__(self, value, mutation_positions=None):
        self.value = value  # DNA序列
        self.mutation_positions = mutation_positions if mutation_positions is not None else []  # 记录突变位置列表（按顺序）
        self.children = []  # 子节点列表

def generate_random_dna(length):
    """生成指定长度的随机DNA序列"""
    bases = ['A', 'T', 'C', 'G']
    return ''.join(random.choice(bases) for _ in range(length))

def generate_mutation_tree(node, sequence_length, max_tree_height, max_total_nodes, max_children_per_node, current_height, total_nodes):
    """递归生成突变树"""
    if current_height >= max_tree_height or total_nodes[0] >= max_total_nodes:
        return

    # 计算当前节点可以生成的最大子节点数量
    max_possible_children = min(max_children_per_node, max_total_nodes - total_nodes[0])

    # 确定可用的突变位置
    # 使用父节点的最后一个突变位点，或者-1表示从位置0开始
    last_mutation_pos = node.mutation_positions[-1] if node.mutation_positions else 0
    available_positions = list(range(last_mutation_pos + 1, sequence_length))
    if not available_positions:
        return

    # 随机决定当前节点的子节点数量
    num_children = random.randint(1, max_possible_children)

    for _ in range(num_children):
        if total_nodes[0] >= max_total_nodes or not available_positions:
            break

        # 从可用位置中随机选择一个突变位置
        mutation_pos = random.choice(available_positions)
        available_positions.remove(mutation_pos)

        # 进行突变
        parent_sequence = node.value
        new_sequence = list(parent_sequence)
        parent_base = parent_sequence[mutation_pos]
        bases = ['A', 'T', 'C', 'G']
        bases.remove(parent_base)
        new_base = random.choice(bases)
        new_sequence[mutation_pos] = new_base
        child_sequence = ''.join(new_sequence)

        # 创建子节点
        child_mutation_positions = node.mutation_positions + [mutation_pos]  # 记录突变位置，保证顺序
        child_node = Node(value=child_sequence, mutation_positions=child_mutation_positions)
        node.children.append(child_node)
        total_nodes[0] +=1

        # 递归生成子节点的子树
        generate_mutation_tree(child_node, sequence_length, max_tree_height, max_total_nodes, max_children_per_node, current_height + 1, total_nodes)

def node_to_dict(node):
    """将节点转换为字典形式"""
    return {
        "value": node.value,
        "children": [node_to_dict(child) for child in node.children]
    }

def create_dna_mutation_tree(sequence_length=10, max_tree_height=4, max_total_nodes=10, max_children_per_node=3):
    """创建DNA序列的突变树"""
    # 生成根节点（初始DNA序列）
    root_sequence = generate_random_dna(sequence_length)
    root_node = Node(value=root_sequence)
    total_nodes = [1]  # 使用列表来传递可变整数

    # 生成突变树
    generate_mutation_tree(
        node=root_node,
        sequence_length=sequence_length,
        max_tree_height=max_tree_height,
        max_total_nodes=max_total_nodes,
        max_children_per_node=max_children_per_node,
        current_height=1,
        total_nodes=total_nodes
    )

    # 将树转换为字典形式
    return node_to_dict(root_node)

def get_mutated_position(parent, child):
    for i, (p, c) in enumerate(zip(parent, child)):
        if p != c:
            return i

# 示例使用
if __name__ == "__main__":
    mutation_tree = create_dna_mutation_tree(sequence_length=10, max_tree_height=4, max_total_nodes=10, max_children_per_node=3)
    from pprint import pprint
    pprint(mutation_tree)
    nodes = []
    root = mutation_tree
    nodes.append(root)
    while len(nodes) > 0:
        node = nodes.pop(0)
        parent = node['value']
        for child in node['children']:
            child_value = child['value']
            print(f"{parent} -> {child_value} at {get_mutated_position(parent, child_value)}")
            nodes.append(child)