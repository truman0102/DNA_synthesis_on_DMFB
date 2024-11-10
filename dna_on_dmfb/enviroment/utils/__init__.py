import numpy as np
import signal
import threading

def read_products_from_json(json_file, sequence="1"):
    import json
    with open(json_file, 'r') as f:
        data = json.load(f)
    products = data[sequence]['products']
    return products

def process_size(size):
    if isinstance(size, int):
        return size, size
    elif isinstance(size, (list, tuple, np.ndarray)):
        assert len(size) == 2
        return size
    else:
        raise ValueError("size must be int or list or tuple or np.ndarray")


class TimeLimitExceeded(Exception):
    """自定义异常：用于标识时间超限"""

    pass


def timeout_handler(signum, frame):
    """信号处理器：当时间超限时抛出异常"""
    raise TimeLimitExceeded


def execute_with_timeout(func, args, kwargs, time_limit):
    """
    执行目标函数并在规定时间内返回结果。

    参数:
    - func: 目标函数
    - args: 目标函数的位置参数
    - kwargs: 目标函数的关键字参数
    - time_limit: 以秒为单位的时间限制
    """
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, time_limit)

    try:
        result = func(*args, **kwargs)
    except TimeLimitExceeded:
        result = None
    finally:
        signal.alarm(0)  # 关闭计时器

    return result


def try_with_different_params(func, params_list, time_limit):
    """
    使用不同参数在指定时间内尝试执行函数。

    参数:
    - func: 目标函数
    - params_list: 参数集合的列表，每个参数集合是一个二元组 (args, kwargs)
    - time_limit: 每次调用的时间限制（秒）

    返回:
    - 成功执行成功的结果
    - 如果所有尝试失败，则返回 None
    """
    for args, kwargs in params_list:
        result = execute_with_timeout(func, args, kwargs, time_limit)
        if result is not None:
            return result
    return None


def get_overlap_or_projection(region_a, region_b):
    """
    计算两个区域的重叠部分，如果没有重叠，计算区域B中心在区域A边界上的投影点。

    参数:
        region_a (tuple): 主区域的坐标 (x1, y1, x2, y2)
        region_b (tuple): 另一区域的坐标 (x1, y1, x2, y2)

    返回:
        tuple: 如果有重叠，返回重叠区域的坐标 (x1, y1, x2, y2)；
               如果没有重叠，返回投影点的坐标 (x, y)
    """
    # 计算重叠区域
    overlap_x1 = max(region_a[0], region_b[0])
    overlap_y1 = max(region_a[1], region_b[1])
    overlap_x2 = min(region_a[2], region_b[2])
    overlap_y2 = min(region_a[3], region_b[3])

    # 检查是否重叠
    if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
        return (overlap_x1, overlap_y1, overlap_x2, overlap_y2)

    # 计算中心点
    center_a = ((region_a[0] + region_a[2]) / 2, (region_a[1] + region_a[3]) / 2)
    center_b = ((region_b[0] + region_b[2]) / 2, (region_b[1] + region_b[3]) / 2)

    # 计算方向向量
    dx = center_b[0] - center_a[0]
    dy = center_b[1] - center_a[1]

    # 初始化t的值列表
    t_list = []

    # 计算与区域A边界的交点参数t
    if dx != 0:
        t_left = (region_a[0] - center_a[0]) / dx
        t_right = (region_a[2] - center_a[0]) / dx
        t_list.extend([t_left, t_right])
    if dy != 0:
        t_top = (region_a[1] - center_a[1]) / dy
        t_bottom = (region_a[3] - center_a[1]) / dy
        t_list.extend([t_top, t_bottom])

    # 筛选正的t值
    t_edge = [t for t in t_list if t > 0]

    if not t_edge:
        # 没有正的t值，说明投影点在中心点
        return center_a

    # 选择最小的正t值
    t_min = min(t_edge)

    # 计算交点坐标
    x = int(center_a[0] + dx * t_min)
    y = int(center_a[1] + dy * t_min)

    # 确保交点在区域A边界上
    x = max(region_a[0], min(x, region_a[2]))
    y = max(region_a[1], min(y, region_a[3]))

    return (x, y)
