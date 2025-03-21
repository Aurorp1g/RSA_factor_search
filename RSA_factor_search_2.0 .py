import math
import os
import random
import time
import itertools
from itertools import chain
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def LCG(start, end, M):
    """生成器模式的线性同余随机数生成"""
    A = (random.randint(0, 2500) << 2) + 1
    x = start
    modulus = 1 << M
    range_size = end - start + 1
    for _ in range(modulus):
        x = (A * x + 1) % modulus
        yield x % range_size + start

def decompose(n):
    """因数分解函数"""
    if n == 0:
        return (0, 0, 0)
    
    M = (n & -n).bit_length() - 1
    K_plus_R = n >> M
    
    max_k = 1
    sqrt_val = int(math.isqrt(K_plus_R)) + 1
    for i in chain(range(1, sqrt_val, 4), reversed(range(1, sqrt_val, 4))):
        if K_plus_R % i == 0 and i % 4 == 1:
            max_k = max(max_k, i)
            break
    return (M, max_k, n - (max_k << M))

def small_search(square_x, square_y, e, pb, pu, g, N):
    """优化后的小范围搜索函数，采用随机行走策略"""
    x0, y0 = square_x, square_y
    x_min, x_max = 1, pb + 1
    y_lower = g * (pb + 1) + 1
    y_upper = g * (pu - pb -1) + 1
    max_steps = e * e

    # 初始位置检查
    if x_min <= x0 <= x_max and y_lower <= y0 <= y_upper:
        h = pb + x0 - ((y0 - 1) // g)
        if (d := gcd(h, N)) > 1:
            return d
    if (d := gcd(x0, N)) > 1:
        return d
    if (d := gcd(y0, N)) > 1:
        return d

    for _ in range(max_steps):
        direction = random.randint(1, 4)
        if direction == 1:
            x = x0 + 1
            y = y0 + g * x
        elif direction == 2:
            x = x0 + 1
            y = y0 - g * x
        elif direction == 3:
            x = x0 - 1
            y = y0 + g * x
        else:  # direction == 4
            x = x0 - 1
            y = y0 - g * x

        # 新位置有效性检查
        in_range = (x_min <= x <= x_max) and (y_lower <= y <= y_upper)
        h = pb + x - ((y - 1) // g) if in_range else random.randint(1, N-1)

        # 三重因子检查
        if (d := gcd(x, N)) > 1 or (d := gcd(y, N)) > 1 or (d := gcd(h, N)) > 1:
            return d

        x0, y0 = x, y

    return -1

def parallel_search(tasks):
    """并行搜索执行器"""
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(small_search, *task) for task in tasks]
        try:
            for future in as_completed(futures):
                if (result := future.result()) > 1:
                    executor.shutdown(wait=False, cancel_futures=True)  # 取消未完成任务
                    return result
        finally:
            executor.shutdown(wait=True)  # 确保资源释放
    return -1

def batch_search(args):
    """批量搜索函数，处理连续区域块"""
    start_idx, end_idx, initial_x, initial_y, dir_x, dir_y, e, pb, pu, g, N = args
    for i in range(start_idx, end_idx + 1):
        current_x = initial_x + dir_x * e * (i + 1)
        current_y = initial_y + dir_y * (i + 1)
        res = small_search(current_x, current_y, e, pb, pu, g, N)
        if res > 1:
            return res
    return -1

def parallel_search_batch(tasks):
    """批量并行搜索执行器"""
    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        futures = {executor.submit(batch_search, task): task for task in tasks}
        try:
            for future in as_completed(futures):
                if (result := future.result()) > 1:
                    executor.shutdown(wait=False, cancel_futures=True)  # 强制终止
                    return result
        finally:
            executor.shutdown(wait=True)  # 确保资源释放
    return -1

def rsa_factor_search(N, g, eta):
    if N <= 3:
        return N
    
    if g == 0 or eta < 1:
        print("无效参数: g不能为0且eta必须≥1")
        return None
    
    pb = math.isqrt(N)
    pu = N - 1 - pb
    # 随机选择搜索方向
    dir_x = random.choice((-1, 1))
    dir_y = math.floor(dir_x * (-1/g))
    
    # 中心点检测优化
    Cx = (N - pu + 1) // 2
    Cy = g * (N + pb + 1) // 4 + 1
    if (res := gcd(N, pu + Cx - ((Cy - 1)//g) - 1)) > 1:
        return res
    
    # 参数计算
    e = max(int(N ** (1/(2*eta))), 2)
    if (n := pb // e) < 1:
        print("eta参数过大")
        return None
    
    M, K, R = decompose(n)
    
    # 第一阶段搜索
    current_x, current_y = Cx + random.randint(-e, e), Cy + random.randint(-e, e)
    if (res := small_search(current_x, current_y, e, pb, pu, g, N)) > 1:
        return res
 

    # 第二阶段并行搜索
    initial_x, initial_y = current_x, current_y
    tasks = []
    # 将R个小区域分为大区域，根据CPU核心数动态计算大区域大小
    batch_size = max(1, (R-1) // (cpu_count()*2))
    for i in range(0, R-1, batch_size):
        end = min(i + batch_size - 1, R-2)
        tasks.append((
            i, end,
            initial_x, initial_y,
            dir_x, dir_y,
            e, pb, pu, g, N
        ))
    
    if tasks and (res := parallel_search_batch(tasks)) > 1:
        return res
    
    # 更新坐标到第二阶段结束后的状态
    current_x = initial_x + dir_x * e * (R-1)
    current_y = initial_y + dir_y * (R-1)
    
    # 并行化第三阶段搜索
    tasks = []
    group_step = (dir_x * e) << M, dir_y << M
    group_x, group_y = current_x + dir_x * e, current_y + dir_y
    
    for _ in range(K):
        group_x += group_step[0]
        group_y += group_step[1]
        lcg_gen = LCG(0, (1<<M)-1, M)
        num = next(itertools.islice(lcg_gen, random.randint(0, (1<<M)-1), None))
        tasks.append((
            group_x + num * dir_x * e,
            group_y + num * dir_y,
            e, pb, pu, g, N
        ))
    
    if (res := parallel_search(tasks)) > 1:
        return res
    
    return None

if __name__ == "__main__":
    N = int(input("请输入N的值: "))
    g = int(input("请输入g的值: "))
    eta = int(input("请输入eta的值: "))
    
    if g == 0 or eta == 0:
        print("输入的g或eta值为0,请重新输入")
    else:
        time_start = time.perf_counter()
        result = rsa_factor_search(N, g, eta)
        time_end = time.perf_counter()
        print(f"Found factor: \033[31m{result}\033[0m")
        print(f"Time cost: \033[34m{time_end - time_start:.3f}\033[0m seconds")
        os._exit(0)  # 强制退出，防止程序继续运行