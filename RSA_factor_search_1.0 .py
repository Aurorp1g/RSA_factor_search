import math
import random
import time
import itertools
from itertools import chain
from concurrent.futures import ProcessPoolExecutor, as_completed

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
    """小范围搜索函数"""
    y_upper = g * (pu - pb -1) + 1
    y_lower = g * (pb + 1) + 1
    x_min, x_max = 1, pb + 1
    base_x = square_x - e//2
    base_y = square_y - e//2

    for offset in range(e*e):
        i, j = divmod(offset, e)
        x = base_x + i
        y = base_y + j
        
        if not (x_min <= x <= x_max) or not (y_lower <= y <= y_upper):
            h = random.randint(1, N-1)
        else:
            h = pu + x - ((y-1)//g) - 1
        
        if (res := gcd(h, N)) > 1:
            return res
    return -1

def parallel_search(tasks):
    """并行搜索执行器"""
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(small_search, *task) for task in tasks]
        for future in as_completed(futures):
            if (result := future.result()) > 1:
                executor.shutdown(wait=False)
                return result
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
    
    # 第二阶段搜索
    for _ in range(R-1):
        current_x += dir_x * e
        current_y += dir_y
        if (res := small_search(current_x, current_y, e, pb, pu, g, N)) > 1:
            return res
    
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
