import pulp as pl

def bin_packing_cutting_plane(item_sizes: list[float], bin_capacity: float) -> tuple[list[list[float]], int]:
    """
    割平面法求解装箱问题（Bin Packing Problem），返回绝对最优解
    :param item_sizes: 待装箱物品的体积列表
    :param bin_capacity: 单个箱子的固定容量
    :return: (装箱结果列表, 使用的箱子总数)
             装箱结果格式：[[箱1物品], [箱2物品], ...]
    """
    # result, bin_num = bin_packing_cutting_plane(ITEM_SIZES, BIN_CAPACITY)
    # item_sizes = ITEM_SIZES
    # bin_capacity = BIN_CAPACITY
    
    # 1. 边界校验：单个物品体积超过箱子容量，无法装箱
    for s in item_sizes:
        if s > bin_capacity:
            raise ValueError(f"物品体积 {s} 超过箱子容量 {bin_capacity}，无法装箱！")
    n = len(item_sizes)  # 物品数量
    if n == 0:
        return [], 0

    # 2. 创建整数规划模型（最小化目标）
    model = pl.LpProblem("Bin_Packing_Cutting_Plane", pl.LpMinimize)

    # 3. 定义决策变量
    # x[i][j]：物品i是否放入箱子j (0-1变量)，i/j从0开始索引
    x = pl.LpVariable.dicts(
        name="x",
        indices=(range(n), range(n)),
        cat=pl.LpBinary
    )
    print(x)
    # y[j]：箱子j是否被使用 (0-1变量)
    y = pl.LpVariable.dicts(
        name="y",
        indices=range(n),
        cat=pl.LpBinary
    )

    # 4. 添加目标函数：最小化使用的箱子总数
    model += pl.lpSum(y[j] for j in range(n)), "Minimize_Bin_Number"

    # 5. 添加约束条件（严格对应整数规划模型）
    # 约束1：每个物品必须且只能放入一个箱子
    for i in range(n):
        model += pl.lpSum(x[i][j] for j in range(n)) == 1, f"Item_{i}_Assign_Constraint"

    # 约束2：每个箱子的物品总体积 ≤ 箱子容量（仅当箱子被使用时生效）
    for j in range(n):
        model += pl.lpSum(item_sizes[i] * x[i][j] for i in range(n)) <= bin_capacity * y[j], f"Bin_{j}_Capacity_Constraint"

    # 约束3：装物品的箱子必须被标记为使用（强化逻辑约束，提升求解效率）
    for i in range(n):
        for j in range(n):
            model += x[i][j] <= y[j], f"Logic_Constraint_{i}_{j}"

    # 6. 求解模型（PuLP内置割平面法求解器，自动处理分数解、构造割平面约束）
    # CBC求解器：支持割平面法+分支定界法，是整数规划的最优开源求解器
    solver = pl.PULP_CBC_CMD(msg=0)  # msg=0 关闭求解过程日志，msg=1 开启
    model.solve(solver)

    # 7. 校验求解状态，提取结果
    if pl.LpStatus[model.status] != "Optimal":
        raise RuntimeError("割平面法求解失败，未找到最优解！")

    # 8. 整理装箱结果：遍历所有箱子，收集其中的物品
    packing_result = []
    used_bin_count = 0
    for j in range(n):
        # 箱子j被使用（y[j] = 1），则收集该箱子内的物品
        if pl.value(y[j]) == 1:
            bin_items = [item_sizes[i] for i in range(n) if pl.value(x[i][j]) == 1]
            packing_result.append(bin_items)
            used_bin_count += 1

    return packing_result, used_bin_count

def print_optimal_result(packing_result: list[list[float]], bin_capacity: float, used_bin: int):
    """格式化输出最优装箱结果"""
    print(f"\n🎉 割平面法求解完成（绝对最优解）")
    print(f"箱子容量：{bin_capacity} | 最优解：共使用 {used_bin} 个箱子")
    print("-" * 80)
    total_item_volume = sum(sum(bin_items) for bin_items in packing_result)
    total_bin_capacity = used_bin * bin_capacity
    utilization_rate = (total_item_volume / total_bin_capacity) * 100  # 箱子整体利用率
    for idx, bin_items in enumerate(packing_result, start=1):
        current_volume = sum(bin_items)
        remaining = bin_capacity - current_volume
        print(f"箱子{idx}：物品={bin_items} | 已用容量={current_volume:.2f} | 剩余容量={remaining:.2f}")
    print("-" * 80)
    print(f"箱子整体利用率：{utilization_rate:.2f}% | 所有物品总体积：{total_item_volume:.2f}")

#%%
# ------------------- 测试示例 -------------------
if __name__ == "__main__":
    # 测试用例1：经典示例（物品数8，箱子容量10）
    BIN_CAPACITY = 10.0
    ITEM_SIZES = [8, 3, 4, 5, 2, 7, 1, 6]

    # 调用割平面法求解
    result, bin_num = bin_packing_cutting_plane(ITEM_SIZES, BIN_CAPACITY)
    # 输出最优结果
    print_optimal_result(result, BIN_CAPACITY, bin_num)

    # 测试用例2：小规模示例（可自行替换）
    # BIN_CAPACITY = 15.0
    # ITEM_SIZES = [10, 7, 6, 5, 4, 3, 2]
    # result, bin_num = bin_packing_cutting_plane(ITEM_SIZES, BIN_CAPACITY)
    # print_optimal_result(result, BIN_CAPACITY, bin_num)
