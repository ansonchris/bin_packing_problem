import pulp as pl
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

# ===================== 1. 配置参数与模拟数据生成 =====================
a = 5    # 单分组占比下限 a%
b = 25   # 单分组占比上限 b%
C = 35   # 相邻分组占比和上限 C%

def generate_simulate_data() -> Dict[str, pd.DataFrame]:
    data_dict = {}
    # 生成10个训练集
    for i in range(10):
        np.random.seed(i)
        df = pd.DataFrame({
            "customer_id": range(1000),
            "label": np.random.choice([0,1], size=1000, p=[0.85, 0.15]),
            "score": np.random.uniform(0, 100, size=1000)
        })
        data_dict[f"Tr{i}"] = df
    # 生成6个测试集
    for i in range(6):
        np.random.seed(10+i)
        df = pd.DataFrame({
            "customer_id": range(800),
            "label": np.random.choice([0,1], size=800, p=[0.83, 0.17]),
            "score": np.random.uniform(0, 100, size=800)
        })
        data_dict[f"Te{i}"] = df
    return data_dict

# ===================== 2. 数据预处理工具（修复报错） =====================
def preprocess_data(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
    preprocessed = {}
    for ds_name, df in data_dict.items():
        # 按打分降序排序
        df_sorted = df.sort_values("score", ascending=False).reset_index(drop=True)
        total_cust = len(df_sorted)
        total_bad = df_sorted["label"].sum()
        preprocessed[ds_name] = {
            "data": df_sorted,  # 这里的"data"是排序后的DataFrame
            "total_cust": total_cust,
            "total_bad": total_bad,
            "scores": df_sorted["score"].values,
            "labels": df_sorted["label"].values
        }
    # ========== 修复点：从preprocessed中取"data"，而不是data_dict ==========
    tr_dfs = [preprocessed[k]["data"] for k in preprocessed if k.startswith("Tr")]
    te_dfs = [preprocessed[k]["data"] for k in preprocessed if k.startswith("Te")]
    # ==================================================================
    preprocessed["TotalTr"] = {"data": pd.concat(tr_dfs, ignore_index=True)}
    preprocessed["TotalTe"] = {"data": pd.concat(te_dfs, ignore_index=True)}
    return preprocessed

# ===================== 3. 割平面法核心求解器 =====================
def customer_segment_cutting_plane(preprocessed_data: Dict, a: int, b: int, C: int) -> Dict:
    
    # optimal_result = customer_segment_cutting_plane(preprocessed_data, a, b, C)
  
    # preprocessed_data['Tr0']
    # dict_keys(['data', 'total_cust', 'total_bad', 'scores', 'labels'])
    tr_ds = [k for k in preprocessed_data if k.startswith("Tr")]
    te_ds = [k for k in preprocessed_data if k.startswith("Te")]
    all_ds = tr_ds + te_ds
    total_ds = ["TotalTr", "TotalTe"]
    all_ds_full = all_ds + total_ds

    model = pl.LpProblem("Customer_Segment_Opt_Cutting_Plane", pl.LpMaximize)

    # 决策变量
    max_possible_K = 20
    K = pl.LpVariable("Group_Number", lowBound=1, upBound=max_possible_K, cat=pl.LpInteger)

    ratio = pl.LpVariable.dicts(
        "Ratio", (all_ds_full, range(1, max_possible_K+1)),
        lowBound=0, upBound=100, cat=pl.LpContinuous
    )

    bad_rate = pl.LpVariable.dicts(
        "BadRate", (all_ds_full, range(1, max_possible_K+1)),
        lowBound=0, upBound=100, cat=pl.LpContinuous
    )

    te_avg_bad_rate = pl.LpVariable.dicts(
        "TeAvgBadRate", range(1, max_possible_K+1),
        lowBound=0, upBound=100, cat=pl.LpContinuous
    )

    split_idx = pl.LpVariable.dicts(
        "Split_Index", range(0, max_possible_K+1),
        lowBound=0, upBound=1000, cat=pl.LpInteger
    )

    # 目标函数
    model += K, "Maximize_Group_Number"

    # 约束1：单分组占比 [a%, b%]
    for ds in all_ds:
        for g in range(1, max_possible_K+1):
            model += ratio[ds][g] >= a, f"Ratio_Lower_{ds}_{g}"
            model += ratio[ds][g] <= b, f"Ratio_Upper_{ds}_{g}"

    # 约束2：相邻分组占比和 ≤ C%
    for ds in all_ds:
        for g in range(1, max_possible_K):
            model += ratio[ds][g] + ratio[ds][g+1] <= C, f"Adjacent_Ratio_{ds}_{g}"

    # 约束3：总训练/测试集坏占比单调性
    for ds in total_ds:
        for g in range(1, max_possible_K):
            model += bad_rate[ds][g] >= bad_rate[ds][g+1] - 100*(1 - pl.LpAffineExpression([(K, 1)], -g)), \
                f"Monotonic_BadRate_{ds}_{g}"

    # 约束4：测试集平均坏占比单调性
    for g in range(1, max_possible_K+1):
        model += te_avg_bad_rate[g] == pl.lpSum(bad_rate[ds][g] for ds in te_ds) / len(te_ds), \
            f"Te_Avg_BadRate_{g}"
    for g in range(1, max_possible_K):
        model += te_avg_bad_rate[g] >= te_avg_bad_rate[g+1], f"Monotonic_TeAvg_{g}"

    # 约束5：各数据集占比之和=100%
    for ds in all_ds_full:
        model += pl.lpSum(ratio[ds][g] for g in range(1, max_possible_K+1)) == 100, f"Ratio_Sum_{ds}"

    # 约束6：分数阈值严格降序（整数变量差值≥1）
    for i in range(max_possible_K):
        model += split_idx[i] - split_idx[i+1] >= 1, f"Threshold_Order_{i}"

    # 约束7：未使用的分组占比为0
    for ds in all_ds_full:
        for g in range(1, max_possible_K+1):
            model += ratio[ds][g] <= 100 * pl.LpAffineExpression([(K,1)], -g+1), f"Valid_Group_{ds}_{g}"

    # 求解
    solver = pl.PULP_CBC_CMD(msg=0, timeLimit=3600)
    model.solve(solver)
    # ✅ 调试打印：查看单个变量的数字结果
    print("🔍 调试 - Tr0数据集分组1的占比：", pl.value(ratio["Tr0"][1]))
    print("🔍 调试 - TotalTr数据集分组2的坏占比：", pl.value(bad_rate["TotalTr"][2]))
    print("🔍 调试 - 测试集平均分组1坏占比：", pl.value(te_avg_bad_rate[1]))

    if pl.LpStatus[model.status] != "Optimal":
        raise RuntimeError("割平面法求解失败，未找到满足所有约束的最优解！")
    
    optimal_K = int(pl.value(K))
    print(f"✅ 割平面法求解完成，最优分组数 = {optimal_K}")

    result = {
        "optimal_K": optimal_K,
        "a": a, "b": b, "C": C,
        "ratio": {ds: {g: pl.value(ratio[ds][g]) for g in range(1, optimal_K+1)} for ds in all_ds_full},
        "bad_rate": {ds: {g: pl.value(bad_rate[ds][g]) for g in range(1, optimal_K+1)} for ds in all_ds_full},
        "te_avg_bad_rate": {g: pl.value(te_avg_bad_rate[g]) for g in range(1, optimal_K+1)},
        "split_idx": [pl.value(split_idx[i]) for i in range(optimal_K+1)]
    }
    return result

# ===================== 4. 结果验证与可视化 =====================
def verify_constraints(result: Dict, preprocessed_data: Dict) -> bool:
    K = result["optimal_K"]
    a, b, C = result["a"], result["b"], result["C"]
    tr_ds = [k for k in preprocessed_data if k.startswith("Tr")]
    te_ds = [k for k in preprocessed_data if k.startswith("Te")]
    all_ds = tr_ds + te_ds
    total_ds = ["TotalTr", "TotalTe"]
    is_valid = True

    print("\n" + "="*80 + "【约束校验报告】" + "="*80)
    # 校验约束1
    print(f"\n1. 单分组占比约束校验 [≥{a}%，≤{b}%]")
    for ds in all_ds:
        for g in range(1, K+1):
            r = result["ratio"][ds][g]
            if not (a - 1e-5 <= r <= b + 1e-5):
                print(f"❌ 数据集{ds}分组{g}占比{r:.2f}%，违反占比约束！")
                is_valid = False
    print("✅ 约束1 校验通过" if is_valid else "❌ 约束1 校验失败")

    # 校验约束2
    print(f"\n2. 相邻分组占比和约束校验 [≤{C}%]")
    for ds in all_ds:
        for g in range(1, K):
            r1 = result["ratio"][ds][g]
            r2 = result["ratio"][ds][g+1]
            if r1 + r2 > C + 1e-5:
                print(f"❌ 数据集{ds}分组{g}+{g+1}占比和{r1+r2:.2f}%，违反约束！")
                is_valid = False
    print("✅ 约束2 校验通过" if is_valid else "❌ 约束2 校验失败")

    # 校验约束3
    print(f"\n3. 总训练/测试集坏占比单调性校验（非递增）")
    for ds in total_ds:
        for g in range(1, K):
            br1 = result["bad_rate"][ds][g]
            br2 = result["bad_rate"][ds][g+1]
            if br1 < br2 - 1e-5:
                print(f"❌ {ds}分组{g}坏占比{br1:.2f}% < 分组{g+1}{br2:.2f}%，违反单调性！")
                is_valid = False
    print("✅ 约束3 校验通过" if is_valid else "❌ 约束3 校验失败")

    # 校验约束4
    print(f"\n4. 测试集平均坏占比单调性校验（非递增）")
    for g in range(1, K):
        br1 = result["te_avg_bad_rate"][g]
        br2 = result["te_avg_bad_rate"][g+1]
        if br1 < br2 - 1e-5:
            print(f"❌ 分组{g}平均坏占比{br1:.2f}% < 分组{g+1}{br2:.2f}%，违反单调性！")
            is_valid = False
    print("✅ 约束4 校验通过" if is_valid else "❌ 约束4 校验失败")
    return is_valid

def print_final_result(result: Dict):
    K = result["optimal_K"]
    print("\n" + "="*80 + "【最终最优分组结果】" + "="*80)
    print(f"📌 业务约束参数：单分组占比[{result['a']}%,{result['b']}%] | 相邻占比和≤{result['C']}%")
    print(f"📌 最优分组数量：{K} 组（分数从高到低分为G1-G{K}）")
    
    print(f"\n📊 各分组核心统计（节选）：")
    print("👉 总训练集(TotalTr)：")
    for g in range(1, K+1):
        r = result["ratio"]["TotalTr"][g]
        br = result["bad_rate"]["TotalTr"][g]
        print(f"   分组{g}：客户占比{r:.2f}% | 坏样本占比{br:.2f}%")
    
    print("\n👉 6个测试集平均：")
    for g in range(1, K+1):
        br = result["te_avg_bad_rate"][g]
        print(f"   分组{g}：平均坏样本占比{br:.2f}%")

# ===================== 5. 主执行流程 =====================
if __name__ == "__main__":
    data_dict = generate_simulate_data()
    preprocessed_data = preprocess_data(data_dict)
    optimal_result = customer_segment_cutting_plane(preprocessed_data, a, b, C)
    is_all_constraint_satisfied = verify_constraints(optimal_result, preprocessed_data)
    if is_all_constraint_satisfied:
        print_final_result(optimal_result)
    else:
        print("\n❌ 最优解未满足所有约束，请调整参数后重试！")