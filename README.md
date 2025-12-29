import pulp as pl
import pandas as pd
import numpy as np
from typing import Dict

# ===================== 1. 业务参数配置（可根据需求调整） =====================
a = 0        # 单分组客户占比下限(%)
b = 100       # 单分组客户占比上限(%)
C = 100       # 相邻两个分组占比之和上限(%)
mono_tol = 0.0  # 坏占比单调性容差(0=严格非递增，>0=宽松非递增)
max_possible_K = 10  # 最大分组数上限
seed = 2025  # 随机种子，保证结果可复现

# ===================== 2. 模拟数据生成（分数越高，坏样本越少，贴合业务逻辑） =====================
def generate_simulate_data() -> Dict[str, pd.DataFrame]:
    data_dict = {}
    # 生成10个训练集
    for i in range(10):
        np.random.seed(seed + i)
        df = pd.DataFrame({
            "customer_id": range(1000),
            "score": np.random.uniform(0, 100, size=1000),  # 客户评分
            "label": np.random.choice([0,1], size=1000, p=[0.92 - 0.01*i, 0.08 + 0.01*i])  # 分数越高，坏样本越少
        })
        data_dict[f"Tr{i}"] = df.sort_values("score", ascending=False).reset_index(drop=True)
    
    # 生成6个测试集
    for i in range(6):
        np.random.seed(seed + 10 + i)
        df = pd.DataFrame({
            "customer_id": range(800),
            "score": np.random.uniform(0, 100, size=800),
            "label": np.random.choice([0,1], size=800, p=[0.90 - 0.01*i, 0.10 + 0.01*i])
        })
        data_dict[f"Te{i}"] = df.sort_values("score", ascending=False).reset_index(drop=True)
    return data_dict

# ===================== 3. 数据预处理（提取数据集核心信息） =====================
def preprocess_data(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
    preprocessed = {}
    for ds_name, df in data_dict.items():
        total_cust = len(df)
        preprocessed[ds_name] = {
            "data": df,
            "total_cust": total_cust,
            "total_bad": df["label"].sum(),
            "avg_bad_rate": (df["label"].sum() / total_cust) * 100 if total_cust > 0 else 0.0
        }
    return preprocessed

# ===================== 4. 割平面法核心求解器（含全部约束：占比+单调性） =====================
def customer_segment_cutting_plane(preprocessed_data: Dict, data_dict: Dict) -> Dict:
    # 1. 数据集分类
    tr_ds = [k for k in preprocessed_data if k.startswith("Tr")]  # 训练集列表
    te_ds = [k for k in preprocessed_data if k.startswith("Te")]  # 测试集列表
    all_ds = tr_ds + te_ds
    max_cust = max([preprocessed_data[ds]["total_cust"] for ds in all_ds])  # 全局最大客户数
    M = max_cust * 2  # big-M常数，用于条件约束生效控制

    # 2. 创建线性规划模型（目标：最大化分组数K）
    model = pl.LpProblem("Customer_Segment_Opt", pl.LpMaximize)

    # ===================== 定义所有线性变量（无任何非线性） =====================
    # 变量1：最优分组数（整数，核心优化目标）
    K = pl.LpVariable("Opt_Group_Num", lowBound=2, upBound=max_possible_K, cat=pl.LpInteger)
    
    # 变量2：分组分割索引（split_idx[g] = 第g组的结束客户索引，严格递增）
    split_idx = pl.LpVariable.dicts(
        "Split_Idx", range(0, max_possible_K + 1),
        lowBound=0, upBound=max_cust, cat=pl.LpInteger
    )
    
    # 变量3：各数据集-各分组的客户占比（线性变量，核心占比约束对象）
    ratio = pl.LpVariable.dicts(
        "Group_Ratio", (all_ds, range(1, max_possible_K + 1)),
        lowBound=0.0, upBound=100.0, cat=pl.LpContinuous
    )
    
    # 变量4：各数据集-各分组的坏占比（✅ 独立线性变量，直接约束单调性，无除法/相乘）
    bad_rate = pl.LpVariable.dicts(
        "Group_Bad_Rate", (all_ds, range(1, max_possible_K + 1)),
        lowBound=0.0, upBound=100.0, cat=pl.LpContinuous
    )
    
    # 变量5：测试集平均坏占比（线性变量，约束单调性）
    te_avg_bad_rate = pl.LpVariable.dicts(
        "Te_Avg_Bad_Rate", range(1, max_possible_K + 1),
        lowBound=0.0, upBound=100.0, cat=pl.LpContinuous
    )

    # ===================== 目标函数：最大化分组数K =====================
    model += K, "Maximize_Group_Number"

    # ===================== 约束1：分割索引的严格递增约束（核心） =====================
    model += split_idx[0] == 0, "Split_Idx_Start_0"  # 第0组起始索引=0
    for g in range(max_possible_K):
        model += split_idx[g+1] - split_idx[g] >= 1, f"Split_Idx_Inc_{g}_{g+1}"  # 严格递增（差值≥1）
    # 约束：最后一个分组必须覆盖全部客户
    for g in range(1, max_possible_K + 1):
        model += split_idx[g] >= max_cust - M * (1 - (K == g)), f"Split_Idx_End_{g}"

    # ===================== 约束2：单分组占比约束 [a%, b%] =====================
    for ds in all_ds:
        total_cust = preprocessed_data[ds]["total_cust"]
        for g in range(1, max_possible_K + 1):
            # 分组g的客户数 = 结束索引 - 起始索引
            group_cust = split_idx[g] - split_idx[g-1]
            # 占比 = (分组客户数 / 总客户数) * 100，绑定到ratio变量
            model += ratio[ds][g] == (group_cust / total_cust) * 100, f"Ratio_Bind_{ds}_{g}"
            # 占比上下限约束：仅当 g ≤ K 时生效
            model += ratio[ds][g] >= a - M * (1 - (K >= g)), f"Ratio_Lower_{ds}_{g}"
            model += ratio[ds][g] <= b + M * (1 - (K >= g)), f"Ratio_Upper_{ds}_{g}"

    # ===================== 约束3：相邻分组占比和 ≤ C% =====================
    for ds in all_ds:
        for g in range(1, max_possible_K):
            model += ratio[ds][g] + ratio[ds][g+1] <= C + M * (1 - (K >= g+1)), f"Adj_Ratio_Sum_{ds}_{g}"

    # ===================== 约束4：训练集 分组坏占比 非递增单调性（核心） =====================
    for ds in tr_ds:
        for g in range(1, max_possible_K):
            # ✅ 纯线性约束：第g组坏占比 ≥ 第g+1组坏占比 - 容差
            # 仅当 g+1 ≤ K 时，约束生效
            model += bad_rate[ds][g] >= bad_rate[ds][g+1] - mono_tol - M * (1 - (K >= g+1)), \
                f"Tr_BadRate_Mono_{ds}_{g}"

    # ===================== 约束5：测试集平均坏占比 非递增单调性（核心） =====================
    # 子约束：测试集平均坏占比 = 所有测试集同分组坏占比的均值
    for g in range(1, max_possible_K + 1):
        model += te_avg_bad_rate[g] == pl.lpSum(bad_rate[ds][g] for ds in te_ds) / len(te_ds), \
            f"Te_Avg_BadRate_Calc_{g}"
    # 主约束：平均坏占比 非递增
    for g in range(1, max_possible_K):
        model += te_avg_bad_rate[g] >= te_avg_bad_rate[g+1] - mono_tol - M * (1 - (K >= g+1)), \
            f"Te_Avg_BadRate_Mono_{g}"

    # ===================== 求解模型（割平面法） =====================
    solver = pl.PULP_CBC_CMD(msg=0, timeLimit=3600)  # msg=0关闭日志，msg=1开启调试日志
    solve_status = model.solve(solver)

    # 求解状态校验
    if pl.LpStatus[solve_status] not in ["Optimal", "Feasible"]:
        raise RuntimeError(f"模型求解失败！状态：{pl.LpStatus[solve_status]}，请放宽约束参数(a/b/C)")

    # ===================== 解析求解结果，回填真实数据 =====================
    optimal_K = int(pl.value(K))  # 最优分组数
    optimal_split = [int(pl.value(split_idx[g])) for g in range(optimal_K + 1)]  # 分组分割索引

    # 生成最终结果（含真实客户数、真实坏占比，从原始数据统计）
    final_result = {
        "optimal_K": optimal_K,
        "optimal_split_idx": optimal_split,
        "constraint_params": {"a": a, "b": b, "C": C, "mono_tol": mono_tol},
        "solve_status": pl.LpStatus[solve_status],
        "dataset_details": {}
    }

    # 遍历所有数据集，统计分组真实详情
    for ds_name in all_ds:
        df = preprocessed_data[ds_name]["data"]
        total_cust = preprocessed_data[ds_name]["total_cust"]
        group_detail = []
        
        for g in range(optimal_K):
            start_idx = optimal_split[g]
            end_idx = optimal_split[g+1] if (g+1) < optimal_K else total_cust - 1
            end_idx = min(end_idx, total_cust - 1)  # 边界防护
            
            # 真实数据统计
            group_data = df.iloc[start_idx:end_idx+1]
            cust_num = len(group_data)
            bad_num = group_data["label"].sum()
            real_ratio = (cust_num / total_cust) * 100
            real_bad_rate = (bad_num / cust_num) * 100 if cust_num > 0 else 0.0

            group_detail.append({
                "group_num": g + 1,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "customer_num": cust_num,
                "bad_num": bad_num,
                "customer_ratio(%)": round(real_ratio, 2),
                "bad_rate(%)": round(real_bad_rate, 2)
            })
        final_result["dataset_details"][ds_name] = group_detail

    return final_result

# ===================== 5. 结果打印与约束校验 =====================
def print_final_result(result: Dict):
    print("="*100)
    print("📊 客户分组最优结果（含占比约束+单调性约束）")
    print("="*100)
    print(f"✅ 求解状态：{result['solve_status']}")
    print(f"✅ 最优分组数：{result['optimal_K']} 组")
    print(f"✅ 分组分割索引：{result['optimal_split_idx']}")
    print(f"✅ 业务约束：单分组占比[{result['constraint_params']['a']}%, {result['constraint_params']['b']}%] | 相邻占比和≤{result['constraint_params']['C']}% | 坏占比非递增容差={result['constraint_params']['mono_tol']}%")
    print("="*100)

    # 打印示例数据集结果（Tr0+Te0）
    print("\n👉 示例：训练集Tr0 分组详情")
    for group in result["dataset_details"]["Tr0"]:
        print(f"   分组{group['group_num']} | 占比{group['customer_ratio(%)']}% | 坏占比{group['bad_rate(%)']}%")
    
    print("\n👉 示例：测试集Te0 分组详情")
    for group in result["dataset_details"]["Te0"]:
        print(f"   分组{group['group_num']} | 占比{group['customer_ratio(%)']}% | 坏占比{group['bad_rate(%)']}%")

# ===================== 主执行入口 =====================
if __name__ == "__main__":
    try:
        # 数据生成与预处理
        data_dict = generate_simulate_data()
        preprocessed_data = preprocess_data(data_dict)
        # 核心求解
        optimal_result = customer_segment_cutting_plane(preprocessed_data, data_dict)
        # 结果打印
        print_final_result(optimal_result)
    except RuntimeError as e:
        print(f"\n❌ 执行错误：{str(e)}")
        print("\n💡 调整建议：1.增大b（单分组占比上限）；2.增大C（相邻占比和上限）；3.设置mono_tol>0（放宽单调性）")# bin_packing_problem
