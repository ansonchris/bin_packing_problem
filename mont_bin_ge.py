import pandas as pd
import numpy as np

class BinningOptimizer:
    def __init__(self, train_dfs, test_dfs, a=0.05, b=0.25, C=0.50):
        self.train_dfs = train_dfs
        self.test_dfs = test_dfs
        self.all_dfs = train_dfs + test_dfs
        self.n_train, self.n_test = len(train_dfs), len(test_dfs)
        self.a, self.b, self.C = a, b, C

    def _calculate_micro_stats(self, n_bins=60):
        combined = pd.concat(self.all_dfs)[['score', 'target']]
        _, self.bin_edges = pd.qcut(combined['score'], q=n_bins, retbins=True, duplicates='drop')
        self.bin_edges = sorted(list(set(self.bin_edges)))
        self.n_micro_bins = len(self.bin_edges) - 1
        self.micro_stats = np.zeros((self.n_micro_bins, len(self.all_dfs), 2))
        for idx, df in enumerate(self.all_dfs):
            cuts = pd.cut(df['score'], bins=self.bin_edges, include_lowest=True, labels=False)
            stats = df.groupby(cuts, observed=False)['target'].agg(['count', 'sum'])
            for b_idx in stats.index:
                if 0 <= b_idx < self.n_micro_bins:
                    self.micro_stats[b_idx, idx, 0] = stats.loc[b_idx, 'count']
                    self.micro_stats[b_idx, idx, 1] = stats.loc[b_idx, 'sum']

    def _get_metrics(self, start, end):
        g_stats = self.micro_stats[start:end].sum(axis=0)
        counts, bads = g_stats[:, 0], g_stats[:, 1]
        totals = self.micro_stats.sum(axis=0)[:, 0]
        props = counts / totals
        
        # CONDITION 1: 5% <= prop <= 25%
        if np.any(props < self.a) or np.any(props > self.b):
            return None

        tr_c, tr_b = counts[:self.n_train].sum(), bads[:self.n_train].sum()
        ts_c, ts_b = counts[self.n_train:].sum(), bads[self.n_train:].sum()
        test_brs = [bads[i]/counts[i] if counts[i] > 0 else 0 for i in range(self.n_train, len(self.all_dfs))]
        
        return {
            'props': props,
            'tr_br': tr_b/tr_c if tr_c > 0 else 0,
            'ts_br': ts_b/ts_c if ts_c > 0 else 0,
            'avg_ts_br': np.mean(test_brs)
        }

    def solve(self):
        self._calculate_micro_stats(n_bins=80)
        # dp[end_bin] = (max_length, prev_start_bin, stats)
        # We use a dict to store chains ending at a specific group (start, end)
        chains = {} 
        
        for j in range(1, self.n_micro_bins + 1):
            for i in range(j):
                curr = self._get_metrics(i, j)
                if curr is None: continue
                
                if i == 0:
                    chains[(i, j)] = (1, None)
                else:
                    best_prev_len = -1
                    best_prev_key = None
                    for (pk, pi) in [k for k in chains if k[1] == i]:
                        prev = self._get_metrics(pk, pi)
                        # COND 2, 3, 4
                        if np.any(prev['props'] + curr['props'] > self.C): continue
                        if curr['tr_br'] >= prev['tr_br'] or curr['ts_br'] >= prev['ts_br']: continue
                        if curr['avg_ts_br'] >= prev['avg_ts_br']: continue
                        
                        if chains[(pk, pi)][0] > best_prev_len:
                            best_prev_len = chains[(pk, pi)][0]
                            best_prev_key = (pk, pi)
                    
                    if best_prev_key:
                        chains[(i, j)] = (best_prev_len + 1, best_prev_key)

        # Find best chain ending at the last bin
        final_key = max([k for k in chains if k[1] == self.n_micro_bins], 
                        key=lambda x: chains[x][0], default=None)
        
        if not final_key: return []
        path = []
        curr_key = final_key
        while curr_key:
            path.append(curr_key)
            curr_key = chains[curr_key][1]
        return [self.bin_edges[0]] + [self.bin_edges[k[1]] for k in reversed(path)]

    def audit(self, cutoffs):
        print(f"\n{'='*95}\nAUDIT REPORT: a={self.a:.1%}, b={self.b:.1%}, C={self.C:.1%}\n{'='*95}")
        groups = [(cutoffs[i], cutoffs[i+1]) for i in range(len(cutoffs)-1)]
        history = []
        violations = 0
        
        for idx, (low, high) in enumerate(groups):
            print(f"\nGROUP {idx+1}: [{low:.2f}, {high:.2f}]")
            metrics = self._get_metrics_raw(low, high, idx == len(groups)-1)
            
            for s_idx, p in enumerate(metrics['props']):
                lbl = f"Tr {s_idx+1:02}" if s_idx < self.n_train else f"Ts {s_idx-self.n_train+1:02}"
                err = "!!" if p < self.a or p > self.b else ""
                if err: violations += 1
                print(f"  {lbl} | Prop: {p:6.2%} {err}", end=" | ")
                if (s_idx + 1) % 4 == 0: print("")
            
            print(f"\n  Agg Train BR: {metrics['tr_br']:.4f} | Agg Test BR: {metrics['ts_br']:.4f} | Avg Test BR: {metrics['avg_ts_br']:.4f}")
            
            if history:
                prev = history[-1]
                m_tr = metrics['tr_br'] < prev['tr_br']
                m_ts = metrics['ts_br'] < prev['ts_br']
                m_avg = metrics['avg_ts_br'] < prev['avg_ts_br']
                adj_ok = np.all(metrics['props'] + prev['props'] <= self.C)
                
                print(f"  Checks: Mono Train: {m_tr} | Mono Test: {m_ts} | Mono Avg: {m_avg} | Adj Sum OK: {adj_ok}")
                if not all([m_tr, m_ts, m_avg, adj_ok]): violations += 1
            history.append(metrics)
        print(f"\n{'='*95}\nAUDIT COMPLETE. TOTAL VIOLATIONS: {violations}\n{'='*95}")

    def _get_metrics_raw(self, low, high, last):
        props, test_brs = [], []
        tr_b, tr_c, ts_b, ts_c = 0, 0, 0, 0
        for i, df in enumerate(self.all_dfs):
            mask = (df['score'] >= low) & (df['score'] <= high) if last else (df['score'] >= low) & (df['score'] < high)
            c, b = len(df[mask]), df[mask]['target'].sum()
            p = c / len(df)
            props.append(p)
            if i < self.n_train: tr_c += c; tr_b += b
            else: ts_c += c; ts_b += b; test_brs.append(b/c if c>0 else 0)
        return {'props': np.array(props), 'tr_br': tr_b/tr_c, 'ts_br': ts_b/ts_c, 'avg_ts_br': np.mean(test_brs)}

# Setup and Execution
np.random.seed(42)
def gen(n):
    s = np.random.randint(300, 850, n)
    return pd.DataFrame({'score': s, 'target': np.random.binomial(1, 1/(1+np.exp((s-580)/50)))})

opt = BinningOptimizer([gen(1500) for _ in range(10)], [gen(800) for _ in range(6)], a=0.05, b=0.25, C=0.50)
cuts = opt.solve()
if cuts: opt.audit(cuts)


#%%

train_df = [gen(1500) for _ in range(10)]
test_df = [gen(800) for _ in range(6)]

df1 = train_df[0]

#%%






