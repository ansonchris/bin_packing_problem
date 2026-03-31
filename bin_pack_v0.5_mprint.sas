/* Import and preprocess data */
proc import datafile = "C://Users//wangyan//Desktop//df_all.csv"
	out = all_data
	dbms = csv
	replace;
	getnames = yes;
run;

data all_data;
	set all_data;
	pd = pd * 100000;
run;

/****************************************************************************************/
/* SAS PROC IML - MPRINT SIMULATION (EQUIVALENT TO SAS MACRO OPTIONS MPRINT)            */
/* ENGLISH ONLY - FOR INTERNATIONAL TEAM USE                                            */
/****************************************************************************************/
proc iml;
    /* Global Switch: 1=ENABLE MPRINT, 0=DISABLE MPRINT */
    MPRINT_ENABLE = 1; 

    /* MPRINT Logging Module (SAS Macro-style output) */
    start MPRINT(msg);
        if MPRINT_ENABLE then do;
            print "MPRINT: " + msg;
        end;
    finish;

/* 1. Global Constraints */
a = 0.05; b = 0.25; C = 0.50; eps = 1e-7;
n_train = 10; n_test = 6;

/* Load data from dataset to matrix */
use all_data; 
    read all var {pd target ds_idx is_train} into raw_data;
close all_data;

n_row = nrow(raw_data);
COLNAME ={"pd","target","ds_idx","is_train"};
row_labs = char(1:n_row);
MATTRIB raw_data ROWNAME = row_labs COLNAME = COLNAME;

call MPRINT("Data loaded successfully | Total observations: "+char(n_row)+" | Constraints: a="+char(a)+" b="+char(b)+" C="+char(C));

/* --- CORE BIN METRIC CALCULATION FUNCTION --- */
start get_bin_metrics(low, high, is_last, data, a, b, C, eps, n_tr, n_ts);
    call MPRINT("Calculating bin metrics | Range=["+char(low,6)+","+char(high,6)+"] | Last bin: "+char(is_last));

    n_ds = n_tr + n_ts;
    counts = j(1, n_ds, 0); 
	bads = j(1, n_ds, 0); 
	props = j(1, n_ds, 0);
    
    do k = 1 to n_ds;
        idx_all = loc(data[,3] = k);
		ds_all = data[idx_all, ];

		if is_last then idx = loc(ds_all[,1] >= low & ds_all[,1] <= high);
        else            idx = loc(ds_all[,1] >= low & ds_all[,1] < high);
        
        if ncol(idx) > 0 then do;
            counts[k] = ncol(idx);
            bads[k] = ds_all[idx, 2][+];
            props[k] = counts[k] / ncol(idx_all);
        end;
    end;

    /* Validation 1: Individual dataset proportion check */
	if any(props < a - eps) | any(props > b + eps) then do;
        call MPRINT("FAILED: Bin proportion out of range [a,b] | Range=["+char(low,6)+","+char(high,6)+"]");
        return(j(1, 10, .));
    end;

    total_tr_pop = nrow(loc(data[,4]=1));
    total_ts_pop = nrow(loc(data[,4]=0));
    tr_cnt = counts[1:n_tr][+]; 
	tr_bad = bads[1:n_tr][+];
    ts_cnt = counts[n_tr+1:n_ds][+]; 
	ts_bad = bads[n_tr+1:n_ds][+];
    
	if tr_cnt > 0 then tr_br = tr_bad/tr_cnt; else tr_br = 0;
	if ts_cnt > 0 then ts_br = ts_bad/ts_cnt; else ts_br = 0;

    /* Calculate average test bad rate */
    ts_br_list = j(1, n_ts, 0);
    do k = 1 to n_ts;
        if counts[n_tr+k] > 0 then ts_br_list[k] = bads[n_tr+k]/counts[n_tr+k];
    end;
    avg_ts_br = ts_br_list[:];

    call MPRINT("PASSED: Bin validation | Train BR="+char(tr_br,6)+" Test BR="+char(ts_br,6)+" Avg Test BR="+char(avg_ts_br,6));
    return (tr_br || ts_br || avg_ts_br || (tr_cnt/total_tr_pop) || (ts_cnt/total_ts_pop) || props);
finish;

/* --- STAGE 1: DYNAMIC PROGRAMMING (GLOBAL OPTIMIZATION) --- */
start solve_stage1(data, a, b, C, eps, n_tr, n_ts);
    call MPRINT("===== START STAGE 1: DYNAMIC PROGRAMMING BINNING =====");

    pd_col = data[,1];
    call qntl(edges, pd_col, do(0, 1, 1/100)); /* n_micro=100 */
    edges = unique(edges);
    n_e = ncol(edges) - 1;

    call MPRINT("Micro bins generated | Total edges: "+char(ncol(edges))+" | Micro bins: "+char(n_e));
    
    /* DP chains storage */
    chains = j(n_e, n_e, .); 
    lengths = j(n_e, n_e, 0);
    
    do j = 1 to n_e;
        do i = 0 to j-1;
            curr = get_bin_metrics(edges[i+1], edges[j+1], (j=n_e), data, a, b, C, eps, n_tr, n_ts);
            if any(curr = .) then do;
                call MPRINT("DP SKIP: Invalid bin | i="+char(i)+" j="+char(j));
                continue;
            end;
            
            if i = 0 then do;
                lengths[i+1, j] = 1;
                call MPRINT("DP INIT: Starting chain | i="+char(i)+" j="+char(j)+" Length=1");
            end;
            else do;
                best_val = -1; best_p = -1;
                do p_start = 1 to i;
                    if lengths[p_start, i] > 0 then do;
                        prev = get_bin_metrics(edges[p_start], edges[i+1], 0, data, a, b, C, eps, n_tr, n_ts);
                        
                        /* Monotonicity & Adjacent Sum Validation */
                        if curr[1] <= prev[1]-eps | curr[2] <= prev[2]-eps | curr[3] <= prev[3]-eps then do;
                            call MPRINT("DP SKIP: Monotonicity violation | Predecessor: "+char(p_start)+"¡ú"+char(i)+" Current: "+char(i)+"¡ú"+char(j));
                            continue;
                        end;
                        if any(prev[6:21] + curr[6:21] > C + eps) then do;
                            call MPRINT("DP SKIP: Adjacent sum violation | Predecessor: "+char(p_start)+"¡ú"+char(i)+" Current: "+char(i)+"¡ú"+char(j));
                            continue;
                        end;
                        
                        if lengths[p_start, i] > best_val then do;
                            best_val = lengths[p_start, i];
                            best_p = p_start;
                        end;
                    end;
                end;
                if best_p > 0 then do;
                    lengths[i+1, j] = best_val + 1;
                    call MPRINT("DP UPDATE: Chain extended | i="+char(i)+" j="+char(j)+" Length="+char(lengths[i+1,j]));
                end;
            end;
        end;
    end;
    
    /* Backtrack to find optimal bin edges */
    max_len = lengths[, n_e][<>];
    if max_len = 0 then do;
        call MPRINT("STAGE 1 FAILED: No valid binning path found");
        return(.);
    end;

    call MPRINT("STAGE 1 SUCCESS: Longest path found | Max bins: "+char(max_len));
    
    curr_j = n_e;
    curr_i = loc(lengths[, n_e] = max_len)[1];
    path = edges[curr_j+1];
    do while (curr_i > 1);
        path = edges[curr_i] || path;
        next_j = curr_i - 1;
        curr_i = loc(lengths[, next_j] = (lengths[curr_i, curr_j]-1))[1];
        curr_j = next_j;
    end;

    call MPRINT("STAGE 1 COMPLETE: Final cutoffs: "+rowcat(char(edges[1] || path,6)));
    return (edges[1] || path);
finish;

/* --- STAGE 2: RECURSIVE REPAIR (VIOLATION FIXING) --- */
start repair_recursive(cutoffs, data, a, b, C, eps, n_tr, n_ts) ;
    call MPRINT("===== START STAGE 2: RECURSIVE REPAIR | Current cutoffs: "+rowcat(char(cutoffs,6))+" =====");

    n_bins = ncol(cutoffs) - 1;
    if n_bins < 1 then return(.);
    
    stats = j(n_bins, 5 + (n_tr + n_ts), 0);
    do i = 1 to n_bins;
        m = get_bin_metrics(cutoffs[i], cutoffs[i+1], (i=n_bins), data, a, b, C, eps, n_tr, n_ts);
        if any(m = .) then do;
            call MPRINT("REPAIR FAILED: Invalid bin at position "+char(i));
            return(.);
        end;
        stats[i,] = m;
    end;

    violation_idx = 0;
    do i = 2 to n_bins;
        prev = stats[i-1,]; curr = stats[i,];
        mono_fail = (curr[1] <= prev[1]-eps) | (curr[2] <= prev[2]-eps) | (curr[3] <= prev[3]-eps);
        adj_fail = any(prev[6:21] + curr[6:21] > C + eps);
        if mono_fail | adj_fail then do; 
            violation_idx = i; 
            call MPRINT("VIOLATION FOUND: Bin "+char(violation_idx)+" | Monotonicity: "+char(mono_fail)+" Adj Sum: "+char(adj_fail));
            leave; 
        end;
    end;

    if violation_idx = 0 then do;
        call MPRINT("REPAIR COMPLETE: No violations found - returning valid cutoffs");
        return(cutoffs);
    end;

    /* Repair Strategies: Merge Left (L), Merge Right (R), Merge Left-Left (LL) */
    /* Strategy L: Merge violation bin with left neighbor */
    idx_l = loc(1:ncol(cutoffs) ^= violation_idx);
    call MPRINT("TRY STRATEGY L: Merge bin "+char(violation_idx-1)+" & "+char(violation_idx));
    res_l = repair_recursive(cutoffs[idx_l], data, a, b, C, eps, n_tr, n_ts);
    
    /* Strategy R: Merge violation bin with right neighbor */
    res_r = .;
    if violation_idx < n_bins then do;
        idx_r = loc(1:ncol(cutoffs) ^= (violation_idx + 1));
        call MPRINT("TRY STRATEGY R: Merge bin "+char(violation_idx)+" & "+char(violation_idx+1));
        res_r = repair_recursive(cutoffs[idx_r], data, a, b, C, eps, n_tr, n_ts);
    end;

    /* Strategy LL: Merge two bins before violation */
    res_ll = .;
    if violation_idx >= 3 then do;
        idx_ll = loc(1:ncol(cutoffs) ^= (violation_idx - 1));
        call MPRINT("TRY STRATEGY LL: Merge bin "+char(violation_idx-2)+" & "+char(violation_idx-1));
        res_ll = repair_recursive(cutoffs[idx_ll], data, a, b, C, eps, n_tr, n_ts);
    end;

    /* Select the optimal result (longest valid bin chain) */
    max_len = 0; final_res = .;
    if ncol(res_l) > max_len then do; max_len = ncol(res_l); final_res = res_l; end;
    if ncol(res_r) > max_len then do; max_len = ncol(res_r); final_res = res_r; end;
    if ncol(res_ll) > max_len then do; max_len = ncol(res_ll); final_res = res_ll; end;
    
    call MPRINT("OPTIMAL RESULT SELECTED | Final bin count: "+char(ncol(final_res)-1));
    return(final_res);
finish;

/* --- MAIN EXECUTION --- */
call MPRINT("============ START FULL BINNING PROCESS ============");
stage1_cuts = solve_stage1(raw_data, a, b, C, eps, n_train, n_test);
print "Stage 1 Cutoffs:", stage1_cuts;

final_cuts = repair_recursive(stage1_cuts, raw_data, a, b, C, eps, n_train, n_test);
print "Final Valid Cutoffs:", final_cuts;

call MPRINT("============ BINNING PROCESS COMPLETED ============");

/* Export final bin edges to SAS dataset */
create final_bin_edges from final_cuts; 
append from final_cuts; 
close final_bin_edges;

call MPRINT("Final bin edges exported to dataset: WORK.FINAL_BIN_EDGES");
quit;
