
%macro gen_data(n, seed, out_ds);
    data &out_ds.;
        call streaminit(&seed.);
        do i = 1 to &n.;
            s = floor(300 + (850 - 300) * rand("Uniform"));
            pd = 1 / (1 + exp((s - 580) / 50));
            target = rand("Bernoulli", pd);
            output;
        end;
        drop i s;
    run;
%mend;

/* Generate 10 Training (tr1-tr10) and 6 Validation (ts1-ts6) datasets  */
%macro run_setup;
    %do i=1 %to 10; %gen_data(2000, &i, tr&i); %end;
    %do i=1 %to 6;  %gen_data(1000, %eval(&i+10), ts&i); %end;
%mend;
%run_setup;

proc iml;
    /* 1. Configuration and Constraints [cite: 4, 10-11] */
    a_val = 0.05; b_val = 0.25; C_val = 0.50; eps = 1e-7;
    n_tr = 10; n_ts = 6;

    /* Load all data into a list of matrices (simulating train_dfs/test_dfs)  */
    free all_pds all_targets;
    total_tr_pop = 0; total_ts_pop = 0;
    
    do i = 1 to (n_tr + n_ts);
        dsname = (i <= n_tr) ? "tr" + strip(char(i)) : "ts" + strip(char(i - n_tr));
        use (dsname); read all var {pd target}; close (dsname);
        if i <= n_tr then total_tr_pop = total_tr_pop + nrow(pd);
        else total_ts_pop = total_ts_pop + nrow(pd);
        /* Store in lists for module access */
        call listadd(all_dfs_pd, pd);
        call listadd(all_dfs_tgt, target);
    end;

    /* --- CORE METRIC CALCULATOR  --- */
    start get_bin_metrics(low, high, is_last, all_pd, all_tgt, n_tr, n_ts, a, b, tr_pop, ts_pop, eps);
        n_all = n_tr + n_ts;
        ind_props = j(1, n_all, 0); counts = j(1, n_all, 0); bads = j(1, n_all, 0);
        
        do k = 1 to n_all;
            pd = listget(all_pd, k); tgt = listget(all_tgt, k);
            if is_last then mask = loc(pd >= low & pd <= high);
            else            mask = loc(pd >= low & pd < high);
            
            if ncol(mask) > 0 then do;
                c = ncol(mask); b = tgt[mask][+];
                ind_props[k] = c / nrow(pd);
                counts[k] = c; bads[k] = b;
            end;
        end;

        /* Condition 1: Individual dataset proportions [cite: 31-33] */
        if any(ind_props < a - eps) | any(ind_props > b + eps) then return(.);

        /* Aggregate Statistics [cite: 34-43] */
        tr_cnt = counts[1:n_tr][+]; tr_bad = bads[1:n_tr][+];
        ts_cnt = counts[n_tr+1:n_all][+]; ts_bad = bads[n_tr+1:n_all][+];
        
        tr_br = (tr_cnt > 0) ? tr_bad/tr_cnt : 0;
        ts_br = (ts_cnt > 0) ? ts_bad/ts_cnt : 0;
        
        ts_brs = j(1, n_ts, 0);
        do k = 1 to n_ts;
            if counts[n_tr+k] > 0 then ts_brs[k] = bads[n_tr+k]/counts[n_tr+k];
        end;
        
        /* Return combined results [cite: 44-51] */
        return (ind_props || (tr_cnt/tr_pop) || (ts_cnt/ts_pop) || tr_br || ts_br || ts_brs[:]);
    finish;

    /* --- STAGE 1: DP SOLVER  --- */
    start solve_stage1(all_pd, all_tgt, n_tr, n_ts, a, b, C, tr_pop, ts_pop, eps);
        /* 1. Pre-calculate micro-bins [cite: 58-61] */
        free combined_pd;
        do i = 1 to (n_tr + n_ts); combined_pd = combined_pd // listget(all_pd, i); end;
        call qntl(edges, combined_pd, do(0, 1, 1/100)); /* n_micro=100 */
        edges = unique(edges);
        n = ncol(edges) - 1;

        /* chains: [len, pred_idx] [cite: 63-83] */
        chains_len = j(n+1, n+1, -1);
        chains_pred = j(n+1, n+1, 0);
        
        do j = 1 to n;
            do i = 0 to j-1;
                curr = get_bin_metrics(edges[i+1], edges[j+1], (j=n), all_pd, all_tgt, n_tr, n_ts, a, b, tr_pop, ts_pop, eps);
                if any(curr = .) then continue;
                if i = 0 then chains_len[1, j+1] = 1;
                else do;
                    best_l = -1; best_p = 0;
                    do p_k = 1 to i;
                        if chains_len[p_k, i+1] > 0 then do;
                            prev = get_bin_metrics(edges[p_k], edges[i+1], 0, all_pd, all_tgt, n_tr, n_ts, a, b, tr_pop, ts_pop, eps);
                            /* Monotonicity and Adj Sum checks [cite: 74-79] */
                            if curr[19] <= prev[19]-eps | curr[20] <= prev[20]-eps | curr[21] <= prev[21]-eps then continue;
                            if any(prev[1:16] + curr[1:16] > C + eps) then continue;
                            if chains_len[p_k, i+1] > best_l then do; best_l = chains_len[p_k, i+1]; best_p = p_k; end;
                        end;
                    end;
                    if best_p > 0 then do; chains_len[i+1, j+1] = best_l + 1; chains_pred[i+1, j+1] = best_p; end;
                end;
            end;
        end;

        /* Backtracking [cite: 85-89] */
        final_idx = loc(chains_len[, n+1] = max(chains_len[, n+1]));
        if ncol(final_idx) = 0 then return(.);
        curr_i = final_idx[1]; curr_j = n+1;
        path = edges[curr_j];
        do while(curr_i > 0);
            path = edges[curr_i] || path;
            next_i = chains_pred[curr_i, curr_j];
            curr_j = curr_i; curr_i = next_i;
        end;
        return(path);
    finish;

    stage1_cuts = solve_stage1(all_dfs_pd, all_dfs_tgt, n_tr, n_ts, a_val, b_val, C_val, total_tr_pop, total_ts_pop, eps);
    store module=(get_bin_metrics solve_stage1);
    store stage1_cuts all_dfs_pd all_dfs_tgt n_tr n_ts a_val b_val C_val total_tr_pop total_ts_pop eps;
quit;

