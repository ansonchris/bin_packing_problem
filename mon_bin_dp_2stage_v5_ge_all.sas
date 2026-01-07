/* Mock Data Generation [cite: 306-314] */
%macro gen_data(n, seed, out_ds, prefix);
    data &out_ds.;
        call streaminit(&seed.);
        do i = 1 to &n.;
            s = floor(300 + (850 - 300) * rand("Uniform"));
            pd = 1 / (1 + exp((s - 580) / 50));
            target = rand("Bernoulli", pd);
            dataset_id = "&prefix.";
            output;
        end;
        drop i s;
    run;
%mend;

/* Generate 10 Train and 6 Test datasets [cite: 312-314] */
%macro setup;
    %do i=1 %to 10; %gen_data(2000, &i., tr&i., Tr); %end;
    %do i=1 %to 6;  %gen_data(1000, %eval(&i.+10), ts&i., Ts); %end;
    
    data all_data;
        set tr1-tr10(in=in_tr) ts1-ts6;
        if in_tr then is_train = 1; else is_train = 0;
        /* Assign unique numeric ID for IML processing */
        if substr(dataset_id, 1, 2) = "Tr" then ds_idx = input(substr(dataset_id, 3), 8.);
        else ds_idx = input(substr(dataset_id, 3), 8.) + 10;
    run;
%mend;
%setup;

proc iml;
/* 1. Global Constraints [cite: 10-11] */
a = 0.05; b = 0.25; C = 0.50; eps = 1e-7;
n_train = 10; n_test = 6;

/* Load data from table into matrices */
use all_data; 
    read all var {pd target ds_idx is_train} into raw_data;
close all_data;

/* --- CORE METRIC CALCULATOR  --- */
start get_bin_metrics(low, high, is_last, data, a, b, C, eps, n_tr, n_ts);
    n_ds = n_tr + n_ts;
    counts = j(1, n_ds, 0); bads = j(1, n_ds, 0); props = j(1, n_ds, 0);
    
    do k = 1 to n_ds;
        ds_all = loc(data[,3] = k);
        if is_last then idx = loc(data[,3] = k & data[,1] >= low & data[,1] <= high);
        else            idx = loc(data[,3] = k & data[,1] >= low & data[,1] < high);
        
        if ncol(idx) > 0 then do;
            counts[k] = ncol(idx);
            bads[k] = data[idx, 2][+];
            props[k] = counts[k] / ncol(ds_all);
        end;
    end;

    /* Condition 1: Individual dataset proportion check [cite: 31-33] */
    if any(props < a - eps) | any(props > b + eps) then return(j(1, 10, .));

    total_tr_pop = nrow(loc(data[,4]=1));
    total_ts_pop = nrow(loc(data[,4]=0));
    
    tr_cnt = counts[1:n_tr][+]; tr_bad = bads[1:n_tr][+];
    ts_cnt = counts[n_tr+1:n_ds][+]; ts_bad = bads[n_tr+1:n_ds][+];
    
    tr_br = (tr_cnt > 0) ? tr_bad/tr_cnt : 0;
    ts_br = (ts_cnt > 0) ? ts_bad/ts_cnt : 0;
    
    /* Avg Test Bad Rate [cite: 43] */
    ts_br_list = j(1, n_ts, 0);
    do k = 1 to n_ts;
        if counts[n_tr+k] > 0 then ts_br_list[k] = bads[n_tr+k]/counts[n_tr+k];
    end;
    avg_ts_br = ts_br_list[:];

    /* Return vector of metrics for DP storage */
    return (tr_br || ts_br || avg_ts_br || (tr_cnt/total_tr_pop) || (ts_cnt/total_ts_pop) || props);
finish;

/* --- STAGE 1: DYNAMIC PROGRAMMING  --- */
start solve_stage1(data, a, b, C, eps, n_tr, n_ts);
    pd_col = data[,1];
    call qntl(edges, pd_col, do(0, 1, 1/100)); /* n_micro=100 [cite: 55, 59] */
    edges = unique(edges);
    n_e = ncol(edges) - 1;
    
    /* chains stores: [len, pred_idx, tr_br, ts_br, avg_ts_br, props(1:16)] */
    chains = j(n_e, n_e, .); 
    lengths = j(n_e, n_e, 0);
    
    do j = 1 to n_e;
        do i = 0 to j-1;
            curr = get_bin_metrics(edges[i+1], edges[j+1], (j=n_e), data, a, b, C, eps, n_tr, n_ts);
            if any(curr = .) then continue;
            
            if i = 0 then lengths[i+1, j] = 1;
            else do;
                best_val = -1; best_p = -1;
                do p_start = 1 to i;
                    if lengths[p_start, i] > 0 then do;
                        prev = get_bin_metrics(edges[p_start], edges[i+1], 0, data, a, b, C, eps, n_tr, n_ts);
                        /* Monotonicity & Adj Sum checks [cite: 74-79] */
                        if curr[1] <= prev[1]-eps | curr[2] <= prev[2]-eps | curr[3] <= prev[3]-eps then continue;
                        if any(prev[6:21] + curr[6:21] > C + eps) then continue;
                        
                        if lengths[p_start, i] > best_val then do;
                            best_val = lengths[p_start, i];
                            best_p = p_start;
                        end;
                    end;
                end;
                if best_p > 0 then lengths[i+1, j] = best_val + 1;
            end;
        end;
    end;
    
    /* Backtrack Path [cite: 85-89] */
    max_len = lengths[, n_e][<>];
    if max_len = 0 then return(.);
    curr_j = n_e;
    curr_i = loc(lengths[, n_e] = max_len)[1];
    path = edges[curr_j+1];
    do while (curr_i > 1);
        path = edges[curr_i] || path;
        next_j = curr_i - 1;
        /* Find predecessor that led to this max length */
        curr_i = loc(lengths[, next_j] = (lengths[curr_i, curr_j]-1))[1];
        curr_j = next_j;
    end;
    return (edges[1] || path);
finish;


