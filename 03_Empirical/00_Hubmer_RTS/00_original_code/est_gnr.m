%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   This script performs the GNR nonparametric tfp estimation
%
%   Created April 11, 2019
%
%   Tasks: Read in data, run step 1 and 2 of estimation, export results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   The code expects a comma delimited csv file with the following variables
%   id (int firm identifier)
%   year (int year)
%   ind (int industry indicator)
%   s (log nominal materials expenditure share of revenue)
%   ls (one period lag of s)
%   r, k, m, l, w (logs of revenue, capital, material expenditure, labor, wages)
%   lr, lk, lm, ll, lw (lags of above)
%   wages are not needed but can be used as an IV in step 2
%   Additionally, the data should be cleaned such that there are no missing observations
%   though this could be handled with the selector vectors below.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LOAD AND CLEAN DATA; SET UP THE ENVIRONMENT 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
global dataloc

%  Set up directory 
dataloc = "D:\projectfolder\tfpdata\";
logloc = "D:\projectfolder\logfiles\";
workloc = "D:\projectfolder\matlab\"; 
inputfilename = "firmdata_tfp.csv";
tfpname = "tfp";
outputfilename = [dataloc + tfpname + '_' + date + '.csv'];
display(['Output file will be '+outputfilename]);

%   Set up logging
logfilename = [logloc+tfpname+date+'_analysis.log'];
system(['del '+logfilename]);
diary(char(logfilename))

%   Read in the data. If the file doesn't exist, return to calling function
if exist([dataloc+inputfilename]) ~= 2
    display([dataloc+inputfilename+' does not exist']);
    return
end

D = readtable([dataloc+inputfilename]);
ND = size(D.year, 1);

%   Read in best guess structure
guessfilename = [workloc + 'guessfile'];
%load(guessfilename)

%   Generate the output table
O = D(:,{'id','year'});

%   State number of terms in polynomials
p1size = 10;    %   Stage 1 elasticity
p2size = 5;     %   Stage 2 constant of integration
h2size = 4;     %   Stage 2 productivity markov process

%   Minimum number of obs to include an industry
indmin = 500;
%   First create industry dummies
dumind = dummyvar(D.ind);
%   For each firm, how many obs in their own industry?
indcount = sum(sum(dumind, 1,'omitnan').*dumind,2);

%   Also create year dummies
yearmin = 500;
yearidx = D.year - min(D.year)+1;
dumyear = dummyvar(yearidx);
yrcount = sum(sum(dumyear, 1,'omitnan').*dumyear,2);

%   We can use this code to trim the data. Probably better to trim based on
%   ratios than levels (so we should change the below code if we want to
%   trim). With mincut = 1 and maxcut = 100 it is only trimming on the m/r
%   ratio and the min industry size.
mincut = 1;
maxcut = 100;
rselector = logical([D.r >= prctile(D.r,mincut)].*[D.r <= prctile(D.r,maxcut)]);
kselector = logical([D.k >= prctile(D.k,mincut)].*[D.k <= prctile(D.k,maxcut)]);
lselector = logical([D.l >= prctile(D.l,mincut)].*[D.l <= prctile(D.l,maxcut)]);
mselector = logical([D.m >= prctile(D.m,mincut)].*[D.m <= prctile(D.m,maxcut)]);
sselector = logical([D.s >= prctile(D.s,mincut)].*[D.s <= prctile(D.s,maxcut)]);
sselectortwo = logical([exp(D.s) >= 0.05].*[exp(D.s) <= 0.95]); % This cuts firms where m/r is over 95% or less than 5%
indselector = logical([D.ind > 0].*[indcount >= indmin]); % This cuts observations from small industries
yrselector = logical([yrcount >= yearmin]);
%lwselector = ~isnan(D.lw);
trimsel = logical(rselector.*kselector.*lselector.*mselector.*indselector.*sselectortwo.*yrselector);

%   Now adjust the industry dummy matrix to remove excluded industries
indcolsel = logical(sum(dumind(trimsel,:),1,'omitnan') >= indmin);
dumind = dumind(:,indcolsel);
numinds = sum(indcolsel);

yrcolsel = logical(sum(dumyear(trimsel,:),1,'omitnan') > 500);
dumyear = dumyear(:,yrcolsel);
numyrs = sum(yrcolsel);

%   Have to drop one column of year dummies to achieve full rank
dumyear = dumyear(:,2:end);

%   Actually let's drop the year dummies altogether and instead drop one
%   industry dummy
dumyear = ones(ND,1);
dumind = dumind(:,2:end);
dumnum = size(dumind,2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% STEP 1 OF GNR ESTIMATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%   Create 2nd degree polynomial for step 1
P1 = [D.k, D.m, D.l, D.k.^2, D.m.^2, D.l.^2, D.k.*D.m, D.k.*D.l, D.m.*D.l];
LP1 = [D.lk, D.lm, D.ll, D.lk.^2, D.lm.^2, D.ll.^2, D.lk.*D.lm, D.lk.*D.ll, D.lm.*D.ll];
N = size(P1, 1);
polyterms = size(P1,2);
P1 = [ones(N,1), P1];
LP1 = [ones(N,1), LP1];

%   Initialize parameters
gamma0 = ones(1,size(P1,2)).*0.1;
gamma0(1) = 1;

%   Also generate initial guess based on ols regression ala GNR (though
%   their code does this with log share, which is not correct).
gamma_ols = (P1(trimsel,:)'*P1(trimsel,:))\P1(trimsel,:)'*exp(D.s(trimsel));
gamma_ols = gamma_ols';
%   Rescale intercept so that initial guess is feasible
bleh = P1(trimsel,:)*gamma_ols' - gamma_ols(1);
gamma_ols(1) = 0.1 - min(bleh);


%   Choose vector of starting guesses based on x0
rng(2020);
tryvec = [rand(99,7).*0.05, (rand(99,3)-0.5).*0.005];
sobolvec1 = [gensobol(6000, 1, -2, 2), gensobol(6000, p1size-4, -0.5, 0.5), gensobol(6000, 3, -0.1, 0.1)];

%   Let's make sure the starting vector satisfies the constraints of the
%   problem (P1*params' > 0)
paramset = [gamma_ols; gamma0; gamma0.*2; sobolvec1(1:100,:)];
%   Make sure 
paramset = nls_const_filter(paramset, P1(trimsel,:));
disp("The number of valid starting guesses: "+size(paramset,1))
tpoints = CustomStartPointSet(paramset);

%%%%%%%%%%%%%%%%%%%   Set smooth solver options %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Seems like the sqp algorithm finds the best minimums here, both using
%   leastsquares_sum and gmm as the objective function value.
fminsearch_opts = optimset('Display', 'iter-detailed', 'MaxFunEvals', 10000, 'MaxIter', 10000);
fminsearch_glob_opts = optimset('Display', 'none');
fmincon_opts = optimoptions('fmincon', 'SpecifyObjectiveGradient',true, 'CheckGradient', true, 'SpecifyConstraintGradient', false, 'Display', 'iter-detailed', 'Algorithm', 'sqp');
fmincon_glob_opts = optimoptions('fmincon', 'SpecifyObjectiveGradient',true, 'MaxFunctionEvaluations', 3000, 'MaxIterations', 1000, 'CheckGradient', false, 'SpecifyConstraintGradient', false, 'Display', 'none', 'Algorithm', 'sqp');
fminunc_glob_opts = optimoptions('fminunc', 'SpecifyObjectiveGradient',true, 'CheckGradient', false, 'Display', 'none', 'Algorithm', 'trust-region', 'StepTolerance', 1.0e-8);

%   Define objective function
step1_func = @(x) nls_obj(x, [D.s(trimsel), P1(trimsel,:)], 'leastsquares_sum');
step1_cons = @(x) nls_obj_const(x, P1(trimsel,:));

%   Set global solver options
%parpool(25)
glob_problem = createOptimProblem('fmincon', 'objective', step1_func, 'x0', gamma0, 'options', fmincon_glob_opts,'lb', -gamma0.*100, 'ub', gamma0.*100);
glob_problem_unc = createOptimProblem('fminunc', 'objective', step1_func, 'x0', gamma0, 'options', fminunc_glob_opts);

%glob_problem = createOptimProblem('fmincon', 'objective', step1_func, 'x0', gamma0, 'options', fmincon_glob_opts,  'lb', -gamma0, 'ub', gamma0.*100);
%glob_problem = createOptimProblem('fminsearch', 'objective', step1_func, 'x0', gamma0, 'options', fminsearch_glob_opts, 'lb', -gamma0, 'ub', gamma0.*100);
ms = MultiStart('UseParallel', true, 'Display', 'iter');

%   Solve using multistart at the vector of starting guesses
tic
[ms_soln, ms_fval, ms_flag, ms_output, ms_allmins] = run(ms, glob_problem, tpoints);
toc
save([tfpname+'_'+date+'_ms_allmins'], 'ms_allmins');

ms_eps = log(P1(trimsel,:)*ms_soln') - D.s(trimsel);
mean(ms_eps)
ms_mdl = fitlm(ms_eps + D.s(trimsel),D.s(trimsel))

%   Now start fminsearch from the best solution to verify that we are at a
%   local minima.
[soln, fval, eflag, output] = fminsearch(step1_func, ms_soln, fminsearch_opts);
O.eps(trimsel) = log(P1(trimsel,:)*soln') - D.s(trimsel);
mean(O.eps(trimsel))
mdl = fitlm(O.eps(trimsel) + D.s(trimsel),D.s(trimsel))

%   Now construct bigeps and the true parameters
%   First, reshift epsilon so it is mean zero (usually close anyways)
bigeps = mean(exp(O.eps(trimsel)));
bigeps

%   Calculate true parameters (gamma) and the materials elasticity
gamma = soln./bigeps;
O.melast(trimsel) = P1(trimsel,:)*gamma';

%   Now integrate up
intvec = [1, 1, 1/2, 1, 1, 1/3, 1, 1/2, 1, 1/2];
O.int_melast(trimsel) = (P1(trimsel,:)*(gamma.*intvec)').*D.m(trimsel);

%   Now y-tilde and lagged y-tilde
%   Note that while the minimization code finds soln such that
%   P1*soln' is always positive, it doesn't ensure that LP1*soln' is always
%   positive. 
negsel = LP1*soln' > 0;
sel = and(trimsel, negsel);

%   This ensures that log wage and lag log wage is not missing
lwsel = ~isnan(D.lw);
%wsel = ~isnan(D.w);
sel = and(sel,lwsel);

%   Construct y-tilde
rt = D.r(sel) - O.eps(sel) - O.int_melast(sel);

%   Lagged y-tilde
leps = log(LP1(sel,:)*soln') - D.ls(sel);
O.leps(sel) = log(LP1(sel,:)*soln') - D.ls(sel);
%leps = leps - epsmean;
lmelast = LP1(sel,:)*gamma';
O.lmelast(sel) = LP1(sel,:)*gamma';
int_lmelast = (LP1(sel,:)*(gamma.*intvec)').*D.lm(sel);
lrt = D.lr(sel) - leps - int_lmelast;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% STEP 2 OF GNR ESTIMATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%   First the polynomials that approximate the constant of integration
P2 = [D.k, D.l, D.k.^2, D.l.^2, D.k.*D.l];
LP2 = [D.lk, D.ll, D.lk.^2, D.ll.^2, D.lk.*D.ll];

%   IV Variables for step 2 (capital for itself and and lagged labor for labor)
IV = [D.k, D.k.^2, D.ll, D.ll.^2, D.k.*D.ll];

%   Initialize alpha and delta parameters
alpha0 = ones(1,size(P2,2)).*0.0001;
delta0 = ones(1,h2size).*0.0001;
delta0(2) = 0.9;
dumind0 = ones(1,dumnum).*0.0001;

%   Initialize the first 5 parameters for the (OLS) nested estimation
X = [ones(size(P2(sel,:),1),1), -P2(sel,:)];
linguess = (X'*X)\X'*rt;
linguess = linguess(2:end)';


%   Set up solver
%parpool(20)
rng(1982);
tryvec2 = rand(99,p2size+h2size+dumnum).*2-1;
sobolvec2 = gensobol(500, p2size+h2size+dumnum, -1, 1);
initvec = [[linguess,dumind0,delta0]; [alpha0,dumind0,delta0]; sobolvec2(1:100,:)];

tpoints2_nested = CustomStartPointSet(initvec(:,1:5));

fminunc_glob_opts_nograd = optimoptions('fminunc', 'SpecifyObjectiveGradient',false, 'Display', 'none');
fminsearch_opts = optimset('Display', 'iter-detailed','MaxFunEvals', 10000, 'MaxIter', 5000);


%   Determine whether we use the nested estimator or not
nested = true;
%   Set this to true if we have more moments than parameters. 
%   We have 5 second stage parameters to estimate (as the industry
%   intercepts and the productivity polynomial are calculated within the
%   nest) so this should be false unless we add moments.
gmm2s = false;

%   Turn off warnings during optimization
warning('off','all')

if nested
    disp("Running the nested estimator")
    W = eye(size(IV,2));
    step2_func_nest = @(x) step2_obj(x, rt, lrt, P2(sel,:), LP2(sel,:), IV(sel,:), [dumind(sel,:)], h2size, W, 'nested');
    glob_problem2_nested = createOptimProblem('fminunc', 'objective', step2_func_nest, 'x0', linguess, 'options', fminunc_glob_opts_nograd);
    ms2 = MultiStart('UseParallel', true, 'Display', 'iter');
    tic
    [ms2_soln, ms2_fval, ms2_flag, ms2_output, ms2_allmins] = run(ms2, glob_problem2_nested, tpoints2_nested);
    toc
    save([tfpname+'_'+date+'_ms2_allmins'], 'ms2_allmins');
    [ms2_soln2, fval, eflag, output] = fminsearch(step2_func_nest, ms2_soln, fminsearch_opts);
    %   Now the second stage if required
    if gmm2s
        % First, calculate the weighting matrix
        W = calc_wmat(ms2_soln2, rt, lrt, P2(sel,:), LP2(sel,:), IV(sel,:), [dumind(sel,:)], h2size, 'nested');
        step3_func_nest = @(x) step2_obj(x, rt, lrt, P2(sel,:), LP2(sel,:), IV(sel,:), [dumind(sel,:)], h2size, W, 'nested');
        glob_problem3_nested = createOptimProblem('fminunc', 'objective', step3_func_nest, 'x0', ms2_soln2, 'options', fminunc_glob_opts_nograd);
        ms3 = MultiStart('UseParallel', true, 'Display', 'iter');
        tpoints3_nested = CustomStartPointSet(ms2_soln2);
        tic
        [ms3_soln, ms3_fval, ms3_flag, ms3_output, ms3_allmins] = run(ms3, glob_problem3_nested, tpoints3_nested);
        toc
        [par2_nest, fval, eflag, output] = fminsearch(step3_func_nest, ms3_soln, fminsearch_opts);
    else
        par2_nest = ms2_soln2;
    end
    
    %   Recover polynomial values given alpha
    [bleh,par2] = step2_func_nest(par2_nest);
else
    disp("Running the non-nested estimator...")
    tic
    [ms2_soln, ms2_fval, ms2_flag, ms2_output, ms2_allmins] = run(ms2, glob_problem2, tpoints2);
    toc
    save([tfpname+'_'+date+'_ms2_allmins'], 'ms2_allmins');
    [par2, fval, eflag, output] = fminsearch(step2_func, ms2_soln, fminsearch_opts);
end

%   Turn warnings back on 
warning('off','all')

%   Add results to our bestguess structure.
%bestguess.alphadelta = rowupdate(bestguess.alphadelta,par2,'add');

O.lomega(sel) = lrt + LP2(sel,:)*par2(1:p2size)';
O.omega(sel) = rt + P2(sel,:)*par2(1:p2size)';
O.nu(sel) = O.omega(sel) + O.eps(sel);
if h2size == 3
    O.eta(sel) = O.omega(sel) - ([dumind(sel,:), O.lomega(sel).^0, O.lomega(sel), O.lomega(sel).^2]*par2(p2size+1:end)');
end
if h2size == 4
    O.eta(sel) = O.omega(sel) - ([dumind(sel,:),O.lomega(sel).^0, O.lomega(sel), O.lomega(sel).^2, O.lomega(sel).^3]*par2(p2size+1:end)');
end
O.nu(sel) = O.omega(sel) + O.eps(sel);
gmm_mdl = fitlm(rt - O.eta(sel), rt);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CALCULATE OUTPUT ELASTICITIES AND RTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Now calculate the output elasticities of capital and labor
k1sel = [0,1,0,0,2,0,0,1/2,1,0]';
k2sel = [1,0,2,0,1]';
O.kelast(sel) = (D.m(sel).*(P1(sel,:)*(k1sel.*gamma')))./D.k(sel) - (P2(sel,:)*(k2sel.*par2(1:p2size)'))./D.k(sel);

l1sel = [0,0,0,1,0,0,2,0,1,1/2]';
l2sel = [0,1,0,2,1]';
O.lelast(sel) = (D.m(sel).*(P1(sel,:)*(l1sel.*gamma')))./D.l(sel) - (P2(sel,:)*(l2sel.*par2(1:p2size)'))./D.l(sel);

% Calculate returns to scale, MRPL and the wedge.
O.rtscale(sel) = O.melast(sel) + O.kelast(sel) + O.lelast(sel);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SAVE RESULTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Save output table
save([dataloc + tfpname+'_'+date+'_O'], 'O');

%   Create a file with our best parameter guesses for use in the future
%   After the first run we can uncomment the proper updater commands above.
bestguess.gamma = soln;
bestguess.alphadelta = par2;
save(guessfilename,'bestguess')
writetable(O,[outputfilename]);

% Print Out Results
disp("gamma: "+num2str(soln))
disp("alphadelta: "+num2str(par2))
disp("elast (k l m): "+num2str(mean(O.kelast(sel)))+", "+num2str(mean(O.lelast(sel)))+", "+num2str(mean(O.melast(sel))))
disp("mean and sd of rts: "+num2str(mean(O.rts(sel)))+", "+num2str(std(O.rts(sel))))
disp("mean and sd of md: "+num2str(mean(O.mrplwedge(sel)))+", "+num2str(std(O.mrplwedge(sel))))
% disp("omega correlations: eta, eps, rev, k, l, lag l, m, w, lag w")
% disp(corrmat(1,2:end))
% disp("eta correlations: omega, eps, rev, k, l, lag l, m, w, lag w")
% disp(corrmat(2,2:end))
disp("sd (omega, eta, eps)"+num2str(std(O.omega(sel)))+", "+num2str(std(O.eta(sel)))+", "+num2str(std(O.eps(sel))))


diary off;