function run_est_gnr_orbis(nworkers, datadir, step2seed)
%RUN_EST_GNR_ORBIS  Run Hubmer et al.'s est_gnr.m (ORBIS copy) on one exported cell.
%   datadir: folder with firmdata_tfp.csv (trailing /); step2seed: rng seed before Step 2.
%   They are passed as arguments and set with setenv here, because est_gnr_orbis.m starts
%   with `clear` and reads them with getenv, and the cluster's containerised MATLAB does not
%   inherit the shell environment. The caller adds 00_GNR_core/00_original_code and this
%   folder to the path. The pool gets a job-specific JobStorageLocation so concurrent array
%   tasks on the cluster do not share the default local-profile folder.
setenv('GNR_DATALOC', datadir);
setenv('GNR_STEP2_SEED', num2str(step2seed));
jsl = fullfile(datadir, 'matlab_pool_jobs');
if ~isfolder(jsl), mkdir(jsl); end
c = parcluster('Processes');
c.JobStorageLocation = jsl;
c.NumWorkers = nworkers;
parpool(c, nworkers);
cd(datadir);
est_gnr_orbis
end
