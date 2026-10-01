function [ pvec ] = nls_const_filter( paramvec, data )

%   This takes a vector of parameter vectors (so a matrix) and returns the
%   subset which satisfy the problem constraints (data*params' > 0)
%
%   Input Parameters
%   - paramvec: the gxp vector of parameter guesses, where p is the number of
%   elements in the first stage polynomial and g is the number of parameter
%   vector guesses
%   - data is the Nxp data matrix P1 (note here we leave out s)

%   Output
%   - pvec is the subset which satisfy the problem constraints (data*params' > 0)

c = min(data*paramvec'); %c is 1xg
pvec = paramvec(c'>0,:);