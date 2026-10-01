function [ obj_value, jacobian ] = nls_obj( params, data, objtype )

%   nls_obj returns the value and jacobian of the nls objective function
%   Note that because we will use a nonlinear least squares algorith, 
%   no need to actually calculate the sum of squares here.
%
%   Input Parameters
%   - params: the 1xp vector of parameter guesses, where p is the number of
%   elements in the first stage polynomial
%   - data is the data matrix, where s is data(:,1) and P1 is data(:,2:end)
%   - objtype determines the shape/form of the objective function and
%   jacobian (least squares vs. gmm/score etc).

%   Output
%   - obj_value is the Nx1 (or scalar) value of the objective function
%   - jacobian is the Nxp (or 1xp) jacobian of the nls function

%   Show current guess
%disp(params);

%   Jacobian size is N x p, where N are obs and p is number of params. 
[N, p] = size(data(:, 2:end));
jacobian = ones(N, p);

%   Extract vars from data to make math cleaner
s = data(:,1);
poly = data(:, 2:end);

%   Calculate objective value
ppval = poly*params';

eps = log(ppval) - s;
%eps = log(ppval./exp(s));

if objtype == "leastsquares_vector"
    % This option is for use with lsqnonlin and similar routines
    % The objective function is the vector of residuals.
    obj_value = eps;

    %   Calculate data-level jacobian
    if nargout > 1
        jacobian = poly./ppval;
    end
end

if objtype == "leastsquares_sum"
    % This option is for use with standard univariate solvers like fmincon
    % The objective function is the sum (or mean) of squared residuals.
    if min(min(ppval)) > 0
        obj_value = mean(eps.^2);

        %   Calculate data-level jacobian
        if nargout > 1
            jacobian = mean(2.*eps.*(poly./ppval));
        end
    else
        obj_value = NaN;
        jacobian = NaN;
    end
end

if objtype == "leastsquares_deriv"
    % This option is for use with standard univariate solvers like fmincon
    % The objective function is the residual * gradient
    if min(min(ppval)) > 0
        g = mean(eps.*(poly./ppval));
        %g = (1/N)*eps.*(poly./ppval);
        obj_value = (g*g')/2;

        %   Calculate data-level jacobian
        if nargout > 1
            y = (poly./(ppval.^2)) .* (1 - eps);

            jacobian = sum(g'.*(y'*poly))./N;
        end
    else
        obj_value = NaN;
        jacobian = NaN;
    end
end

if objtype == "gmm"
    % This option is for use with standard univariate solvers like fmincon
    % The (gmm) objective function is the residual * moments (polynomial)
    if min(min(ppval)) > 0
        g = sum(eps.*poly,1)/N;
        %obj_value = g*g'/2;
        obj_value = (g*g')/2;

        %   Calculate data-level jacobian
        if nargout > 1
            Y = poly./ppval;
            jacobian = sum((1/N)*g'.*(Y'*poly)');            
        end
    else
        obj_value = NaN;
        jacobian = NaN;
    end
end
