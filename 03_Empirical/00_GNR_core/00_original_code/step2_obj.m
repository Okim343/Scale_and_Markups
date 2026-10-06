function [ obj_value, output2 ] = step2_obj( params, y, ly, C2, LC2, IV, indmat, hpolydegree, W, algorithm)

%   stage2_obj returns the value and jacobian of the 2nd stage gmm objective function
%
%   Input Parameters
%   - a: the pax1 vector of parameter guesses for alpha, where pa is the number of
%   elements in the second stage polynomial.
%   - d: the pdx1 vector of guesses for delta, where pd is the number of
%   elements in the productivity polynomial.
%   - y and ly are y-tilde and lagged y-tilde
%   - C2 and LC2 are the constant of integration polynomial (and lagged)
%   - IV is the vector of instruments to identify alpha (could be LC2
%   again)
%   - If algorithm is "nested" then delta is calculated given a guess of
%   alpha using OLS.
%   - indmat is either a vector of ones, or a matrix of industry dummies.
%   - hpolydegree determines the degree of the omega poly. 2 is a standard
%   AR(1).

%   Output
%   - obj_value is the Nx1 (or scalar) value of the objective function
%   - jacobian is the Nxp (or 1xp) jacobian of the obj function

%   We can either do this directly, calculating delta and alpha jointly
%   Or we can search over alpha, while solving for delta within (which is
%   linear given alpha). The former allows for easy gradients, but is
%   slower.
warning('off','all')
obs = size(y,1);

if algorithm == "nested"
    a = params(1:5);
    
    onesvec = ones(obs,1);
    
    %   Calculate polynomial values given alpha
    cpoly = C2*a';
    lcpoly = LC2*a';
    
    %   Calculate value of (lagged) omega
    lomega = ly + lcpoly;
    
    if hpolydegree == 4
        O2 = [indmat, onesvec, lomega, lomega.^2, lomega.^3];
    end
    if hpolydegree == 3 
        O2 = [indmat, onesvec, lomega, lomega.^2];
    end
    if hpolydegree == 2
        O2 = [indmat, onesvec, lomega];
    end
    
    %   Calculate value of omega
    omega = y + cpoly;
    
    %   Regress omega on lagged omega
    %   fitlm is SLOW because it calculates all sorts of other stats.
    %regresult = fitlm(lomega,omega,'poly2');
    %d = regresult.Coefficients.Estimate';
    
    d = (O2'*O2)\O2'*omega;
    
    opoly = O2*d;

    eta = omega - opoly;

    %   Here IV could be LC2, or maybe moments in k and L2.labor
    %M = [IV];

    g = mean(eta.*IV);
    obj_value = obs*g*W*g'/2;
    
    %   We will not have analytic derivatives for this method, so nargout>1
    %   will produce the full set of alpha and delta parameters.
    if nargout > 1
        output2 = [a,d'];
    end
    
else
    a = params(1:5);
    d = params(6:end);
    onesvec = ones(obs,1);

    %   Calculate polynomial values given alpha
    cpoly = C2*a';
    lcpoly = LC2*a';

    %   Calculate value of (lagged) omega
    lomega = ly + lcpoly;
    
    %   Set lag omega poly and the moments
    if hpolydegree == 4 
        O2 = [onesvec, lomega, lomega.^2, lomega.^3];
        %   Here IV could be LC2, or maybe moments in k and L2.labor
        M = [IV, onesvec, ly, ly.^2, ly.^3];
    end
    if hpolydegree == 3 
        O2 = [onesvec, lomega, lomega.^2];
        %   Here IV could be LC2, or maybe moments in k and L2.labor
        M = [IV, onesvec, ly, ly.^2];
    end
    if hpolydegree == 2
        O2 = [onesvec, lomega];
        %   Here IV could be LC2, or maybe moments in k and L2.labor
        M = [IV, onesvec, ly];
    end

    opoly = O2*d';

    eta = y + cpoly - opoly;

    g = mean(eta.*M);
    obj_value = g*g';

    if nargout > 1
        %   alpha gradient
        if hpolydegree == 4
            ag = C2 - LC2.*(d(2) + d(3)*2*lomega + d(4)*3*lomega.^2);
        end
        if hpolydegree == 3
            ag = C2 - LC2.*(d(2) + d(3)*2*lomega);
        end
        if hpolydegree == 2
            ag = C2 - LC2.*d(2);
        end
        dg = -O2;
        %dg = [-onesvec, -lomega, -(lomega.^2)];
        A = [ag, dg];
        %G = M.*g;
        %   Calculate the jacobian
        output2 = sum(2*(1/obs)*g'.*(A'*M)');
    end
end