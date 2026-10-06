function [returnmat] = rowupdate(themat, newguess, action, tol)

%   This function adds or removes a row or column from matrix "themat"
%   The row or column to add is "newguess". action can be 'add' or 'remove'

updated = false;
returnmat = themat;

switch nargin
    case 4
        % All variables have been specified, including matching tolerance
        if (action ~= "add" && action ~= "remove")
            disp(['action "',action,'" is not possible.']);
            disp('No action taken');
            return;
        end
    case 3
        % Assume tol has not been specified
        disp('Using default tolerance (1e-06)');
        tol = 1e-06;
        if (action ~= "add" && action ~= "remove")
            disp(['action "',action,'" is not possible.']);
            disp('No action taken');
            return;
        end
    case 2
        % Assume tol and action have not been specified.
        disp('Using default action (add) and default tolerance (1e-06)');
        action = "add";
        tol = 1e-06;
    otherwise
        disp('Error in rowupdate: Must specify target matrix and row to add/remove');
        disp('No action taken');
        return;
end

if action == "add"
    %if ~max(ismember(themat,newguess, 'rows'))
    if ~max(ismembertol(themat, newguess, tol, 'ByRows', true))
        returnmat = [themat; newguess];
        updated = true;
        disp('bestguesses updated...');
    else
        disp('guess already exists -- not updated.');
    end
end

if action == "remove"
    if max(ismembertol(themat, newguess, tol, 'ByRows', true))
        returnmat = themat(~ismembertol(themat, newguess, tol, 'ByRows', true),:);
        updated = true;
        disp('guess removed...');
    else
        disp('guess not present -- not updated.');
    end
end