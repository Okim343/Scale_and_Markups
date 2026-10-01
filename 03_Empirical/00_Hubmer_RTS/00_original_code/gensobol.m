function [ sseq ] = gensobol( Nobs, Nparams, lb, ub )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

lbubdiff = ub-lb;
sobolseq = sobolset(Nparams,'Skip',1000,'Leap',100);
sobolseq = scramble(sobolseq,'MatousekAffineOwen');
sseq = net(sobolseq,Nobs);
sseq = sseq.*lbubdiff + lb;

end

