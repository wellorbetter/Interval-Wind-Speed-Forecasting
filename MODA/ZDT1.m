%___________________________________________________________________%
%  Multi-Objective Dragonfly Algorithm (MODA) source codes demo     %
%                           version 1.0                             %
%                                                                   %
%  Developed in MATLAB R2011b(7.13)                                 %
%                                                                   %
%  Author and programmer: Seyedali Mirjalili                        %
%                                                                   %
%         e-Mail: ali.mirjalili@gmail.com                           %
%                 seyedali.mirjalili@griffithuni.edu.au             %
%                                                                   %
%       Homepage: http://www.alimirjalili.com                       %
%                                                                   %
%   Main paper:                                                     %
%                                                                   %
%   S. Mirjalili, Dragonfly algorithm: a new meta-heuristic         %
%   optimization technique for solving single-objective, discrete,  %
%   and multi-objective problems, Neural Computing and Applications %
%   DOI: http://dx.doi.org/10.1007/s00521-015-1920-1                %
%___________________________________________________________________%

% Modify this file with respect to your objective function
function o = ZDT1(x, X_train, y_train)  
tmp = py.my_bpnn.my_bpnn(X_train, y_train, x(1), x(2));
%disp(tmp);
%disp(class(tmp{1}))
%disp(class(tmp{2}))
o = [double(tmp{1}), double(tmp{2})];
end



