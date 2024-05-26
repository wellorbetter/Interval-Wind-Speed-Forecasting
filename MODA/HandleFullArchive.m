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

function [Archive_X_Chopped, Archive_F_Chopped, Archive_mem_ranks_updated, Archive_member_no]=HandleFullArchive(Archive_X, Archive_F, Archive_member_no, Archive_mem_ranks, ArchiveMaxSize)

for i=1:size(Archive_F,1)-ArchiveMaxSize
    index=RouletteWheelSelection(Archive_mem_ranks);
    
    Archive_X=[Archive_X(1:index-1,:) ; Archive_X(index+1:Archive_member_no,:)];
    Archive_F=[Archive_F(1:index-1,:) ; Archive_F(index+1:Archive_member_no,:)];
    Archive_mem_ranks=[Archive_mem_ranks(1:index-1) Archive_mem_ranks(index+1:Archive_member_no)];
    Archive_member_no=Archive_member_no-1;
end

Archive_X_Chopped=Archive_X;
Archive_F_Chopped=Archive_F;
Archive_mem_ranks_updated=Archive_mem_ranks;