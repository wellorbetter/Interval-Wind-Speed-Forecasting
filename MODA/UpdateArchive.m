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


function [Archive_X_updated, Archive_F_updated, Archive_member_no]=UpdateArchive(Archive_X, Archive_F, Particles_X, Particles_F, Archive_member_no)
Archive_X_temp=[Archive_X ; Particles_X'];
Archive_F_temp=[Archive_F ; Particles_F];
o=zeros(1,size(Archive_F_temp,1));

for i=1:size(Archive_F_temp,1)
    o(i)=0;
    for j=1:i-1
        if any(Archive_F_temp(i,:) ~= Archive_F_temp(j,:))
            if dominates(Archive_F_temp(i,:),Archive_F_temp(j,:))
                o(j)=1;
            elseif dominates(Archive_F_temp(j,:),Archive_F_temp(i,:))
                o(i)=1;
                break;
            end
        else
            o(j)=1;
            o(i)=1;
        end
    end
end

Archive_member_no=0;
index=0;
for i=1:size(Archive_X_temp,1)
    if o(i)==0
        Archive_member_no=Archive_member_no+1;
        Archive_X_updated(Archive_member_no,:)=Archive_X_temp(i,:);
        Archive_F_updated(Archive_member_no,:)=Archive_F_temp(i,:);
        
    else
        index=index+1;
    end
end

end