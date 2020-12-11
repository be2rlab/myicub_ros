function [Qm1,Qm2,Qm3,Qm4,Qm5,Qm6,Qm7,Qm8]=volume_divider(Qm,Q,Qa,mas)

% figure(199)
% scatter3(Q(:,1),Q(:,2),Q(:,3),'r.'),grid on, hold on
% xlabel('q1')
% ylabel('q2')
% zlabel('q3')

Q1 = [0 0 0]; Q2 = [0 0 0]; Q3 = [0 0 0]; Q4 = [0 0 0]; Q5 = [0 0 0]; Q6 = [0 0 0];
Q7 = [0 0 0]; Q8 = [0 0 0];
size1 = length(Qm(:,1,1));
size2 = length(Qm(1,:,1));
size3 = length(Qm(1,1,:));
iii = 1;
for s1 = 1:1:size1/2
    for s2 = 1:1:size2/2
        for s3 = 1:1:size3/2
            Qm1(s1,s2,s3) = Qm(s1,s2,s3);
            if Qm(s1,s2,s3) == 1
                for kk=1:length(mas(:,1))
                    if s1 == mas(kk,2) && s2 == mas(kk,3) && s3 == mas(kk,4)
                        ii=mas(kk,1);
                        Q1(iii,1) = Qa(ii,1); Q1(iii,2) = Qa(ii,2); Q1(iii,3) = Qa(ii,3);
                        iii=iii+1;
                    end
                end
            end
        end
    end
end
for s1 = 1:1:size1/2
    for s2 = 1:1:size2/2
        for s3 = size3/2+1:1:size3
            Qm2(s1,s2,s3-size3/2) = Qm(s1,s2,s3);
            if Qm(s1,s2,s3) == 1
                for kk=1:length(mas(:,1))
                    if s1 == mas(kk,2) && s2 == mas(kk,3) && s3 == mas(kk,4)
                        ii=mas(kk,1);
                        Q2(iii,1) = Qa(ii,1); Q2(iii,2) = Qa(ii,2); Q2(iii,3) = Qa(ii,3);
                        iii=iii+1;
                    end
                end
            end
        end
    end
end
for s1 = 1:1:size1/2
    for s2 = size2/2+1:1:size2
        for s3 = size3/2+1:1:size3
            Qm3(s1,s2-size2/2,s3-size3/2) = Qm(s1,s2,s3);
            if Qm(s1,s2,s3) == 1
                for kk=1:length(mas(:,1))
                    if s1 == mas(kk,2) && s2 == mas(kk,3) && s3 == mas(kk,4)
                        ii=mas(kk,1);
                        Q3(iii,1) = Qa(ii,1); Q3(iii,2) = Qa(ii,2); Q3(iii,3) = Qa(ii,3);
                        iii=iii+1;
                    end
                end
            end
        end
    end
end
for s1 = 1:1:size1/2
    for s2 = size2/2+1:1:size2
        for s3 = 1:1:size3/2
            Qm4(s1,s2-size2/2,s3) = Qm(s1,s2,s3);
            if Qm(s1,s2,s3) == 1
                for kk=1:length(mas(:,1))
                    if s1 == mas(kk,2) && s2 == mas(kk,3) && s3 == mas(kk,4)
                        ii=mas(kk,1);
                        Q4(iii,1) = Qa(ii,1); Q4(iii,2) = Qa(ii,2); Q4(iii,3) = Qa(ii,3);
                        iii=iii+1;
                    end
                end
            end
        end
    end
end
for s1 = size1/2+1:1:size1
    for s2 = 1:1:size2/2
        for s3 = 1:1:size3/2
            Qm5(s1-size1/2,s2,s3) = Qm(s1,s2,s3);
            if Qm(s1,s2,s3) == 1
                for kk=1:length(mas(:,1))
                    if s1 == mas(kk,2) && s2 == mas(kk,3) && s3 == mas(kk,4)
                        ii=mas(kk,1);
                        Q5(iii,1) = Qa(ii,1); Q5(iii,2) = Qa(ii,2); Q5(iii,3) = Qa(ii,3);
                        iii=iii+1;
                    end
                end
            end
        end
    end
end
for s1 = size1/2+1:1:size1
    for s2 = 1:1:size2/2
        for s3 = size3/2+1:1:size3
            Qm6(s1-size1/2,s2,s3-size3/2) = Qm(s1,s2,s3); 
            if Qm(s1,s2,s3) == 1
                for kk=1:length(mas(:,1))
                    if s1 == mas(kk,2) && s2 == mas(kk,3) && s3 == mas(kk,4)
                        ii=mas(kk,1);
                        Q6(iii,1) = Qa(ii,1); Q6(iii,2) = Qa(ii,2); Q6(iii,3) = Qa(ii,3);
                        iii=iii+1;
                    end
                end
            end
        end
    end
end
for s1 = size1/2+1:1:size1
    for s2 = size2/2+1:1:size2
        for s3 = size3/2+1:1:size3
            Qm7(s1-size1/2,s2-size2/2,s3-size3/2) = Qm(s1,s2,s3);
            if Qm(s1,s2,s3) == 1
                for kk=1:length(mas(:,1))
                    if s1 == mas(kk,2) && s2 == mas(kk,3) && s3 == mas(kk,4)
                        ii=mas(kk,1);
                        Q7(iii,1) = Qa(ii,1); Q7(iii,2) = Qa(ii,2); Q7(iii,3) = Qa(ii,3);
                        iii=iii+1;
                    end
                end
            end
        end
    end
end
for s1 = size1/2+1:1:size1
    for s2 = size2/2+1:1:size2
        for s3 = 1:1:size3/2
            Qm8(s1-size1/2,s2-size2/2,s3) = Qm(s1,s2,s3);
            if Qm(s1,s2,s3) == 1
                for kk=1:length(mas(:,1))
                    if s1 == mas(kk,2) && s2 == mas(kk,3) && s3 == mas(kk,4)
                        ii=mas(kk,1);
                        Q8(iii,1) = Qa(ii,1); Q8(iii,2) = Qa(ii,2); Q8(iii,3) = Qa(ii,3);
                        iii=iii+1;
                    end
                end
            end
        end
    end
end
% scatter3(Q1(:,1),Q1(:,2),Q1(:,3),'bo')
% scatter3(Q2(:,1),Q2(:,2),Q2(:,3),'bo')
% scatter3(Q3(:,1),Q3(:,2),Q3(:,3),'bo')
% scatter3(Q4(:,1),Q4(:,2),Q4(:,3),'bo')
% scatter3(Q5(:,1),Q5(:,2),Q5(:,3),'bo')
% scatter3(Q6(:,1),Q6(:,2),Q6(:,3),'bo')
% scatter3(Q7(:,1),Q7(:,2),Q7(:,3),'bo')
% scatter3(Q8(:,1),Q8(:,2),Q8(:,3),'bo')
% hold off
% figure(200)
% scatter3(Q1(:,1),Q1(:,2),Q1(:,3),'m.'),grid on, hold on
% scatter3(Q2(:,1),Q2(:,2),Q2(:,3),'m.')
% scatter3(Q3(:,1),Q3(:,2),Q3(:,3),'m.')
% scatter3(Q4(:,1),Q4(:,2),Q4(:,3),'m.')
% scatter3(Q5(:,1),Q5(:,2),Q5(:,3),'m.')
% scatter3(Q6(:,1),Q6(:,2),Q6(:,3),'m.')
% scatter3(Q7(:,1),Q7(:,2),Q7(:,3),'m.')
% scatter3(Q8(:,1),Q8(:,2),Q8(:,3),'m.')
% xlabel('q1')
% ylabel('q2')
% zlabel('q3')
% hold off