
% gl=rbt.ikcon(win, initial); %Решение ОЗК для цели

%Формирование параллелепипедов для описания запретных зон
% counter = 0;
% [km,obs] = islands_finder(Qa,Qm,counter,mas,1,1,1);


[Qm1,Qm2,Qm3,Qm4,Qm5,Qm6,Qm7,Qm8] = volume_divider(Qm,Q,Qa,mas);
[Qm11,Qm12,Qm13,Qm14,Qm15,Qm16,Qm17,Qm18] = volume_divider(Qm1,Q,Qa,mas);
[Qm21,Qm22,Qm23,Qm24,Qm25,Qm26,Qm27,Qm28] = volume_divider(Qm2,Q,Qa,mas);
[Qm31,Qm32,Qm33,Qm34,Qm35,Qm36,Qm37,Qm38] = volume_divider(Qm3,Q,Qa,mas);
[Qm41,Qm42,Qm43,Qm44,Qm45,Qm46,Qm47,Qm48] = volume_divider(Qm4,Q,Qa,mas);
[Qm51,Qm52,Qm53,Qm54,Qm55,Qm56,Qm57,Qm58] = volume_divider(Qm5,Q,Qa,mas);
[Qm61,Qm62,Qm63,Qm64,Qm65,Qm66,Qm67,Qm68] = volume_divider(Qm6,Q,Qa,mas);
[Qm71,Qm72,Qm73,Qm74,Qm75,Qm76,Qm77,Qm78] = volume_divider(Qm7,Q,Qa,mas);
[Qm81,Qm82,Qm83,Qm84,Qm85,Qm86,Qm87,Qm88] = volume_divider(Qm8,Q,Qa,mas);
% figure, scatter3(Qm1(:,1),Qm1(:,2),Qm1(:,3)),grid on
% figure, scatter3(Qm2(:,1),Qm2(:,2),Qm2(:,3)),grid on
% figure, scatter3(Qm3(:,1),Qm3(:,2),Qm3(:,3)),grid on
% figure, scatter3(Qm4(:,1),Qm4(:,2),Qm4(:,3)),grid on
% figure, scatter3(Qm5(:,1),Qm5(:,2),Qm5(:,3)),grid on
% figure, scatter3(Qm6(:,1),Qm6(:,2),Qm6(:,3)),grid on
% figure, scatter3(Qm7(:,1),Qm7(:,2),Qm7(:,3)),grid on
% figure, scatter3(Qm8(:,1),Qm8(:,2),Qm8(:,3)),grid on
% [Qm11,Qm12,Qm13,Qm14] = volume_divider(Qm1);
% [Qm21,Qm22,Qm23,Qm24] = volume_divider(Qm2);
% [Qm31,Qm32,Qm33,Qm34] = volume_divider(Qm3);
% [Qm41,Qm42,Qm43,Qm44] = volume_divider(Qm4);

obs = [ ];
% obs1 = islands_finder(Qa,Qm1,mas,0,0,0);
% figure(3)
% scatter3(Q(:,1),Q(:,2),Q(:,3),'r.'),grid on,hold on
% xlabel('q1')
% ylabel('q2')
% zlabel('q3')
obs1 = islands_finder(Qa,Qm11,mas,0,0,0);
obs = [ obs; obs1];
obs1 = islands_finder(Qa,Qm12,mas,0,0,1);
obs1(3,:,:) = [];
obs1(1,:,:) = [];
obs = [ obs; obs1];
obs1 = islands_finder(Qa,Qm13,mas,0,1,1);
obs = [ obs; obs1];
obs1 = islands_finder(Qa,Qm14,mas,0,1,0);
obs = [ obs; obs1];
obs1 = islands_finder(Qa,Qm15,mas,1,0,0); 
obs = [ obs; obs1];
obs1 = islands_finder(Qa,Qm16,mas,1,0,1); 
obs = [ obs; obs1];
obs1 = islands_finder(Qa,Qm17,mas,1,1,1); 
obs = [ obs; obs1];
obs1 = islands_finder(Qa,Qm18,mas,1,1,0); 
obs = [ obs; obs1];


obs1 = islands_finder(Qa,Qm21,mas,0,0,2);
obs = [ obs; obs1];
obs1 = islands_finder(Qa,Qm22,mas,0,0,3);
obs = [ obs; obs1];
obs1 = islands_finder(Qa,Qm23,mas,0,1,3);
obs = [ obs; obs1];
obs1 = islands_finder(Qa,Qm24,mas,0,1,2);
obs = [ obs; obs1];
obs1 = islands_finder(Qa,Qm25,mas,1,0,2);
obs = [ obs; obs1];
obs1 = islands_finder(Qa,Qm26,mas,1,0,3);
obs = [ obs; obs1];
obs1 = islands_finder(Qa,Qm27,mas,1,1,3);
obs = [ obs; obs1];
obs1 = islands_finder(Qa,Qm28,mas,1,1,2);
obs = [ obs; obs1];


obs1 = islands_finder(Qa,Qm31,mas,0,2,0);
obs = [ obs; obs1];
obs1 = islands_finder(Qa,Qm32,mas,0,2,1);
obs = [ obs; obs1];
obs1 = islands_finder(Qa,Qm33,mas,0,3,1);
obs = [ obs; obs1];
obs1 = islands_finder(Qa,Qm34,mas,0,3,0);
obs = [ obs; obs1];
obs1 = islands_finder(Qa,Qm35,mas,1,2,0);
obs = [ obs; obs1];
obs1 = islands_finder(Qa,Qm36,mas,1,2,1);
obs = [ obs; obs1];
obs1 = islands_finder(Qa,Qm37,mas,1,3,1);
obs = [ obs; obs1];
obs1 = islands_finder(Qa,Qm38,mas,1,3,0);
obs = [ obs; obs1];


obs1 = islands_finder(Qa,Qm41,mas,0,2,2);
obs = [ obs; obs1];
obs1 = islands_finder(Qa,Qm42,mas,0,2,3);
obs = [ obs; obs1];
obs1 = islands_finder(Qa,Qm43,mas,0,3,3);
obs = [ obs; obs1];
obs1 = islands_finder(Qa,Qm44,mas,0,3,2);
obs = [ obs; obs1];
obs1 = islands_finder(Qa,Qm45,mas,1,2,2);
obs = [ obs; obs1];
obs1 = islands_finder(Qa,Qm46,mas,1,2,3);
obs = [ obs; obs1];
obs1 = islands_finder(Qa,Qm47,mas,1,3,3);
obs = [ obs; obs1];
obs1 = islands_finder(Qa,Qm48,mas,1,3,2);
obs = [ obs; obs1];


obs1 = islands_finder(Qa,Qm51,mas,2,0,0);
obs = [ obs; obs1];
obs1 = islands_finder(Qa,Qm52,mas,2,0,1);
obs = [ obs; obs1];
obs1 = islands_finder(Qa,Qm53,mas,2,1,1);
obs = [ obs; obs1];
obs1 = islands_finder(Qa,Qm54,mas,2,1,0);
obs = [ obs; obs1];
obs1 = islands_finder(Qa,Qm55,mas,3,0,0);
obs = [ obs; obs1];
obs1 = islands_finder(Qa,Qm56,mas,3,0,1); 
obs = [ obs; obs1];
obs1 = islands_finder(Qa,Qm57,mas,3,1,1);
obs = [ obs; obs1];
obs1 = islands_finder(Qa,Qm58,mas,3,1,0);
obs = [ obs; obs1];


obs1 = islands_finder(Qa,Qm61,mas,2,0,2);
obs = [ obs; obs1];
obs1 = islands_finder(Qa,Qm62,mas,2,0,3);
obs = [ obs; obs1];
obs1 = islands_finder(Qa,Qm63,mas,2,1,3); 
obs = [ obs; obs1];
obs1 = islands_finder(Qa,Qm64,mas,2,1,2);
obs = [ obs; obs1];
obs1 = islands_finder(Qa,Qm65,mas,3,0,2);
obs = [ obs; obs1];
obs1 = islands_finder(Qa,Qm66,mas,3,0,3);
obs = [ obs; obs1];
obs1 = islands_finder(Qa,Qm67,mas,3,1,3);
obs = [ obs; obs1];
obs1 = islands_finder(Qa,Qm68,mas,3,1,2);
obs = [ obs; obs1];



obs1 = islands_finder(Qa,Qm71,mas,2,2,0);
obs = [ obs; obs1];
obs1 = islands_finder(Qa,Qm72,mas,2,2,1);
obs = [ obs; obs1];
obs1 = islands_finder(Qa,Qm73,mas,2,3,1);
obs = [ obs; obs1];
obs1 = islands_finder(Qa,Qm74,mas,2,3,0);
obs = [ obs; obs1];
obs1 = islands_finder(Qa,Qm75,mas,3,2,0);
obs = [ obs; obs1];
obs1 = islands_finder(Qa,Qm76,mas,3,2,1);
obs = [ obs; obs1];
obs1 = islands_finder(Qa,Qm77,mas,3,3,1);
obs = [ obs; obs1];
obs1 = islands_finder(Qa,Qm78,mas,3,3,0);
obs = [ obs; obs1];



obs1 = islands_finder(Qa,Qm81,mas,2,2,2);
obs = [ obs; obs1];
obs1 = islands_finder(Qa,Qm82,mas,2,2,3);
obs = [ obs; obs1];
obs1 = islands_finder(Qa,Qm83,mas,2,3,3);
obs = [ obs; obs1];
obs1 = islands_finder(Qa,Qm84,mas,2,3,2);
obs = [ obs; obs1];
obs1 = islands_finder(Qa,Qm85,mas,3,2,2);
obs = [ obs; obs1];
obs1 = islands_finder(Qa,Qm86,mas,3,2,3);
obs = [ obs; obs1];
obs1 = islands_finder(Qa,Qm87,mas,3,3,3); 
obs = [ obs; obs1];
obs1 = islands_finder(Qa,Qm88,mas,3,3,2);
obs = [ obs; obs1];
% figure(4)
% scatter3(Q(:,1),Q(:,2),Q(:,3),'r.'),grid on, hold on
% xlabel('q1')
% ylabel('q2')
% zlabel('q3')
% obs2 = islands_finder(Qa,Qm2,mas,0,0,1);
% figure(5)
% scatter3(Q(:,1),Q(:,2),Q(:,3),'r.'),grid on, hold on
% xlabel('q1')
% ylabel('q2')
% zlabel('q3')
% obs3 = islands_finder(Qa,Qm3,mas,0,1,1);
% figure(6)
% scatter3(Q(:,1),Q(:,2),Q(:,3),'r.'),grid on, hold on
% xlabel('q1')
% ylabel('q2')
% zlabel('q3')
% obs4 = islands_finder(Qa,Qm4,mas,0,1,0);
% figure(7)
% scatter3(Q(:,1),Q(:,2),Q(:,3),'r.'),grid on, hold on
% xlabel('q1')
% ylabel('q2')
% zlabel('q3')
% obs5 = islands_finder(Qa,Qm5,mas,1,0,0);
% figure(8)
% scatter3(Q(:,1),Q(:,2),Q(:,3),'r.'),grid on, hold on
% xlabel('q1')
% ylabel('q2')
% zlabel('q3')
% obs6 = islands_finder(Qa,Qm6,mas,1,0,1);
% figure(9)
% scatter3(Q(:,1),Q(:,2),Q(:,3),'r.'),grid on, hold on
% xlabel('q1')
% ylabel('q2')
% zlabel('q3')
% obs7 = islands_finder(Qa,Qm7,mas,1,1,1);
% figure(10)
% scatter3(Q(:,1),Q(:,2),Q(:,3),'r.'),grid on, hold on
% xlabel('q1')
% ylabel('q2')
% zlabel('q3')
% obs8 = islands_finder(Qa,Qm8,mas,1,1,0);
% hold off
% obs = [obs1; obs2; obs3; obs4; obs5; obs6; obs7; obs8];
% figure(2)
% scatter3(Q(:,1),Q(:,2),Q(:,3),'r.'),grid on,hold on
% xlabel('q1')
% ylabel('q2')
% zlabel('q3')
% for i =1:1:counter
%     scatter3(obs(i,1,1),obs(i,2,1),obs(i,3,1),'g*','LineWidth',5.5)  %границы параллелепипедов
%     scatter3(obs(i,1,2),obs(i,2,2),obs(i,3,2),'g*','LineWidth',5.5)
% end
% hold off
% [km,obs1] = islands_finder(Qa,Qm11,counter); counter = counter + km;
% [km,obs2] = islands_finder(Qa,Qm12,counter); counter = counter + km;
% [km,obs3] = islands_finder(Qa,Qm13,counter); counter = counter + km;
% [km,obs4] = islands_finder(Qa,Qm14,counter); counter = counter + km;
% [km,obs5] = islands_finder(Qa,Qm21,counter); counter = counter + km;
% [km,obs6] = islands_finder(Qa,Qm22,counter); counter = counter + km;
% [km,obs7] = islands_finder(Qa,Qm23,counter); counter = counter + km;
% [km,obs8] = islands_finder(Qa,Qm24,counter); counter = counter + km;
% [km,obs9] = islands_finder(Qa,Qm31,counter); counter = counter + km;
% [km,obs10] = islands_finder(Qa,Qm32,counter); counter = counter + km;
% [km,obs11] = islands_finder(Qa,Qm33,counter); counter = counter + km;
% [km,obs12] = islands_finder(Qa,Qm34,counter); counter = counter + km;
% [km,obs13] = islands_finder(Qa,Qm41,counter); counter = counter + km;
% [km,obs14] = islands_finder(Qa,Qm42,counter); counter = counter + km;
% [km,obs15] = islands_finder(Qa,Qm43,counter); counter = counter + km;
% [km,obs16] = islands_finder(Qa,Qm44,counter); counter = counter + km;
% obs = [obs1; obs2; obs3; obs4; obs5; obs6; obs7; obs8; obs9; obs10; obs11; obs12; obs13; obs14; obs15; obs16];
% hold off
% 

% %RRT_star
% 
% figure(3)
% scatter3(Q(:,1),Q(:,2),Q(:,3),'r.'),grid on, hold on
% xlabel('q1')
% ylabel('q2')
% zlabel('q3')
% xlim([0 360])
% ylim([0 360])
% zlim([0 360])
% 
% 
% start.cost = 0;
% start.parent = 0;
% start.coord = [50 80 80];
% goal.coord = [340 300 180];
% goal.cost = 0;
% goal.parent = 0;
% 
% q_new=start.coord;
% 
% % d=0.3; l=0.1; R=0.15; y0=0.2; z0=0.3; x0=d+l/2; l1=0.25; l2=0.25;
% % theta=atan(z0/y0); yy=R*cos(theta); zz=R*sin(theta); tolerance=10;
% % k=-1:0.005:1;
% x_max = 360;
% y_max = 360;
% z_max = 360;
% 
% EPS = 40;
% numNodes = 1000;
% 
% nodes(1) = start;
% 
% figure(3)
% plot3(start.coord(1,1),start.coord(1,2),start.coord(1,3),'b*')
% plot3(goal.coord(1,1),goal.coord(1,2),goal.coord(1,3),'b*')
% for i = 1:1:numNodes
%         % Формируем случайную точку
%     q_rand = [rand(1)*x_max rand(1)*y_max rand(1)*z_max];
%     figure(3)
%     plot3(q_rand(1), q_rand(2), q_rand(3), 'x', 'Color',  [0 0.4470 0.7410])
% %     % Если точка попала в цель, то выход из цикла
% %     % Break if goal node is already reached
%     for j = 1:1:length(nodes) 
%         if nodes(j).coord == goal.coord
%             break
%         end
%     end
%     [q_new,nodes]=RRT_star_c_space(start.coord,goal.coord,q_rand,EPS, nodes,obs);
% end
% 
% D = [];
% for j = 1:1:length(nodes)
%     tmpdist = dist_3d(nodes(j).coord, goal.coord);
%     D = [D tmpdist];
% end
% 
% % Search backwards from goal to start to find the optimal least cost path
% [val, idx] = min(D);
% q_final = nodes(idx);
% goal.parent = idx;
% q_end = goal;
% nodes = [nodes goal];
% tr=1;
% while q_end.parent ~= 0
%     startt = q_end.parent;
%     traj(tr,1) = nodes(startt).coord(1);
%     traj(tr,2) = nodes(startt).coord(2);
%     traj(tr,3) = nodes(startt).coord(3);
%     tr=tr+1;
%     figure(3)
%     line([q_end.coord(1), nodes(startt).coord(1)], [q_end.coord(2), nodes(startt).coord(2)], [q_end.coord(3), nodes(startt).coord(3)], 'Color', 'r', 'LineWidth', 4);
% %     hold on
%     q_end = nodes(startt);
% end
%  hold off
% figure(4)
% plot3(traj(:,1),traj(:,2),traj(:,3)),grid on
% xlabel('x')
% ylabel('y')
% zlabel('z')

%Bi_RRT_star

% xlim([qd_low_lim(1) qd_up_lim(1)])
% ylim([qd_low_lim(2) qd_up_lim(2)])
% zlim([qd_low_lim(3) qd_up_lim(3)])

start.cost = 0;
start.parent = 0;
% start.coord = [rad2deg(q_sp(4)) rad2deg(q_sp(5)) rad2deg(q_sp(7))];
 start.coord = [0 0 0];
goal.coord = [-60 30 45];
goal.cost = 0;
goal.parent = 0;

EPS = 10;
numNodes = 1000;


x_max = 100;
y_max = 160;
z_max = 100;

q_new1=start.coord;
q_new2=goal.coord;
nodes1(1) = start;
nodes2(1) = goal;

figure(5)
scatter3(Q(:,1),Q(:,2),Q(:,3),'r.'),grid on, hold on
xlabel('q1')
ylabel('q2')
zlabel('q3')
plot3(start.coord(1,1),start.coord(1,2),start.coord(1,3),'g*')
plot3(goal.coord(1,1),goal.coord(1,2),goal.coord(1,3),'m*')

for i = 1:1:numNodes
    
    % Формируем случайную точку
    q_rand = [qd_low_lim(1) + rand(1)*x_max qd_low_lim(2) + rand(1)*y_max qd_low_lim(3) + rand(1)*z_max];
    figure(5)
%     plot3(q_rand(1), q_rand(2), q_rand(3), 'x', 'Color',  [0 0.4470 0.7410])
    
    
%     % Если точка попала в цель, то выход из цикла
%     % Break if goal node is already reached
%     for j = 1:1:length(nodes)
%         if nodes(j).coord == q_goal.coord
%             break
%         end
%     end
        if i/2==round(i/2)
            [q_new1,nodes1]=RRT_star_c_space(start.coord,goal.coord,q_rand,EPS, nodes1,obs);
        else [q_new2,nodes2]=RRT_star_c_space(goal.coord,start.coord, q_rand, EPS, nodes2,obs);
        end
        for j1=1:1:length(nodes1)
            for j2=1:1:length(nodes2)
                if dist_3d(nodes1(j1).coord,nodes2(j2).coord) <= EPS && isCollision(nodes1(j1).coord,nodes2(j2).coord)
                    k1=1;
                    figure(5)
                    line([nodes1(j1).coord(1,1), nodes2(j2).coord(1,1)], [nodes1(j1).coord(1,2),nodes2(j2).coord(1,2)], [nodes1(j1).coord(1,3),nodes2(j2).coord(1,3)], 'Color', 'k', 'LineWidth', 2);
                    drawnow
                    hold on
                    break
                else k1=0;
                end
            end
                if k1 == 1
            break
        end    
        end
        if k1 == 1
            break
        end
end
ln2=length(nodes2);
ln1=length(nodes1);

D = [];
for j = 1:1:length(nodes1)
    tmpdist = dist_3d(nodes1(j).coord, nodes2(j2).coord);
    D = [D tmpdist];
end

tr1=1;
% Search backwards from goal to start to find the optimal least cost path
[val, idx] = min(D);
q_final = nodes1(idx);
goal.parent = idx;
goal.coord = nodes2(j2).coord;
q_end = goal;
nodes1 = [nodes1 goal];
while q_end.parent ~= 0
    startt = q_end.parent;
    traj1(tr1,1) = nodes1(startt).coord(1);
    traj1(tr1,2) = nodes1(startt).coord(2);
    traj1(tr1,3) = nodes1(startt).coord(3);
    tr1=tr1+1;
    figure(5)
    line([q_end.coord(1), nodes1(startt).coord(1)], [q_end.coord(2), nodes1(startt).coord(2)], [q_end.coord(3), nodes1(startt).coord(3)], 'Color', 'r', 'LineWidth', 4);
    hold on
    q_end = nodes1(startt);
end

D = [];
for j = 1:1:length(nodes2)
    tmpdist = dist_3d(nodes2(j).coord, nodes1(j1).coord);
    D = [D tmpdist];
end
tr2=1;
% Search backwards from goal to start to find the optimal least cost path
[val, idx] = min(D);
q_final = nodes2(idx);
goal.parent = idx;
goal.coord = nodes1(j1).coord;
q_end = goal;
nodes2 = [nodes2 goal];
while q_end.parent ~= 0
    startt = q_end.parent;
    traj2(tr2,1) = nodes2(startt).coord(1);
    traj2(tr2,2) = nodes2(startt).coord(2);
    traj2(tr2,3) = nodes2(startt).coord(3);
    tr2=tr2+1;
    figure(5)
    line([q_end.coord(1), nodes2(startt).coord(1)], [q_end.coord(2), nodes2(startt).coord(2)], [q_end.coord(3), nodes2(startt).coord(3)], 'Color', 'r', 'LineWidth', 4);
%     hold on
    q_end = nodes2(startt);
end

N=length(traj2(:,1));
for i=1:N
    traj3(i,1)=traj2(N-(i-1),1);
    traj3(i,2)=traj2(N-(i-1),2);
    traj3(i,3)=traj2(N-(i-1),3);
end

N=length(traj1(:,1));
for i=1:N
    traj4(i,1)=traj1(N-(i-1),1);
    traj4(i,2)=traj1(N-(i-1),2);
    traj4(i,3)=traj1(N-(i-1),3);
end

traj_f3=[traj4; traj2];

figure(5)
traj_f=[traj1; traj3];
scatter3(traj_f(:,1),traj_f(:,2),traj_f(:,3),'g*'),grid on
xlabel('x')
ylabel('y')
zlabel('z')

traj_f2=[traj1; traj2];

[xi,yi,zi] = trajectory_intrpolation(traj_f3);

figure(1),hold on
rbt.plot([q1 q2 q3 q4 q5 q6 q7 q8 q9 q10],'scale',0.4)

% rosinit
% [pub,msg] = rospublisher('joint1', 'trajectory_msgs/JointTrajectoryPoint');
%  for rr=1:1:length(xi)
% msg.Positions = [xi(rr) yi(rr) zi(rr)];
% send(pub,msg);
%  pause(0.1);
%  end
%  rosshutdown