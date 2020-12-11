function [xi,yi,zi] = trajectory_intrpolation(traj_f3)
% Inteprolation using Parametric cubic spline 
% Given points 
% P = [30 0 0.2035  ;
%      0 0 0.0357 ;
%      30 60 0.4717 ; 
%      30 0 0.1038 ;
%      30 30 0.4606] ;
x = traj_f3(:,1); y = traj_f3(:,2); z = traj_f3(:,3);
% Get the arc lengths of the curve 
n = length(x) ;
L = zeros(n,1) ;
for i=2:n     
    arc_length = sqrt((x(i)-x(i-1))^2+(y(i)-y(i-1))^2+(z(i)-z(i-1))^2);     
    L(i) = L(i-1) + arc_length; 
end
% Normalize the arc lengths
L=L./L(n);
% do the spline 
x_t = spline(L,x) ;
y_t = spline(L,y) ;
z_t = spline(L,z) ;
% for interpolation 
tt = linspace(0,1,500) ;
xi = ppval(x_t,tt) ;
yi = ppval(y_t,tt) ;
zi = ppval(z_t,tt) ;
figure(6)
plot3(x,y,z,'.-r') ;
hold on
plot3(xi,yi,zi,'.b') ;
legend('Given points' , 'interpolated') ;
