clc,clear all,close all
d=0.25; l=0.1; R=0.15; y0=0.1; z0=0.15; 
x0=d+l/2;
theta=atan(z0/abs(x0)); yy=R*cos(theta); zz=R*sin(theta);
k=-1:0.05:1;
% figure(1), plot3(x0,y0,z0,'g*','LineWidth',4),hold on
% Зануляем всё пространство
for i=1:1:length(k)
    for j=1:1:length(k)
        for m=1:1:length(k)
            A(i,j,m)=0;
        end
    end
end
%Обозначаем все точки стенки единичками
for z=-1:0.05:1
    for y=-1:0.05:1
        for x=-1:0.05:1
            if x<=-d && x>=-d-l
                n=round((1+0.05+x)/0.05); p=round((1+0.05+y)/0.05); r=round((1+0.05+z)/0.05); A(n,p,r)=1; %round - для преобразования в int
            end
        end
    end
end
u1=1;
%Обозначаем дырку в стене ноликами
for z=-1:0.05:1
    for y=-1:0.05:1
        for x=-1:0.05:1
            if(x<=-d) && (x>=-(d+l+0.01))
                if(y>=y0-yy) && (y<=y0+yy)
                    if(z>=z0-zz) && (z<=z0+zz)
                    n=round((1+0.05+x)/0.05); p=round((1+0.05+y)/0.05); r=round((1+0.05+z)/0.05); A(n,p,r)=0; C(u1,:)=[x y z]; u1=u1+1;
                    end
                end
            end
        end
    end
end
% figure(1),scatter3(C(:,1),C(:,2),C(:,3),'go'),hold on
t=1;
for i=1:1:length(k)
    for j=1:1:length(k)
        for m=1:1:length(k)
            if (A(i,j,m)==1)
                x1(t)=(0.05*i-1); y1(t)=(0.05*j-1); z1(t)=(0.05*m-1);
                t=t+1;
            end
        end
    end
end
A1(:,1)=x1;
A1(:,2)=y1;
A1(:,3)=z1;

figure(1)
scatter3(x1,y1,z1),grid on, hold on
xlabel('x')
ylabel('y')
zlabel('z')
xlim([-1.2 1.2])
ylim([-1.2 1.2])
zlim([-1.2 1.2])


% rbt.plot([q1 q2 q3])
% rbt.A([1],[0])
%  
% 
% ans = 
%          1         0         0         0
%          0         0        -1         0
%          0         1         0         0
%          0         0         0         1
% rbt.base * rbt.A([1],[0])
%  
% 
% ans = 
%    -0.5000    0.0000   -0.8660   -0.1107
%    -0.0001   -1.0000    0.0000    0.1748
%    -0.8660    0.0001    0.5000   0.04834
%          0         0         0         1
% rbt.plot([0 0 0])
