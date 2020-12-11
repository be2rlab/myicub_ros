l1=0.25; l2=0.25;  %syms t;
% manipulator part

q_low_lim = [deg2rad(-95) deg2rad(0) deg2rad(6) ]; % ограничение по вращению сочленений(нижнее)
q_up_lim = [deg2rad(5)  deg2rad(160) deg2rad(106) ]; % -//- (верхнее)


% Перевод в градусы
qd_low_lim = [ rad2deg(q_low_lim(1)) rad2deg(q_low_lim(2)) rad2deg(q_low_lim(3)) ];
qd_up_lim = [ rad2deg(q_up_lim(1)) rad2deg(q_up_lim(2)) rad2deg(q_up_lim(3)) ];

% Создание робота в Corke Robotic Toolbox

L1 = Link('revolute','d', 0, 'a', 32/1000, 'alpha', pi/2,'offset',0);%,'qlim',[deg2rad(-22) deg2rad(84)]);%
L2 = Link('revolute','d',-5.5/1000, 'a', 0, 'alpha', pi/2,'offset',-pi/2);%,'qlim',[deg2rad(-39) deg2rad(39)]);%
L3 = Link('revolute','d', -143.3/1000, 'a', -23.3647/1000, 'alpha', pi/2,'offset',deg2rad(-105));%,'qlim',[deg2rad(-59) deg2rad(59)]);%
L4 = Link('revolute','d', -107.74/1000, 'a', 0, 'alpha', pi/2,'offset',deg2rad(-105));%,'qlim',[deg2rad(5) deg2rad(-95)]);%,'offset',-pi/2);
L5 = Link('revolute','d',0, 'a', 0, 'alpha', -pi/2,'offset',-pi/2);%,'qlim',[deg2rad(0) deg2rad(160.8)]);%
L6 = Link('revolute','d', -152.28/1000, 'a', -15/1000, 'alpha', -pi/2,'offset',deg2rad(-105));%,'qlim',[deg2rad(-37) deg2rad(100)]);%
L7 = Link('revolute','d', 0, 'a', 15/1000, 'alpha', pi/2,'offset', 0);%deg2rad(60),'qlim',[deg2rad(5.5) deg2rad(106)]);%
L8 = Link('revolute','d',-137.3/1000, 'a', 0, 'alpha', pi/2,'offset',-pi/2);%,'qlim',[deg2rad(-50) deg2rad(50)]);%
L9 = Link('revolute','d', 0, 'a', 0, 'alpha', pi/2,'offset',pi/2);%,'qlim',[deg2rad(10) deg2rad(-65)]);%
L10 = Link('revolute','d', 16/1000, 'a', 62.5/1000, 'alpha', 0,'offset',pi);%,'qlim',[deg2rad(-25) deg2rad(25)]);%
rbt = SerialLink([L1,L2,L3,L4,L5,L6,L7,L8,L9,L10],'name','iCub');
B = [0 -1 0 0; 0 0 -1 0; 1 0 0 0 ; 0 0 0 1];
rbt.base = B;

q1 = 0;
q2 = 0;
q3 = 0;
q4 = 0;
q5 = 0;
q6 = 0;
q7 = 0;
q8 = 0;
q9 = 0;
q10 = 0;

q_sp = [q1 q2 q3 q4 q5 q6 q7 q8 q9 q10];

% figure(1),hold on, rbt.plot(q_sp,'scale',0.4)

% start_point = rbt.fkine(q_fk);
% sp = start_point.transl;
% rbt.plot(q_fk)

% rbt2.plot([0 -pi/2 deg2rad(-105) -pi/2 -pi/2 deg2rad(-105) 0 -pi/2 pi/2 2*pi])
% rbt.plot([q1 q2 q3 q4 q5 q6 q7 q8 q9 10]) 
% 4 - плечо вперед-назад
% 5 - плечо вправо-влево
% 6 - плечо вокруг оси
% 7 - локоть
win=transl([0.3 0.3 0.3]);
initial=[0 0 0];



% L1 = Link('revolute','d', 0, 'a', 0, 'alpha', pi/2);
% L2 = Link('revolute','d',0, 'a', l1, 'alpha', pi/2);
% L3 = Link('revolute','d', 0, 'a', l2, 'alpha', 0);
% win=transl([0.3 0.3 0.3]);
% L1.qlim=[rad2deg(q_low_lim(1)) rad2deg(q_up_lim(1))];
% L2.qlim=[rad2deg(q_low_lim(2)) rad2deg(q_up_lim(2))];
% L3.qlim=[rad2deg(q_low_lim(3)) rad2deg(q_up_lim(3))];
% initial=[0 0 0];
% 
% rbt = SerialLink([L1 L2 L3]);
i1=1; i2=1; 
j1 = 0; j2 = 0; j3 = 0; mas=[];
tolerance=0.02;
dq1=deg2rad(100)/15;
dq2=deg2rad(160)/15;
dq3=deg2rad(100)/15;
%Конфигурационное пространство(создание)

for q4=deg2rad(-95):dq1:deg2rad(5)
     j1=j1+1; j2 = 0; j3 = 0;
    for q5=deg2rad(0):dq2:deg2rad(160)
        j2=j2+1; j3 = 0;
        for q7=deg2rad(6):dq3:deg2rad(106)
            j3=j3+1;
            H=rbt.base*rbt.A([1:5],[q1,q2,q3,q4,q5,q6,q7,q8,q9,q10]);
            p1=H.t;
            H=rbt.base*rbt.A([1:7],[q1,q2,q3,q4,q5,q6,q7,q8,q9,q10]);
            p2=H.t;
            H=rbt.base*rbt.A([1:9],[q1,q2,q3,q4,q5,q6,q7,q8,q9,q10]);
            p3=H.t;
    %заполнить матрицу D расстояниями до каждой из точек матрицы А
            for i=1:1:length(x1)
                D3(i)=point_to_line_distance([A1(i,1),A1(i,2),A1(i,3)],p1.',p2.');
                D4(i)=point_to_line_distance([A1(i,1),A1(i,2),A1(i,3)],p2.',p3.');
                D5(i)=dist_3d([A1(i,1),A1(i,2),A1(i,3)],p1.');
                D6(i)=dist_3d([A1(i,1),A1(i,2),A1(i,3)],p2.');
                D7(i)=dist_3d([A1(i,1),A1(i,2),A1(i,3)],p3.');
            end
    %Найти минимальное расстояние(минимальный элемент матрицы D)
    M3=min(D3);
    M4=min(D4);
    M5=min(D5);
    M6=min(D6);
    M7=min(D7);
    
    
    if M4 < tolerance
        if M7 < tolerance
           Q(i1,1)=q4*(180/pi); Qa(i2,1)=Q(i1,1);
           Q(i1,2)=q5*(180/pi); Qa(i2,2)=Q(i1,2);
           Q(i1,3)=q7*(180/pi); Qa(i2,3)=Q(i1,3);  Qm(j1,j2,j3)=1; mas=[mas; i2 j1 j2 j3];
           i1=i1+1; i2=i2+1;
        elseif M6 < tolerance
           Q(i1,1)=q4*(180/pi); Qa(i2,1)=Q(i1,1);
           Q(i1,2)=q5*(180/pi); Qa(i2,2)=Q(i1,2);
           Q(i1,3)=q7*(180/pi); Qa(i2,3)=Q(i1,3);  Qm(j1,j2,j3)=1; mas=[mas; i2 j1 j2 j3];
           i1=i1+1; i2=i2+1;         
        else 
           Qa(i2,1)=0;
           Qa(i2,2)=0;
           Qa(i2,3)=0;
           Qm(j1,j2,j3)=0; mas=[mas; i2 j1 j2 j3];
           i2=i2+1;   
        end
    else if M3 < tolerance
            if M6 < tolerance
           Q(i1,1)=q4*(180/pi); Qa(i2,1)=Q(i1,1);
           Q(i1,2)=q5*(180/pi); Qa(i2,2)=Q(i1,2);
           Q(i1,3)=q7*(180/pi); Qa(i2,3)=Q(i1,3);  Qm(j1,j2,j3)=1; mas=[mas; i2 j1 j2 j3];
           i1=i1+1; i2=i2+1;         
            elseif M5 < tolerance
           Q(i1,1)=q4*(180/pi); Qa(i2,1)=Q(i1,1);
           Q(i1,2)=q5*(180/pi); Qa(i2,2)=Q(i1,2);
           Q(i1,3)=q7*(180/pi); Qa(i2,3)=Q(i1,3);  Qm(j1,j2,j3)=1; mas=[mas; i2 j1 j2 j3];
           i1=i1+1; i2=i2+1;   
            else 
           Qa(i2,1)=0;
           Qa(i2,2)=0;
           Qa(i2,3)=0;
           Qm(j1,j2,j3)=0; mas=[mas; i2 j1 j2 j3];
           i2=i2+1;   
            end
        end
    end
        end
    end
end
i1=1; i2=1;


%  Конфигурационное пространство(изображение)
figure(2)
scatter3(Q(:,1),Q(:,2),Q(:,3),'r.'),grid on
xlabel('q1')
ylabel('q2')
zlabel('q3')
% xlim([-180 180])
% ylim([-180 180])
% zlim([-180 180])
