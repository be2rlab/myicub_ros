function obs = islands_finder(Qa,Qm,mas,dx,dy,dz)
size1 = length(Qm(:,1,1));
size2 = length(Qm(1,:,1));
size3 = length(Qm(1,1,:));
% figure
% scatter3(Q(:,1),Q(:,2),Q(:,3),'r.'),grid on, hold on
% xlabel('q1')
% ylabel('q2')
% zlabel('q3')
iii=1; obs = [];
L = bwlabeln(Qm);
km=max(max(max(L)));
for i = 1:1:km
    [ix,iy,iz]=findND(L==i);  
%     ix = ix+ix*dx; iy = iy+iy*dy; iz = iz+iz*dz;
        for i1=1:length(ix)
            for kk=1:length(mas(:,1))
                if ix(i1)+size1*dx == mas(kk,2) && iy(i1)+size2*dy == mas(kk,3) && iz(i1)+size3*dz == mas(kk,4)
                    ii=mas(kk,1);
                    Qc(iii,1) = Qa(ii,1); Qc(iii,2) = Qa(ii,2); Qc(iii,3) = Qa(ii,3);
                    iii=iii+1;
                end
            end
        end
    if iii > 1
%      scatter3(Qc(:,1),Qc(:,2),Qc(:,3),'bo'), grid on, hold on
    xmin=min(Qc(:,1)); ymin=min(Qc(:,2)); zmin=min(Qc(:,3));
    xmax=max(Qc(:,1)); ymax=max(Qc(:,2)); zmax=max(Qc(:,3)); 
    x0 = (xmax + xmin)/2; y0 = (ymax + ymin)/2; z0 = (zmax + zmin)/2;
    x1 = (xmax - xmin)/2; y1 = (ymax - ymin)/2; z1 = (zmax - zmin)/2;
    if xmin == 0 && ymin == 0 && zmin == 0 && ...
            xmax == 0 && ymax == 0 && zmax == 0
        iii = 1;
    else
    iii=1;
    obs(i,:,:)=[x0 y0 z0; x1 y1 z1].';
    figure(2),hold on 
    plot3(xmin,y0,z0,'g*','LineWidth',5.5)
    plot3(xmax,y0,z0,'g*','LineWidth',5.5)
    plot3(x0,ymin,z0,'g*','LineWidth',5.5)
    plot3(x0,ymax,z0,'g*','LineWidth',5.5)
    plot3(x0,y0,zmin,'g*','LineWidth',5.5)
    plot3(x0,y0,zmax,'g*','LineWidth',5.5)
%     scatter3(obs(i,1,1),obs(i,2,1),obs(i,3,1),'g*','LineWidth',5.5)  %границы параллелепипедов
%     scatter3(obs(i,1,2),obs(i,2,2),obs(i,3,2),'g*','LineWidth',5.5)
    end
    end
end
% hold off