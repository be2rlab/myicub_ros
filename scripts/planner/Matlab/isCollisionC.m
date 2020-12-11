function flag=isCollisionC(qnew,qnear,obs)
flag = true;

n = size(obs,1);
for k = 1:n %острова
    x0 = obs(k,1,1); xr = obs(k,1,2);
    y0 = obs(k,2,1); yr = obs(k,2,2);
    z0 = obs(k,3,1); zr = obs(k,3,2);
    R = max([xr yr zr]);
    x_left_lim = x0 - xr*sqrt(2); x_right_lim = x0 + xr*sqrt(2); %*sqrt(2)
    y_left_lim = y0 - yr*sqrt(2); y_right_lim = y0 + yr*sqrt(2); 
    z_left_lim = z0 - zr*sqrt(2); z_right_lim = z0 + zr*sqrt(2); 
    j = 0;
    d = dist_3d(qnew,[x0 y0 z0]);
    if d <= R
        flag = false;
        break
    else
    if qnew(1) == qnear(1)
        if qnew(1) > x_left_lim && qnew(1) < x_right_lim %|| qnew(1) < x_left_lim && qnew(1) > x_right_lim
            j = j + 1;
        end
    else if qnew(1) > x_left_lim && qnew(1) < x_right_lim || qnear(1) > x_left_lim && qnear(1) < x_right_lim %|| ...
                %qnew(1) < x_left_lim && qnew(1) > x_right_lim || qnear(1) < x_left_lim && qnear(1) > x_right_lim
            j = j + 1;
        end
    end
    
    if qnew(2) == qnear(2)
        if qnew(2) > y_left_lim && qnew(2) < y_right_lim %|| qnew(2) < y_left_lim && qnew(2) > y_right_lim
            j = j + 1;
        end
    else if qnew(2) > y_left_lim && qnew(2) < y_right_lim || qnear(2) > y_left_lim && qnear(2) < y_right_lim % || ...
              % qnew(2) < y_left_lim && qnew(2) > y_right_lim  || qnear(2) < y_left_lim && qnear(2) > y_right_lim
            j = j + 1;
        end
    end

     if qnew(3) == qnear(3)
        if qnew(3) > z_left_lim && qnew(3) < z_right_lim %|| qnew(3) < z_left_lim && qnew(3) > z_right_lim
            j = j + 1;
        end
    else if qnew(3) > z_left_lim && qnew(3) < z_right_lim || qnear(3) > z_left_lim && qnear(3) < z_right_lim % || ...
               % qnew(3) < z_left_lim && qnew(3) > z_right_lim || qnear(3) < z_left_lim && qnear(3) > z_right_lim
            j = j + 1;
        end
    end
    end
    
    if j == 3
        flag = false;
        break
    end
    
%     x1 = qnear(1); y1 = qnear(2); z1 = qnear(3);
%     x2 = qnew(1); y2 = qnew(2); z2 = qnew(3);
%     d=dist_3d([x0 y0 z0],qnew);
%     if d <= R
%         flag = false;
%         break
%     else
%         sol = solve((((x2-x1)*t+x1-x0)^2 + ((y2 - y1)*t + y1 - y0)^2 +((z2 - z1)*t + z1 - z0)^2 == R^2),t);
%         if isreal(sol)
%             flag = false;
%             break
%         else flag = true;
%         end
%     end
%     j=0;
%     for i = 1:3
%         if qnew(i) > obs(k,i,1) && qnew(i) < obs(k,i,2)
%             j=j+1;
%             if j~=3
%                  flag = true;
%             else flag = false;
%             break
%             end
%         end
%     end
%        if flag == false
%        break
%     end
end
            