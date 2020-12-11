function sol=isCollision(qn,qnew)

R=0.15*180; y0=(0.2+1)*180; z0=(0.3+1)*180; O1=([0.3 0.2 0.3]+1)*180; O2=([0.4 0.2 0.3]+1)*180;
theta=atan(z0/y0); yy=R*cos(theta); zz=R*sin(theta);

if qn(1,1) < (0.3+1)*180 && qnew(1,1) < (0.3+1)*180
    sol = 1;
elseif qn(1,1) < (0.3+1)*180 && qnew(1,1) > (0.3+1)*180 && qnew(1,1) < (0.4+1)*180
        if (qnew(1,2) >= (y0-yy)) && (qnew(1,2) <= (y0+yy))
            if(qnew(1,3)>=(z0-zz)) && (qnew(1,3)<=(z0+zz))
                sol=1;
            else sol = 0;
            end
            else sol = 0;
        end
elseif qn(1,1) < (0.3+1)*180 && qnew(1,1) > (0.4+1)*180
        if (qnew(1,2) >= (y0-yy)) && (qnew(1,2) <= (y0+yy))
            if(qnew(1,3)>=(z0-zz)) && (qnew(1,3)<=(z0+zz))
                sol=1; 
                else sol = 0;
            end
            else sol = 0;
        end
elseif qn(1,1) > (0.4+1)*180 && qnew(1,1) > (0.4+1)*180
    sol = 1;
elseif qn(1,1) > (0.4+1)*180 && qnew(1,1) < (0.4+1)*180
        if (qnew(1,2) >= (y0-yy)) && (qnew(1,2) <= (y0+yy))
            if(qnew(1,3)>=(z0-zz)) && (qnew(1,3)<=(z0+zz))
                sol=1; 
                else sol = 0;
            end
            else sol = 0;
        end
elseif qn(1,1) > (0.4+1)*180 && qnew(1,1) < (0.3+1)*180
        if (qnew(1,2) >= (y0-yy)) && (qnew(1,2) <= (y0+yy))
            if(qnew(1,3)>=(z0-zz)) && (qnew(1,3)<=(z0+zz))
                sol=1; 
                else sol = 0;
            end
            else sol = 0;
        end
else sol = 0;
end
% p1=intersection1(qn,qr);
% p2=intersection2(qn,qr);
% 
% if qn(1,1)<(0.3+1)*180 && qr(1,1)<(0.3+1)*180
%     sol = 1;
% elseif qn(1,1)>(0.4+1)*180 && qn(1,1)>(0.4+1)*180
%     sol = 1;
% elseif qn(1,1) < (0.3+1)*180 && dist_3d(p1,qr) < dist_3d(qnew,qr)
%     sol = 1;
% elseif qn(1,1) < (0.3+1)*180 && dist_3d(p1,qr) > dist_3d(qnew,qr) && dist_3d(p2,qr) < dist_3d(qnew,qr)
% %     if qnew(1,1) >= (0.3+1)*180 && qnew(1,1) <= (0.4+1)*180
%         if (qnew(1,2) >= (y0-yy)*180) && (qnew(1,2) <= (y0+yy)*180)
%             if(qnew(1,3)>=(z0-zz)*180) && (qnew(1,3)<=(z0+zz)*180)
%                 sol=1;
% %             end
%         end
%     end
% elseif qn(1,1) < (0.3+1)*180 && dist_3d(p1,qr) > dist_3d(qnew,qr) && dist_3d(p2,qr) > dist_3d(qnew,qr)
%     if (qnew(1,2) >= (y0-yy)*180) && (qnew(1,2) <= (y0+yy)*180)
%             if(qnew(1,3)>=(z0-zz)*180) && (qnew(1,3)<=(z0+zz)*180)
%                 sol = 1;
%             end
%     end
% elseif qn(1,1) > (0.4+1)*180 && qnew > (0.4+1)*180 
%     sol = 1;
% else sol = 0;
% end
    
    
           
% if q_near(1,1)<=(0.3+1)*180 && q_rand(1,1)<=(0.3+1)*180
%     sol=1;
% elseif q_near(1,1)>=(0.4+1)*180 && q_rand(1,1)>=(0.4+1)*180
%     sol=1;
% else
%     p1=intersection1(q_near,q_rand);
%     d1=dist_3d(q_near,p1); d2=dist_3d(q_rand,q_near);
%     if d1 < d2
%         sol = 1;
%     else sol = 0;
%     end
%         
    
    
% if q_near(1,1)>(0.3+1)*180 && q_near(1,1)<(0.4+1)*180 && (q_rand(1,1)<(0.3+1)*180 || q_rand(1,1)>(0.4+1)*180)
%     if (norm(q_near(2:3) - [y0 z0]) < R)
%     sol=1;
%     else sol=0;
%     end
% else sol=0;
% end
%                     if(q_near(1,2)>=(y0-yy)*180) && (q_near(1,2)<=(y0+yy)*180)
%                         if(q_near(1,3)>=(z0-zz)*180) && (q_near(1,3)<=(z0+zz)*180)
% %                             if q_near(1,1)-q_rand(1,1) ~= 0
% %                                 p1=intersection1(q_rand,q_near);
% %                                 if dist_3d(O1,p1)>R
% %                                  sol=0;
% %                                 else sol=1;
%                                     sol=1;
% %                         else sol=0;
% %                                 end
% %                             end
%                         end
%                     end
% elseif q_rand(1,1)>(0.3+1)*180 && q_rand(1,1)<(0.4+1)*180 && q_near(1,1)>(0.4+1)*180
%                         if(q_rand(1,2)>=(y0-yy)*180) && (q_rand(1,2)<=(y0+yy)*180)
%                         if(q_rand(1,3)>=(z0-zz)*180) && (q_rand(1,3)<=(z0+zz)*180)
%                             sol=1;
%                         else sol=0;
% %                             if q_near(1,1)-q_rand(1,1) ~= 0
% %                                 p2=intersection2(q_rand,q_near);
% %                                 if dist_3d(O2,p2)>R
% %                                  sol=0;
% %                                 else sol=1;
% %                                 end
% %                             end
%                         end
%                         end
% elseif q_rand(1,1)<(0.3+1)*180 && q_near(1,1)>(0.4+1)*180
%     if q_near(1,1)-q_rand(1,1) ~= 0
%     p1=intersection1(q_rand,q_near);
%     p2=intersection2(q_rand,q_near);
%     if p1(1,1)==(0.3+1)*180
%           if (p1(1,2)>=(y0-yy)*180) && (p1(1,2)<=(y0+yy)*180)
%                if (p1(1,3)>=(z0-zz)*180) && (p1(1,3)<=(z0+zz)*180)
%                    if p2(1,1)==(0.3+1)*180
%                         if (p2(1,2)>=(y0-yy)*180) && (p2(1,2)<=(y0+yy)*180)
%                             if (p2(1,3)>=(z0-zz)*180) && (p2(1,3)<=(z0+zz)*180)
%                                 sol=1;
%                             end
%                         end
%                    end
%                end
%           end
%     end
%                
% %         if dist_3d(O1,p1)>R
% %             sol=0;
% %         elseif dist_3d(O2,p2)>R
% %             sol=0;
% %         else sol=1;
% %         end
%     end
% elseif q_near(1,1)>(0.4+1)*180 && q_rand(1,1)>(0.4+1)*180
%     sol=1;
% else sol=0;
% end
% %                                  elseif dist_3d(O2,p2)>R
% %                                     sol=0;
% %                                 else sol=1;
% %                             sol=1;
% %                         end
% %                     end
% %                     
% % elseif q_near(1,1)>(0.4+1)*180 && q_rand(1,1)>(0.4+1)*180
% % 
% % elseif q_near(1,1)-q_rand(1,1) ~= 0
% %     p1=intersection1(q_rand,q_near);
% %     p2=intersection2(q_rand,q_near);
% %         if dist_3d(O1,p1)>R
% %             sol=0;
% %         elseif (q_near(1,1)>=(0.3+1)*180 && q_near(1,1)<=(0.4+1)*180)
% %                 if(q_near(1,2)>=(y0-yy)*180) && (q_near(1,2)<=(y0+yy)*180)
% %                     if(q_near(1,3)>=(z0-zz)*180) && (q_near(1,3)<=(z0+zz)*180)
% %                       sol=1;
% %                     end
% %                 end
% %         elseif dist_3d(O2,p2)>R
% %             sol=0;
% %         else sol=1;
% %         end
% % end
        
