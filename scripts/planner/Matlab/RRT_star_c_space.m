function [q_new1,nodes] = RRT_star_c_space(start, goal, q_rand, EPS, nodes,obs)

q_start.coord = start;
q_start.cost = 0;
q_start.parent = 0;
q_goal.coord = goal;
q_goal.cost = 0;


nodes(1) = q_start;
    
    %Берём ближайший узел из списка, к которому будет строится граф
    % Pick the closest node from existing list to branch out from
            ndist = [];
    for j = 1:1:length(nodes)
        n = nodes(j);
        tmp = dist_3d(n.coord, q_rand);
        ndist = [ndist tmp];
    end
    [val, idx] = min(ndist);
    q_near = nodes(idx);
    q_new.coord = steer3d(q_rand, q_near.coord, val, EPS);
    if isCollisionC(q_new.coord,q_near.coord,obs)
    line([q_near.coord(1), q_new.coord(1)], [q_near.coord(2), q_new.coord(2)], [q_near.coord(3), q_new.coord(3)], 'Color', 'k', 'LineWidth', 2);
    drawnow
    hold on
    q_new.cost = dist_3d(q_new.coord, q_near.coord) + q_near.cost;
    
    % Within a radius of r, find all existing nodes
    q_nearest = [];
    r = 50;
    neighbor_count = 1;
    for j = 1:1:length(nodes)
        if  isCollisionC(q_new.coord,nodes(j).coord,obs) && (dist_3d(nodes(j).coord, q_new.coord)) <= r
            q_nearest(neighbor_count).coord = nodes(j).coord;
            q_nearest(neighbor_count).cost = nodes(j).cost;
            neighbor_count = neighbor_count+1;
        end
    end
    
    % Initialize cost to currently known value
    q_min = q_near;
    C_min = q_new.cost;
    
    % Iterate through all nearest neighbors to find alternate lower
    % cost paths
    
    for k = 1:1:length(q_nearest)
        if isCollisionC(q_new.coord,q_nearest(k).coord,obs) && q_nearest(k).cost + dist_3d(q_nearest(k).coord, q_new.coord) < C_min
            q_min = q_nearest(k);
            C_min = q_nearest(k).cost + dist_3d(q_nearest(k).coord, q_new.coord);
            line([q_min.coord(1), q_new.coord(1)], [q_min.coord(2), q_new.coord(2)], [q_min.coord(3), q_new.coord(3)], 'Color', 'g');            
            hold on
        end
    end
    
    % Update parent to least cost-from node
    for j = 1:1:length(nodes)
        if nodes(j).coord == q_min.coord
            q_new.parent = j;
        end
    end
    
    % Append to nodes
    nodes = [nodes q_new];
    end
q_new1=q_new.coord;