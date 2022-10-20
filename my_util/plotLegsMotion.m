function [ax, frames, rate] = plotLegsMotion(legs_Data)
%plotLegsMotion: input specific data type(legs info extracted from asf/amc files) to play legs motion
%   Detailed explanation goes here
N = size(legs_Data.root.globalPose, 1);

for i=1:size(legs_Data.root.child, 2) %for each leg
    currentNode = legs_Data.root;
    legs(i).points(1).localRot = eye(3);
    legs(i).points(1).localPos = zeros(3,1);
    legs(i).points(1).dof = currentNode.globalPose(:, 4:6);
%     legs(i).points(1).dof = zeros(N,3);
    legs(i).points(1).globalPos = currentNode.globalPose(:, 1:3);
%     legs(i).points(1).globalPos = zeros(N,3);
    
    j = 1;
    
    while ~strcmp(currentNode.child{1}, 'Null')       
        if size(currentNode.child, 2)>1
            currentNode = getfield(legs_Data, currentNode.child{i});
        else
            currentNode = getfield(legs_Data, currentNode.child{1});
        end
        
        j=j+1;
        
        legs(i).points(j).localRot = eul2rotm(flip(currentNode.axis), 'ZYX');
        legs(i).points(j).localPos = legs(i).points(j).localRot' * currentNode.localOffset';

        legs(i).points(j).dof = zeros(N, 3);
        legs(i).points(j).globalPos = zeros(N, 3); 

        switch j
            case 3
                legs(i).points(j).dof = currentNode.localPos; %hip
            case 4
                legs(i).points(j).dof(:, 1) =currentNode.localPos; %knee
            case 5
                legs(i).points(j).dof(:, [1,3]) = currentNode.localPos; %ankle
            case 6
                legs(i).points(j).dof(:, 1) =currentNode.localPos; %toe
        end
    end

    for n =1:N
        rootRot = eul2rotm(flip(legs(i).points(1).dof(n,:)), 'ZYX') ;
        legs(i).points(1).relativeRot = eye(3);
        for j=2:6
            legs(i).points(j).relativeRot = legs(i).points(j-1).relativeRot * eul2rotm(flip(legs(i).points(j).dof(n,:)), 'ZYX') ;
            legs(i).points(j).globalPos(n, :) = legs(i).points(j-1).globalPos(n, :) + ...
                                                                    (rootRot * legs(i).points(j).localRot * legs(i).points(j).relativeRot * legs(i).points(j).localPos)';
        end
    end

end

%% Plot leg motion
X = zeros(1,6); Y = zeros(1,6); Z = zeros(1,6);
figure('Position',[100,100,1300,800]);

xlim = [min([legs(1).points(5).globalPos(:, 1); legs(2).points(5).globalPos(:, 1)])-0.2, ...
                max([legs(1).points(5).globalPos(:, 1); legs(2).points(5).globalPos(:, 1)])+0.2];
ylim = [min([legs(1).points(4).globalPos(:, 3); legs(2).points(4).globalPos(:, 3)])-0.2, ...
                max([legs(1).points(6).globalPos(:, 3); legs(2).points(6).globalPos(:, 3)])+0.2];
zlim = [min([legs(1).points(6).globalPos(:, 2); legs(2).points(6).globalPos(:, 2)]),max(legs(1).points(1).globalPos(:, 2))];

frames(N) = struct('cdata',[],'colormap',[]);
for i=1:N
    clf; 
    for k=1:2
        for j=1:6
            X(j) =  legs(k).points(j).globalPos(i, 1);
            Y(j) = legs(k).points(j).globalPos(i, 2);
            Z(j) = legs(k).points(j).globalPos(i, 3);
        end
        plot3(X, Z, Y, '-o', 'MarkerSize', 8, 'LineWidth', 5, 'Color', 'blue', 'MarkerFaceColor', 'blue'); hold on;
    end
    grid on; 
    axis ij; 
    axis equal; 
    set(gca, 'xlim', xlim, ...
             'ylim', ylim, ...
             'zlim', zlim);
    xlabel 'x'; ylabel 'z'; zlabel 'y'; 
%     view(16,23);
    frames(i)=getframe(gcf);
%     pause(legs_Data.rate/100);
end

ax = gca;
rate = legs_Data.rate;
end

