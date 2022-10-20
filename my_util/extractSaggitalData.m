for i=1:5
walkDemons_07{i}.vel = diff(walkDemons_07{i}.pos')'/(1/120);
walkDemons_07{i}.accel = diff(walkDemons_07{i}.vel')'/(1/120);
walkDemons_07{i}.pos = walkDemons_07{i}.pos(:, 3:end);
walkDemons_07{i}.vel = walkDemons_07{i}.vel(:, 2:end);
end
