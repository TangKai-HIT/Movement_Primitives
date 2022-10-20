function [x, dx]=VanDerPol_Osc(initial, alpha, omega, timeQuery, plotPortrait)
%van der pol  oscillator

    sol = ode45(@(t,y) oscFunc(t,y,alpha, omega), [timeQuery(1), timeQuery(end)], initial);
    y = deval(sol, timeQuery);
    x = y(1, :);
    dx = y(2, :);
    
    %plot
    if plotPortrait==true
        figure()
        plot(x,dx,'LineWidth', 2); hold on;
        plot(initial(1), initial(2), '*','MarkerSize', 12, 'Color', 'magenta')
        title(sprintf('Van Der Pol Oscillator ($\\alpha=%.2f, \\omega=%.2f$)', alpha, omega), 'Interpreter', 'latex', 'FontSize', 14);
        xlabel('$x$', 'Interpreter', 'latex', 'FontSize', 15);
        ylabel('$\dot{x}$', 'Interpreter', 'latex', 'FontSize', 15);
        hold on;
        % vector field
        [X, Y] = meshgrid(min(x):0.4:max(x), min(dx):0.4:max(dx));
        [U, V] = oscVectors(X, Y, alpha, omega);
        quiver(X,Y,U,V, 'LineWidth', 1, 'AutoScaleFactor', 1.5, 'Color', 'red');
%         axis equal
    end
    
    function dydt = oscFunc(t, y, alpha, omega)
        dydt = [y(2); alpha*(1-y(1)^2)*y(2)-omega^2*y(1)];
    end

    function [U, V] = oscVectors(X, Y, alpha, omega)
        U = Y; 
        V =  alpha*(1-X.^2).*Y - omega^2.*X;
    end
end