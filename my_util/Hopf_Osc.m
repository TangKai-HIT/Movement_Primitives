function [x, dx]=Hopf_Osc(initial, mu, omega, timeQuery, plotPortrait)
%van der pol  oscillator

    sol = ode45(@(t,y) oscFunc(t,y,mu, omega), [timeQuery(1), timeQuery(end)], initial);
    y = deval(sol, timeQuery);
    x = y(1, :);
    dx = y(2, :);
    
    %plot
    if plotPortrait==true
        figure()
        plot(x,dx,'LineWidth', 2); hold on;
        plot(initial(1), initial(2), '*','MarkerSize', 12, 'Color', 'magenta')
        title(sprintf('Hopf Oscillator ($\\mu=%.2f, \\omega=%.2f$)', mu, omega), 'Interpreter', 'latex', 'FontSize', 14);
        xlabel('$x$', 'Interpreter', 'latex', 'FontSize', 15);
        ylabel('$\dot{x}$', 'Interpreter', 'latex', 'FontSize', 15);
        
        % vector field
        [X, Y] = meshgrid(min(x):0.4:max(x), min(dx):0.4:max(dx));
        [U, V] = oscVectors(X, Y, mu, omega);
        quiver(X,Y,U,V, 'LineWidth', 1, 'AutoScaleFactor', 1, 'Color', 'red');
%         axis equal
    end
    
    function dydt = oscFunc(t, y, mu, omega)
        dydt = [(mu - y(1)^2 - y(2)^2)*y(1) - omega*y(2); 
                   (mu - y(1)^2 - y(2)^2)*y(2) + omega*y(1)];
    end

    function [U, V] = oscVectors(X, Y, mu, omega)
        U = (mu - X.^2 - Y.^2).*X - omega.*Y; 
        V =  (mu - X.^2 - Y.^2).*Y + omega.*X;
    end
end