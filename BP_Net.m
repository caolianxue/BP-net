% BP神经网络
% x:样本输入，y为样本输出，hide_node_arry为神经网络隐藏层神经元个数
% w:权重 b:偏置项
function [w, b] = BP_Net(x, y, hide_node_array)

    eps = 1e-10;
    % 神经元节点个数
    [xnum, xlen] = size(x);
    [~, ylen] = size(y);
    node_array = [xlen, hide_node_array, ylen];
    
    % 初始化权重和偏置项
    [w, b] = init_w_b(node_array);
    
    L = length(node_array) - 1;
    
    d_error = 1;
    maxItorNum = 100000000;
    num = 0;
    while abs(d_error) > eps
        
        % 遍历样本
        sum_error = 0;
        for i = 1: xnum

            % 计算网络中的输入和输出值
            [layer_in, layer_out] = net_value(w, b, x(i, :));

            % 计算各层神经元误差
            d = layer_out{L + 1};
            error = calc_error(d, y(i), w, b, layer_in);
            sum_error = sum_error + error{L};

            % 更新权重w和偏置项b
            [w, b] = adjust_w_b(error, w, b, layer_out);
            
        end
        
        d_error = sum_error / xnum;
        
        fprintf('Iteration number = %d;   d_error = %d\n', num, d_error);
        num = num + 1;
        
        if mod(num, 30) == 0
            plotf(w, b, num);
        end
        
        if num > maxItorNum
            fprintf('d_error = %d\n', d_error);
            break;
        end
        
    end
    
end

% 计算各层神经元误差
function error = calc_error(d, y, w, b, layer_in)

    % 神经元层数（不包括输入层）
    L = length(b);
    error = cell(1, L);
    
    % 计算输出层误差（输出层采用线性函数）
    error{L} = (d-y);
    
    % 计算2~L-1层误差
    for i = 1: L-1
        layer = L - i;
        error{layer} = calc_error_2(error{layer+1}, w{layer+1}, layer_in{layer});
    end
    
end

% 计算所有神经元节点误差(2~L-1层)
function delta = calc_error_2(back_error, back_w, layer_in)
    
    diff_f = @sigmod_diff_func;
    
    node_num = length(layer_in);
    back_node_num = length(back_error);
    delta = zeros(node_num, 1);
    for i = 1: node_num   
        for j = 1: back_node_num
            delta(i) =  delta(i) + back_error(j)*back_w(j, i)*diff_f(layer_in(i));
        end
    end
    
end

% 调整权重w和偏置项b，从前往后调节
function [w, b] = adjust_w_b(delta, w, b, layer_out)

    alpha = 0.005;   % 步长
    L = length(b);
    
    for i = 1: L
        w{i} = adjust_w(w{i}, delta{i}, layer_out{i}, alpha);
        b{i} = adjust_b(b{i}, delta{i}, alpha);
    end
    
end

% 调整权重w
function w = adjust_w(w, delta, pre_layer_out, alpha)

    node_num = length(delta);   % 该层神经元节点个数和误差个数一样
    input_node_num = length(pre_layer_out);  % 前一层神经元节点个数
    % 调整每个神经元节点对应的权重
    for i = 1: node_num
        for j = 1: input_node_num
            w(i, j) = w(i, j) - alpha.*delta(i).*pre_layer_out(j);
        end
    end
    
end

% 调整偏置项
function b = adjust_b(b, delta, alpha)
    node_num = length(delta);   % 该层神经元节点个数和误差个数一样
    for i = 1: node_num
        b(i) = b(i) - alpha.*delta(i);
    end
end

% 初始化权重和偏置项
function [w, b] = init_w_b(node_array)
    layer = length(node_array);
    w = cell(1, layer-1);
    b = cell(1, layer-1);
    for i = 2: layer
          input_node_num = node_array(i-1);
          node_num = node_array(i);
          w{i-1} = rands(node_num, input_node_num);
          b{i-1} = rands(node_num);
    end
end

% sigmod导数
function v = sigmod_diff_func(x)
    f = @sigmod_func;
    v = 1 - f(x).*f(x);
end

function plotf(w, b, num)

    f = @test_func;
    x = linspace(-pi, pi, 50)';
    y = f(x);
    
    tx = -pi:0.01:pi;
    ty = tx;
    index = 1;
    for xi = tx
        [~,o] = net_value(w, b, xi);
        ty(index) = o{4};
        index = index + 1;
    end

    hold on; cla reset;
    plot(x, y, '*r');
    hold on; 
    plot(tx, ty, '-g');
    legend('points on test_func', 'net fit line');
    title(['Iteration Num = ', num2str(num)]);
    
    pause(0.001);
end
