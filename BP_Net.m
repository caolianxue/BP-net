% BP������
% x:�������룬yΪ���������hide_node_arryΪ���������ز���Ԫ����
% w:Ȩ�� b:ƫ����
function [w, b] = BP_Net(x, y, hide_node_array)

    eps = 1e-10;
    % ��Ԫ�ڵ����
    [xnum, xlen] = size(x);
    [~, ylen] = size(y);
    node_array = [xlen, hide_node_array, ylen];
    
    % ��ʼ��Ȩ�غ�ƫ����
    [w, b] = init_w_b(node_array);
    
    L = length(node_array) - 1;
    
    d_error = 1;
    maxItorNum = 100000000;
    num = 0;
    while abs(d_error) > eps
        
        % ��������
        sum_error = 0;
        for i = 1: xnum

            % ���������е���������ֵ
            [layer_in, layer_out] = net_value(w, b, x(i, :));

            % ���������Ԫ���
            d = layer_out{L + 1};
            error = calc_error(d, y(i), w, b, layer_in);
            sum_error = sum_error + error{L};

            % ����Ȩ��w��ƫ����b
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

% ���������Ԫ���
function error = calc_error(d, y, w, b, layer_in)

    % ��Ԫ����������������㣩
    L = length(b);
    error = cell(1, L);
    
    % ����������������������Ժ�����
    error{L} = (d-y);
    
    % ����2~L-1�����
    for i = 1: L-1
        layer = L - i;
        error{layer} = calc_error_2(error{layer+1}, w{layer+1}, layer_in{layer});
    end
    
end

% ����������Ԫ�ڵ����(2~L-1��)
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

% ����Ȩ��w��ƫ����b����ǰ�������
function [w, b] = adjust_w_b(delta, w, b, layer_out)

    alpha = 0.005;   % ����
    L = length(b);
    
    for i = 1: L
        w{i} = adjust_w(w{i}, delta{i}, layer_out{i}, alpha);
        b{i} = adjust_b(b{i}, delta{i}, alpha);
    end
    
end

% ����Ȩ��w
function w = adjust_w(w, delta, pre_layer_out, alpha)

    node_num = length(delta);   % �ò���Ԫ�ڵ������������һ��
    input_node_num = length(pre_layer_out);  % ǰһ����Ԫ�ڵ����
    % ����ÿ����Ԫ�ڵ��Ӧ��Ȩ��
    for i = 1: node_num
        for j = 1: input_node_num
            w(i, j) = w(i, j) - alpha.*delta(i).*pre_layer_out(j);
        end
    end
    
end

% ����ƫ����
function b = adjust_b(b, delta, alpha)
    node_num = length(delta);   % �ò���Ԫ�ڵ������������һ��
    for i = 1: node_num
        b(i) = b(i) - alpha.*delta(i);
    end
end

% ��ʼ��Ȩ�غ�ƫ����
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

% sigmod����
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
