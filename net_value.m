% ��������ֵ
% layer_inÿһ������ֵ(�����������)��layer_outÿһ�����ֵ x��������
function [layer_in, layer_out] = net_value(w, b, x)

    layer = length(b);
    layer_in = cell(1, layer);
    layer_out = cell(1, layer + 1);
    layer_out{1} = x';
    f = @sigmod_func;
    
    for i = 1: layer
        wl = w{i};
        bl = b{i};
        node_num = length(bl);
        input_node_num = length(layer_out{i});
        pre_in = layer_out{i};
        
        in = zeros(node_num, 1);
        for j = 1: node_num
            for k = 1: input_node_num
                in(j) = in(j) + wl(j, k)*pre_in(k);
            end
            in(j) = in(j) + bl(j);
        end
        
        layer_in{i} = in;
        
        if i == layer
            layer_out{i+1} = in;   % �����������Ժ���
        else
            layer_out{i+1} = f(in);
        end
        
    end
    
end