% ªÒ»°sigmod÷µ
function v = sigmod_func(x)
    v = (exp(x) - exp(-x))./(exp(x) + exp(-x));
end