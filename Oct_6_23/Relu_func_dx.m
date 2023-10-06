

function fr = Relu_func_dx(x)
    f = zeros(length(x),1);
    for i = 1:length(x)
    if x(i)>=0
        f(i) = 1;
    else
        f(i) = 0.2*exp(x(i));
    end
    end
    fr = f;
end