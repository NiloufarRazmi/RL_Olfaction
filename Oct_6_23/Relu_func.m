

function fr = Relu_func(x)
    f = zeros(length(x),1);
    for i = 1:length(x)
    if x(i)>=0
        f(i) = x(i);
    else
        f(i) = 0.2*(exp(x(i))-1);
    end
    end
    fr = f;
end