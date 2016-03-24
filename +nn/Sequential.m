classdef Sequential < nn.Contrainer
    properties
    end
    methods
        function obj = Sequential()
            obj = obj@nn.Contrainer();
        end
        function output = fprop(obj, input)
            for i = 1:length(obj.modules)
                output = obj.modules{i}.fprop(input);
                input = output;
            end
            obj.output = output;
        end
        function grad_input = bprop(obj, input, grad_output)
            for i = length(obj.modules):-1:2                
                grad_input = obj.modules{i}.bprop(obj.modules{i-1}.output, grad_output);
                grad_output = grad_input;
            end
            grad_input = obj.modules{1}.bprop(input, grad_output);
            obj.grad_input = grad_input;
        end
    end
end