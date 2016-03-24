classdef Sigmoid < nn.Module
    properties
    end
    methods
        function obj = Sigmoid()
            obj = obj@nn.Module();
        end
        function output = fprop(obj, input)
            obj.output = 1 ./ (1 + exp(-input));
            output = obj.output;
        end
        function grad_input = bprop(obj, input, grad_output)
            grad_input = grad_output .* obj.output .* (1 - obj.output);
        end
    end
end