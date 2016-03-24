classdef Tanh < nn.Module
    properties
    end
    methods
        function obj = Tanh()
            obj = obj@nn.Module();
        end
        function output = fprop(obj, input)
            obj.output = tanh(input);
            output = obj.output;
        end
        function grad_input = bprop(obj, input, grad_output)
            grad_input = grad_output .* (1 - obj.output.^2);
        end
    end
end