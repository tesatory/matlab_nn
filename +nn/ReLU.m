classdef ReLU < nn.Module
    properties
    end
    methods
        function obj = ReLU()
            obj = obj@nn.Module();
        end
        function output = fprop(obj, input)
            obj.output = input .* (input > 0);
            output = obj.output;
        end
        function grad_input = bprop(obj, input, grad_output)
            obj.grad_input = grad_output .* (input > 0);
            grad_input = obj.grad_input;
        end
    end
end