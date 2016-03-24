classdef Duplicate < nn.Module
    properties
    end
    methods
        function obj = Duplicate()
            obj = obj@nn.Module();
        end
        function output = fprop(obj, input)
            obj.output = {input, input};
            output = obj.output;
        end
        function grad_input = bprop(obj, input, grad_output)
            obj.grad_input = grad_output{1} + grad_output{2};
            grad_input = obj.grad_input;
        end
    end
end