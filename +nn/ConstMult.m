classdef ConstMult < nn.Module
    properties
        c
    end
    methods
        function obj = ConstMult(c)
            obj = obj@nn.Module();
            obj.c = c;
        end
        function output = fprop(obj, input)
            obj.output = obj.c * input;
            output = obj.output;
        end
        function grad_input = bprop(obj, input, grad_output)
            obj.grad_input = obj.c * grad_output;
            grad_input = obj.grad_input;
        end
    end
end