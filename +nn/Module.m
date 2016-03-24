classdef Module < handle
    properties
        output
        grad_input
    end
    methods
        function obj = Module()
            obj = obj@handle();
        end
        function output = fprop(obj, input)
            obj.output = input;
            output = obj.output;
        end
        function grad_input = bprop(obj, input, grad_output)
            obj.grad_input = grad_output;
            grad_input = obj.grad_input;
        end
        function update(obj, params)            
        end
        function share(obj, m)
        end
    end
end