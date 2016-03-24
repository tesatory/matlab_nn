classdef AddTable < nn.Module
    properties
    end
    methods
        function obj = AddTable()
            obj = obj@nn.Module();
        end
        function output = fprop(obj, input)
            obj.output = input{1};
            for i = 2:length(input)
                obj.output = obj.output + input{i};
            end
            output = obj.output;
        end
        function grad_input = bprop(obj, input, grad_output)
            obj.grad_input = {};
            for i = 1:length(input)
                obj.grad_input{i} = grad_output;
            end
            grad_input = obj.grad_input;
        end
    end
end