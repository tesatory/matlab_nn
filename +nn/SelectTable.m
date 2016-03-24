classdef SelectTable < nn.Module
    properties
        index
    end
    methods
        function obj = SelectTable(index)
            obj = obj@nn.Module();
            obj.index = index;
        end
        function output = fprop(obj, input)
            obj.output = input{obj.index};
            output = obj.output;
        end
        function grad_input = bprop(obj, input, grad_output)
            obj.grad_input = {};
            for i = 1:length(input)
                if i == obj.index
                    obj.grad_input{i} = grad_output;
                else
                    obj.grad_input{i} = zeros(size(input{i}), 'single');
                end
            end
            grad_input = obj.grad_input;
        end
    end
end