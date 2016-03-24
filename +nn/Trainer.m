classdef Trainer < handle
    properties
        batch_size = 100;
        model;
        loss;
        costs = [];
        error = [];
        error_test = [];
        costs_test = [];
        epoch = 1;
    end
    methods 
        function obj = Trainer(model, loss)
            obj = obj@handle();
            obj.model = model;
            obj.loss = loss;
        end
        function train(obj, max_epoch, data, labels, test_data, test_labels, params)
            while obj.epoch <= max_epoch
                % shuffle
                rp = randperm(size(data,2));
                data = data(:,rp);
                labels = labels(:,rp);

                % train
                cost = 0;
                err = 0;
                for batch = 1:floor((size(data,2)/obj.batch_size))
                    D = data(:,(1:obj.batch_size)+(batch-1)*obj.batch_size);
                    L = labels(:,(1:obj.batch_size)+(batch-1)*obj.batch_size);
                    out = obj.model.fprop(D);
                    cost = cost + obj.loss.fprop(out, L);
                    err = err + obj.loss.get_error(out, L);
                    g = obj.loss.bprop(out, L);
                    obj.model.bprop(D, g);
                    obj.model.update(params)
                end
                if obj.loss.size_average
                    obj.costs(end+1) = cost / floor((size(data,2)/obj.batch_size));
                    obj.error(end+1) = err / floor((size(data,2)/obj.batch_size));
                else
                    obj.costs(end+1) = cost / floor((size(data,2)/obj.batch_size)) / obj.batch_size;
                    obj.error(end+1) = err / floor((size(data,2)/obj.batch_size)) / obj.batch_size;
                end

                % test
                out = obj.model.fprop(test_data);
                cost = obj.loss.fprop(out, test_labels);
                err = obj.loss.get_error(out, test_labels);
                if obj.loss.size_average
                    obj.costs_test(end+1) = cost;
                    obj.error_test(end+1) = err;
                else
                    obj.costs_test(end+1) = cost / size(test_data,2);
                    obj.error_test(end+1) = err / size(test_data,2);
                end

                obj.show_stat();
                obj.epoch = obj.epoch + 1;
            end
        end
        
        function show_stat(obj)
            disp([num2str(obj.epoch), '. ', ...
                'train cost:', num2str(obj.costs(end)), ...
                ' err:', num2str(obj.error(end)), ...
                ' | test cost:', num2str(obj.costs_test(end)), ...
                ' err:', num2str(obj.error_test(end))])                
            
                        figure(1)
            subplot(2,1,1)
            lnd = {};
            plot(obj.error_test,'r');
            hold on
            plot(obj.error,'b');
            lnd{end+1} = 'test';
            lnd{end+1} = 'train';
            xlabel('epochs')
            ylabel('error')
            legend(lnd)
            grid on
            hold off

            subplot(2,1,2)
            lnd = {};
            plot(log(obj.costs_test),'r')
            hold on
            plot(log(obj.costs),'b')
            lnd{end+1} = 'test';
            lnd{end+1} = 'train';
            xlabel('epochs')
            ylabel('log(cost)')
            legend(lnd)
            grid on
            hold off
            
            pause(0.01)
        end
    end
end