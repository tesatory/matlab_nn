% generate toy data
nwords = 20;
patlen = 5;
patnum = 100;
patterns = randi(nwords, [patlen, patnum]);
story = zeros(patlen, 1000000,1);
for i = 1:size(story,2)
    story(:,i) = patterns(:, randi(patnum));
end
story = story(:);
config = {};
config.hid_dim = 50;
config.dict_sz = nwords;
config.bprop_step = 10;
rnn = nn.RNN(config);
loss = nn.CrossEntropyLoss();
params = {};
params.lrate = 0.1;
batch_size = 64;

for ep = 1:100
    r = randi(length(story)-100, [batch_size,1]);   
    total_err = 0;
    for i = 1:100
        input = story(r)';
        r = r + 1;
        target = story(r)';
        
        out = rnn.fprop(input);
        cost = loss.fprop(out, target);
        total_err = total_err + loss.get_error(out, target);
        grad = loss.bprop(out, target);
        rnn.bprop(input, grad);
        
        if mod(i,5) == 0
            rnn.update(params);
        end        
    end
    disp([ep, total_err/100/batch_size]);
end
