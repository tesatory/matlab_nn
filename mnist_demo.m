load data/mnist_14x14.mat

model = nn.Sequential();
model.add(nn.Linear(14*14, 50));
model.add(nn.ReLU());
model.add(nn.Linear(50, 10));
model.add(nn.Softmax());

loss = nn.CrossEntropyLoss();

trainer = nn.Trainer(model, loss);
params = {};
params.lrate = 0.3;

% params.lrate = 0.01;
% params.mom = 0.;
% params.mom2 = 0.999;

trainer.train(50, train_data, train_labels, test_data, test_labels, params);