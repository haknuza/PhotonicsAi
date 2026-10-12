Task 5: PINN for the Silverbox oscillator

Load the MATLAB file SNLS80mV.mat. V1 is input voltage u and V2 is output voltage y,
sampled at 610.35 Hz (dt=1/610.35). Use samples [42650:46746] (4096 samples);
subtract each channel's mean over this excerpt. The first 3072 samples are training;
the last 1024 are test.

Model the centered signals with
    y'' + a y' + b y + c y^3 = g u(t),
where a,b,c,g are circuit coefficients. Implement acceleration with a PyTorch
multilayer fully connected network. For v=y', integrate y'=v,
v'=alpha_theta(y,v,u) at step dt, driven by measured input (RK4 is suitable).

Compare at least two network architectures and tune optimizer, learning rate,
training steps, and rollout length. Train only on the first 3072 samples.
After training, make one free run from the excerpt's first measured position and
finite-difference velocity, continuing through training and test without resetting
state. During test, the model may read u but must not read y. Report free-run train
and test errors; test RMSE is
    sqrt((1/1024) * sum_{i=1..1024} (yhat_i-y_i)^2).

Deliver the retained architecture and training settings, other settings tried and
their test RMSEs, both train and test free-run RMSEs, and a plot of measured and
predicted test output.
