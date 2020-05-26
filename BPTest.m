% Test
clear;clc;close;
format long g;

f = @test_func;
x = linspace(-pi, pi, 50)';
y = f(x);

close all; figure;

[w, b] = BP_Net(x, y, [3,3]);