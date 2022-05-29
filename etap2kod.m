clear; clc; close all;

%% Podstawowe założenia sieci
% liczba neuronów w warstwie wejściowej
input_size = 5;
% liczba neuronów w warstwie ukrytej
hidden_size = 20;
% liczba neuronów w warstwie wyjściowej
output_size = 1;
% liczba cech
num_feat = 5;

%% Przygotowanie danych wejściowych
data_Input = readtable('data.csv');

% perimeter_mean
data_Table(:,1) = data_Input(:,"perimeter_mean");
% area_mean
data_Table(:,2) = data_Input(:,"area_mean");
% area_se
data_Table(:,3) = data_Input(:,"area_se");
% concavity_worst
data_Table(:,4) = data_Input(:,"concavity_worst");
% concavePoints_worst
data_Table(:,5) = data_Input(:,"concavePoints_worst");

% Macierz z wybranymi atrybutami
data_Before_Normalization = table2array(data_Table);

% diagnosis
for k = 1:height(data_Input)
    if(strcmp(data_Input.diagnosis(k), 'B'))
        data_Before_Normalization(k,6) = 0;
    elseif(strcmp(data_Input.diagnosis(k), 'M'))
        data_Before_Normalization(k,6) = 1;
    end
end

% normalizacja wartości atrybutów
u = 1;
l = 0;
maxValues = max(data_Before_Normalization);
minValues = min(data_Before_Normalization);

data=zeros(height(data_Before_Normalization),width(data_Before_Normalization));

for j = 1:width(data_Before_Normalization)-1
    for k = 1:height(data_Before_Normalization)
      data(k ,j)= ((data_Before_Normalization(k,j) - minValues(j)) / (maxValues(j)-minValues(j))) * (u-l) + l;
    end
end
data(:,6)=data_Before_Normalization(:,6); 

% W macierzy data - wszystkie znormalizowane cechy, w ostatniej kolumnie
% diagnoza: B->0, M->1


% Podział na zbiór uczący i testujący

% Procent danych, które zostaną przypisane do zbioru testowego
data(:,7)=zeros(height(data),1);

test=0.2;
test_count= uint16(test*height(data));
test_index= randperm(height(data), test_count);


% Jeżeli dane z wiersza mają trafić do zbioru testującego to w 7 kolumnie
% tabeli ustawiamy flage->1
for k = 1:test_count
    data(k, 7)= 1;
end


data_test= zeros(test_count, width(data)-1);
data_train= zeros(height(data)- test_count, width(data)-1);
a=1;
b=1;
for k = 1:height(data)
    if(data(k, 7)==1)
        data_test(a, :)= data(k,1:6);
        a=a+1;
    elseif(data(k, 7)==0)
       data_train(b, :)= data(k,1:6); 
       b=b+1;
    end
end


%% Macierze wag w sieci

% Macierz współczynników między wejściem a warstwą ukrytą
% Początkowo liczby pseudolosowe z przedziału 0-1
w1 = rand([hidden_size,input_size]);

% Macierz współczynników między warstwą ukrytą a wyjściem 
% Początkowo liczby pseudolosowe z przedziału 0-1
w2 = rand([output_size, hidden_size]);

%% Uczenie
cycles = 600;
[w1, w2, errors_train, errors_test, calculated_y] = train(data_train, num_feat, w1, w2, cycles, data_test);
figure(1)
subplot(1, 2, 1)
plot(errors_train)
title('Błąd dla danych treningowych');
xlabel('iteracje');
ylabel('wartość błędu');
subplot(1, 2, 2)
plot(errors_test)
title('Błąd dla danych testujących');
xlabel('iteracje');
ylabel('wartość błędu');

%% Wyznaczanie krzywych ROC
% true - 1; false - 0
calculated_y = calculated_y';
thresholds = linspace(0, 1, 100);
sensitivity_train = zeros(length(thresholds), 1);
specificity_train = zeros(length(thresholds), 1);
thresholded_train_y = zeros(height(data_train), 1);

for j = 1:length(thresholds)
    TP = 0; FP = 0; TN = 0; FN = 0;
    for sample = 1:height(calculated_y)
        if calculated_y(sample) < thresholds(j)
            thresholded_train_y(sample) = 0;
        else
            thresholded_train_y(sample) = 1;
        end
        if thresholded_train_y(sample) == 1 && thresholded_train_y(sample) == data_train(sample, num_feat+1)
            TP = TP + 1;
        elseif thresholded_train_y(sample) == 1 && thresholded_train_y(sample) ~= data_train(sample, num_feat+1)
            FP = FP + 1;
        elseif thresholded_train_y(sample) == 0 && thresholded_train_y(sample) ~= data_train(sample, num_feat+1)
            FN = FN + 1;
        elseif thresholded_train_y(sample) == 0 && thresholded_train_y(sample) == data_train(sample, num_feat+1)
            TN = TN + 1;
        end
    end
    sensitivity_train(j) = TP/(TP+FN);
    specificity_train(j) = TN/(FP+TN);
    FPR(j) = 1 - specificity_train(j);
end
figure(2)
plot(FPR, sensitivity_train)
title('Krzywa ROC');
ylabel('czułość');
xlabel('1 - swoistość');

figure(3)
plot(thresholds, sensitivity_train);
hold on
xlabel('Próg decycyjny');
plot(thresholds, specificity_train);
title('Czułość i specyficzność sieci');
legend('Czułość', "Specyficzność")

% Szukanie progu decyzyjnego

min_d = 100;
best_threshold_index = [];

for j = 1:length(thresholds) 
    d = sqrt((0 - FPR(j))^2 + (1 - sensitivity_train(j))^2);
    if d < min_d
        min_d = d;
        best_threshold_index = j;
    end
end

best_treshold = thresholds(1, best_threshold_index);

%% Przewidywanie
predicted_y = zeros(height(data_test), 1);
for sample = 1:height(data_test)
    predicted_y(sample, 1) = predict(data_test(sample, 1:num_feat), w1, w2);
end

% Wyznaczanie czułości i specyficzności sieci

thresholded_test_y = zeros(height(data_test), 1);

TP = 0; FP = 0; TN = 0; FN = 0;
for sample = 1:height(predicted_y)
    if predicted_y(sample) < best_treshold
        thresholded_test_y(sample) = 0;
    else
        thresholded_test_y(sample) = 1;
    end
    if thresholded_test_y(sample) == 1 && thresholded_test_y(sample) == data_test(sample, num_feat+1)
        TP = TP + 1;
    elseif thresholded_test_y(sample) == 1 && thresholded_test_y(sample) ~= data_test(sample, num_feat+1)
        FP = FP + 1;
    elseif thresholded_test_y(sample) == 0 && thresholded_test_y(sample) ~= data_test(sample, num_feat+1)
        FN = FN + 1;
    elseif thresholded_test_y(sample) == 0 && thresholded_test_y(sample) == data_test(sample, num_feat+1)
        TN = TN + 1;
    end
end
sensitivity_test = TP/(TP+FN);
specificity_test = TN/(FP+TN);
FPR = 1 - specificity_test;



%% Funkcje

% funkcja aktywacji- sigmoidalna funkcja unipolarna
function y = sigmoid(x)
y = zeros(height(x),1);
    for j=1:width(x)
        for k = 1:height(x)
        y(k,j) = 1/(1+exp(-x(k,j)));
        end
    end
end

% pochodna funkcji aktywacji
function y = sigmoidprim(x)
    for j=1:width(x)
        for k = 1:height(x)
            y(k,j) = sigmoid(x(k,j)) * (1-sigmoid(x(k,j)));
        end
    end
end

% funkcja do liczenia wyjścia
% sample - wiersz
function [y, v1, y1, v2] = feedforward(sample, num_feat, w1, w2)
    x = sample(1:num_feat)';
    v1 = w1 * x;
    y1 = sigmoid(v1);
    v2 = w2 * y1;
    y = sigmoid(v2);
end

% funkcja błędu
function e = error(d, y)
    e = d - y;
end

% propagacja błędu
function [w1, w2] = backpropagation(e, w1, w2, x, v1, y1, v2)
    delta = sigmoidprim(v2) .* e;
    e_hidden = w2' * delta;
    delta_hidden = sigmoidprim(v1) .* e_hidden;
    % współczynnik uczenia
    alfa = 0.01;
    w2 = w2 + alfa * (delta * y1');
    w1 = w1 + alfa * (delta_hidden * x);
end

% trenowanie

function [w1, w2, errors_train, errors_test, outputs] = train(data_train, num_feat, w1, w2, epochs, data_test)
    errors_train = zeros(epochs, 1);
    errors_test = zeros(epochs, 1);
    for epoch = 1:epochs
        for j = 1:height(data_train)
            [output, v1, y1, v2] = feedforward(data_train(j, :), num_feat, w1, w2);
            e = error(data_train(j, num_feat+1), output);
            [w1, w2] = backpropagation(e, w1, w2, data_train(j, 1:num_feat), v1, y1, v2);
        end
         for j = 1:height(data_train)
            [output, ~, ~, ~] = feedforward(data_train(j, :), num_feat, w1, w2);
            e = error(data_train(j, num_feat+1), output);
            errors_train(epoch) = errors_train(epoch) + 0.5 * e^2;
            outputs(j) = output;
         end
         errors_train(epoch) = errors_train(epoch) / height(data_train);

         for j = 1:height(data_test)
            [output, ~, ~, ~] = feedforward(data_test(j, :), num_feat, w1, w2);
            e = error(data_test(j, num_feat+1), output);
            errors_test(epoch) = errors_test(epoch) + 0.5 * e^2;
         end
         errors_test(epoch) = errors_test(epoch) / height(data_test);
    end
end

% przewidywanie

function output = predict(x, w1, w2)
    [output, ~, ~, ~] = feedforward(x, width(x), w1, w2);
end
