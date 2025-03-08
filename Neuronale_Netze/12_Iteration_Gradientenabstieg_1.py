weight = 0.1
alpha = 0.01

def nnw(input, weight):
    prediction = input * weight
    return prediction

number_of_toes = [8.5]
win_or_lose_binary = [1] #win

input = number_of_toes[0]
goal_pred = win_or_lose_binary[0]

pred = nnw(input, weight)
error = (pred - goal_pred) ** 2

delta = pred - goal_pred
weight_delta = input * delta
weight -= weight_delta * alpha

print(weight)
print(delta)
print(pred)

print("/n")
print("/n")

weight, goal_pred, input = (0.0, 0.8, 0.5)

for iteration in range(4):
    error = ((input * weight) - goal_pred) ** 2
    delta = (input * weight) - goal_pred
    weight_delta = delta * input
    weight -= weight_delta
    print("Fehler:" + str(error) + " Vorhersage:" + str(input * weight))