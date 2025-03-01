weight = 0.5
goal_pred = 0.8
input = 0.5
for iteration in range(20):
    pred = input * weight
    error = (pred - goal_pred) ** 2
    direction_and_amount = (pred - goal_pred) * input
    weight = weight - direction_and_amount
    print("Fehler:" + str(error) + " Vorhersage:" + str(pred))