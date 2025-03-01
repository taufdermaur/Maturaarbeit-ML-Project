knob_weight = 0.5
input_01 = 0.5
goal_pred = 0.8

prediction = input_01 * knob_weight

error = (prediction - goal_pred) ** 2
print(error)