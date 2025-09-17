import numpy as np
import matplotlib.pyplot as plt

# Erstelle die Daten für die x-Achse
x = np.linspace(-5, 5, 100)

# Berechne die ReLU-Funktion
relu = np.maximum(0, x)

# Berechne die Sigmoid-Funktion
sigmoid = 1 / (1 + np.exp(-x))

# --- Graph für die ReLU-Funktion ---
fig_relu, ax_relu = plt.subplots(figsize=(6, 4))
ax_relu.plot(x, relu, label='ReLU', color='blue')

# Setze die Achsenbeschriftungen
ax_relu.set_xlabel('x', fontsize=12, loc='right')
ax_relu.set_ylabel('f(x)', fontsize=12, loc='top', rotation=0)

# Achsen ohne Zahlen
ax_relu.spines['left'].set_position('zero')
ax_relu.spines['bottom'].set_position('zero')
ax_relu.spines['right'].set_color('none')
ax_relu.spines['top'].set_color('none')

# Beschrifte den Nullpunkt
ax_relu.text(-0.2, -0.2, '0', fontsize=10, ha='right', va='top')

# Pfeile an den Achsenenden
ax_relu.plot(1, 0, ">k", transform=ax_relu.get_yaxis_transform(), clip_on=False)
ax_relu.plot(0, 1, "^k", transform=ax_relu.get_xaxis_transform(), clip_on=False)
ax_relu.set_xticks([])
ax_relu.set_yticks([])

# --- Graph für die Sigmoid-Funktion ---
fig_sigmoid, ax_sigmoid = plt.subplots(figsize=(6, 4))
ax_sigmoid.plot(x, sigmoid, label='Sigmoid', color='red')

# Setze die Achsenbeschriftungen
ax_sigmoid.set_xlabel('x', fontsize=12, loc='right')
ax_sigmoid.set_ylabel('f(x)', fontsize=12, loc='top', rotation=0)

# Achsen ohne Zahlen
ax_sigmoid.spines['left'].set_position('zero')
ax_sigmoid.spines['bottom'].set_position('zero')
ax_sigmoid.spines['right'].set_color('none')
ax_sigmoid.spines['top'].set_color('none')

# Beschrifte den Nullpunkt
ax_sigmoid.text(-0.2, -0.05, '0', fontsize=10, ha='right', va='top')

# Pfeile an den Achsenenden
ax_sigmoid.plot(1, 0, ">k", transform=ax_sigmoid.get_yaxis_transform(), clip_on=False)
ax_sigmoid.plot(0, 1, "^k", transform=ax_sigmoid.get_xaxis_transform(), clip_on=False)
ax_sigmoid.set_xticks([])
ax_sigmoid.set_yticks([])

# Zeige die Plots an
plt.show()