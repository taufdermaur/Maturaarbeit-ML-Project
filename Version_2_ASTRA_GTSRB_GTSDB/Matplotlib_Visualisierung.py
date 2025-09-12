import matplotlib.pyplot as plt

# =========================
# ASTRA Training Logs
# =========================
epochs_astra = list(range(1, 26))
train_acc_astra = [38.76, 62.81, 75.13, 82.14, 88.19, 91.37, 92.75, 94.09, 95.85, 96.03,
                   97.05, 97.65, 97.28, 97.46, 97.46, 98.85, 98.52, 98.57, 98.43, 98.71,
                   99.40, 98.89, 96.82, 98.66, 98.62]
val_acc_astra = [50.74, 73.06, 80.07, 84.13, 93.91, 94.65, 95.20, 97.97, 97.79, 99.26,
                 99.45, 99.63, 98.71, 98.89, 99.26, 98.71, 99.45, 97.97, 99.08, 99.45,
                 99.26, 98.71, 97.23, 99.08, 99.45]
train_loss_astra = [1.6793, 1.0492, 0.7082, 0.4737, 0.3433, 0.2467, 0.2094, 0.1681, 0.1335, 0.1222,
                    0.0949, 0.0677, 0.0908, 0.0773, 0.0828, 0.0344, 0.0441, 0.0415, 0.0454, 0.0410,
                    0.0255, 0.0322, 0.1327, 0.0532, 0.0437]
val_loss_astra = [1.2533, 0.7629, 0.5637, 0.3997, 0.2190, 0.1617, 0.1300, 0.0682, 0.0771, 0.0361,
                  0.0369, 0.0152, 0.0282, 0.0388, 0.0221, 0.0383, 0.0201, 0.0376, 0.0395, 0.0195,
                  0.0303, 0.0391, 0.0804, 0.0321, 0.0166]

# =========================
# GTSRB Training Logs
# =========================
epochs_gtsrb = list(range(1, 26))
train_acc_gtsrb = [50.57, 90.96, 95.38, 96.62, 97.18, 97.29, 97.59, 97.77, 97.96, 97.98,
                   97.96, 98.11, 98.04, 98.36, 98.23, 98.09, 98.50, 98.27, 98.35, 98.40,
                   98.23, 98.50, 98.51, 98.50, 98.64]
val_acc_gtsrb = [90.53, 97.56, 98.60, 98.41, 99.36, 99.25, 99.20, 99.31, 99.57, 99.31,
                 99.36, 99.38, 99.06, 99.38, 98.93, 99.58, 99.21, 98.95, 99.71, 99.40,
                 99.41, 99.71, 99.50, 99.31, 99.55]
train_loss_gtsrb = [1.6520, 0.2777, 0.1498, 0.1158, 0.0954, 0.0948, 0.0866, 0.0794, 0.0749, 0.0729,
                    0.0759, 0.0727, 0.0784, 0.0666, 0.0749, 0.0800, 0.0629, 0.0794, 0.0724, 0.0717,
                    0.0755, 0.0677, 0.0698, 0.0680, 0.0675]
val_loss_gtsrb = [0.3093, 0.0833, 0.0479, 0.0535, 0.0277, 0.0259, 0.0247, 0.0240, 0.0172, 0.0251,
                  0.0250, 0.0222, 0.0387, 0.0237, 0.0470, 0.0292, 0.0302, 0.0424, 0.0141, 0.0236,
                  0.0272, 0.0125, 0.0310, 0.0265, 0.0189]

# =========================
# Funktion für kombinierten Plot
# =========================
def plot_combined(epochs, train_acc, val_acc, train_loss, val_loss, titel):
    fig, ax1 = plt.subplots(figsize=(10,5))

    # Genauigkeit auf der linken Achse
    ax1.set_xlabel('Epoche', color='black')
    ax1.set_ylabel('Genauigkeit (%)', color='black')
    ax1.plot(epochs, train_acc, label='Trainingsgenauigkeit', color='tab:blue', marker='o', zorder=1)
    ax1.plot(epochs, val_acc, label='Validierungsgenauigkeit', color='tab:cyan', marker='o', zorder=1)
    ax1.tick_params(axis='y', colors='black')
    ax1.tick_params(axis='x', colors='black')
    ax1.set_ylim(0, 105)
    
    # Verlust auf der rechten Achse
    ax2 = ax1.twinx()
    ax2.set_ylabel('Verlust', color='black')
    ax2.plot(epochs, train_loss, label='Trainingsverlust', color='tab:red', marker='x', zorder=1)
    ax2.plot(epochs, val_loss, label='Validierungsverlust', color='tab:orange', marker='x', zorder=1)
    ax2.tick_params(axis='y', colors='black')
    
    # Titel
    plt.title(titel)
    
    # Legende kombinieren und zorder setzen
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(
        lines + lines2, 
        labels + labels2, 
        loc='center right', 
        facecolor='white', 
        edgecolor='black'
    )
    legend.set_zorder(10)  # Legende immer über den Linien
    
    # Gitter hinter alles setzen
    plt.grid(True, zorder=0)
    
    plt.tight_layout()
    plt.show()

# =========================
# ASTRA Plot
# =========================
plot_combined(epochs_astra, train_acc_astra, val_acc_astra, train_loss_astra, val_loss_astra, 'ASTRA-Modell - Genauigkeit- und Verlustkurven')

# =========================
# GTSRB Plot
# =========================
plot_combined(epochs_gtsrb, train_acc_gtsrb, val_acc_gtsrb, train_loss_gtsrb, val_loss_gtsrb, 'GTSRB-Modell - Genauigkeit- und Verlustkurven')
