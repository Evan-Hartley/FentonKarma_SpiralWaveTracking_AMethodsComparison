import numpy as np
import matplotlib.pyplot as plt
import glob
import os

grid_size = (512, 512)

# Create filename list
filename = "../Results/Test/vlt_vals_t700.bin"


# Load data
data = np.fromfile(filename, dtype=np.double).reshape(grid_size)

# Create plot
fig, ax = plt.subplots()
cax = ax.imshow(data, cmap='rainbow', interpolation='nearest', vmin=-0.2, vmax=1.2)
fig.colorbar(cax)
ax.set_title("Voltage at t=700ms")

# Save plot
try:
    os.mkdir("../Figures/") # Creates a single directory
except FileExistsError:
    print(f"Directory already exists.\n")
except PermissionError:
    print(f"Permission denied: Unable to create directory.\n")
except Exception as e:
    print(f"An error occurred: {e}\n")
plt.savefig("../Figures/fk_spiral.png")
print("Saved Figure.\n")
plt.close()