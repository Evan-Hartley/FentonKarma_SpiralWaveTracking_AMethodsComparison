import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob
import os


# Change variables to match Simulation and Output parameters
grid_size = (512, 512)
results_folder_name = "../Results/Test/"
output_name_vlt = "vlt_vals_evolution"
output_name_fig = "fig_vals_evolution"
output_name_sig = "sig_vals_evolution"
plot_title_vlt = "Voltage Evolution"
plot_title_fig = "Fast Intake Gating Variable Evolution"
plot_title_sig = "Slow Intake Gating Variable Evolution"

# Create filename list
file_list_vlt = sorted(glob.glob(results_folder_name + "vlt_vals_t*.bin"), key=lambda x: int(''.join(filter(str.isdigit, x))))
file_list_fig = sorted(glob.glob(results_folder_name + "fig_vals_t*.bin"), key=lambda x: int(''.join(filter(str.isdigit, x))))
file_list_sig = sorted(glob.glob(results_folder_name + "sig_vals_t*.bin"), key=lambda x: int(''.join(filter(str.isdigit, x))))


# Load data
frames_vlt = []
for filename in file_list_vlt:
    data = np.fromfile(filename, dtype=np.double).reshape(grid_size)

    if data.shape == grid_size:
        frames_vlt.append(data)
    else:
        print(f"Skipping {filename}: shape {data.shape} != {grid_size}")

frames_fig = []
for filename in file_list_fig:
    data = np.fromfile(filename, dtype=np.double).reshape(grid_size)

    if data.shape == grid_size:
        frames_fig.append(data)
    else:
        print(f"Skipping {filename}: shape {data.shape} != {grid_size}")


frames_sig = []
for filename in file_list_sig:
    data = np.fromfile(filename, dtype=np.double).reshape(grid_size)

    if data.shape == grid_size:
        frames_sig.append(data)
    else:
        print(f"Skipping {filename}: shape {data.shape} != {grid_size}")

# Fill animation definition
def update(frame):
    cax.set_array(frame)
    return [cax]

# Create animation
fig, ax = plt.subplots()
cax = ax.imshow(frames_vlt[0], cmap='rainbow', interpolation='nearest', vmin=-0.2, vmax=1.2)
fig.colorbar(cax)
ax.set_title(plot_title_vlt)

# Compile animation
ani = animation.FuncAnimation(fig, update, frames=frames_vlt, interval=1000, blit=True)
try:
    os.mkdir("../Figures/") # Creates a single directory
except FileExistsError:
    print(f"Directory already exists.\n")
except PermissionError:
    print(f"Permission denied: Unable to create directory.\n")
except Exception as e:
    print(f"An error occurred: {e}\n")
ani.save("../Figures/" + output_name_vlt + ".gif", writer='pillow')
print("Saved gif\n")
plt.close()

# Create animation
fig, ax = plt.subplots()
cax = ax.imshow(frames_fig[0], cmap='rainbow', interpolation='nearest', vmin=-0.2, vmax=1.2)
fig.colorbar(cax)
ax.set_title(plot_title_fig)

# Compile and save animation
ani = animation.FuncAnimation(fig, update, frames=frames_fig, interval=1000, blit=True)
try:
    os.mkdir("../Figures/") # Creates a single directory
except FileExistsError:
    print(f"Directory already exists.\n")
except PermissionError:
    print(f"Permission denied: Unable to create directory.\n")
except Exception as e:
    print(f"An error occurred: {e}\n")
ani.save("../Figures/"+ output_name_fig + ".gif", writer='pillow')
print("Saved gif\n")
plt.close()

# Create animation
fig, ax = plt.subplots()
cax = ax.imshow(frames_sig[0], cmap='rainbow', interpolation='nearest', vmin=-0.2, vmax=1.2)
fig.colorbar(cax)
ax.set_title(plot_title_sig)

# Compile and save animation
ani = animation.FuncAnimation(fig, update, frames=frames_sig, interval=1000, blit=True)
ani.save("../Figures/" + output_name_sig + ".gif", writer='pillow')
print("Saved gif\n")
plt.close()
