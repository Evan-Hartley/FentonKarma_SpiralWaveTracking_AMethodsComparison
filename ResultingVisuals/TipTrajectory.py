import numpy as np
import matplotlib.pyplot as plt
import os


# Adjust variables to match run
grid_size = (512, 512)
results_folder_name = "../Results/Test/"
toggle_JDM = True
toggle_contour = True
toggle_phase = True
output_name = 'TipTrajectories'
delay = 400

# Construct plot
fig, ax = plt.subplots()

# If toggle_JDM is true, read data from files and plot the data (ignoring dummy values)
if toggle_JDM:
    filename_x_JDM = results_folder_name + "JDM_tip_x_tracker.bin"
    filename_y_JDM  = results_folder_name + "JDM_tip_y_tracker.bin"
    
    data_x_JDM  = np.fromfile(filename_x_JDM , dtype=np.int32)
    data_y_JDM  = np.fromfile(filename_y_JDM , dtype=np.int32)
    
    plot_x_JDM = []
    plot_y_JDM = []
    
    for ii in range(len(data_x_JDM)):
        if (ii >= delay and data_x_JDM[ii] != -1 and data_y_JDM[ii] != -1 and data_x_JDM[ii] != 0 and data_y_JDM[ii] != 0 and data_x_JDM[ii] < grid_size[0]  and data_y_JDM[ii] < grid_size[1]):
            plot_x_JDM.append(data_x_JDM[ii])
            plot_y_JDM.append(data_y_JDM[ii])
            
    ax.plot(plot_x_JDM,plot_y_JDM, color = 'b', label = 'Jacobian Determinate Method')

  # If toggle_contour is true, read data from files and plot the data (ignoring dummy values)  
if toggle_contour:
    filename_x_volt = results_folder_name + "volt_tip_x_tracker.bin"
    filename_y_volt  = results_folder_name + "volt_tip_y_tracker.bin"
    
    data_x_volt  = np.fromfile(filename_x_volt , dtype=np.double)
    data_y_volt = np.fromfile(filename_y_volt , dtype=np.double)
    
    plot_x_volt = []
    plot_y_volt = []
    
    for ii in range(len(data_x_volt)):
        if (ii >= delay and data_x_volt[ii] != -1 and data_y_volt[ii] != -1 and data_x_volt[ii] != 0 and data_y_volt[ii] != 0 and data_x_volt[ii] < grid_size[0]  and data_y_volt[ii] < grid_size[1]):
            plot_x_volt.append(data_x_volt[ii])
            plot_y_volt.append(data_y_volt[ii])
            
    ax.plot(plot_x_volt,plot_y_volt, color = 'k', alpha = 0.4, label = 'Contour Method')
    
# If toggle_phase is true, read data from files and plot the data (ignoring dummy values)
if toggle_phase:
    filename_x_phase = results_folder_name + "phase_tip_x_tracker.bin"
    filename_y_phase = results_folder_name + "phase_tip_y_tracker.bin"

    data_x_phase  = np.fromfile(filename_x_phase , dtype=np.double)
    data_y_phase  = np.fromfile(filename_y_phase , dtype=np.double)

    plot_x_phase = []
    plot_y_phase = []

    for ii in range(len(data_x_phase)):
        if (ii >= delay and data_x_phase[ii] != -1 and data_y_phase[ii] != -1 and data_x_phase[ii] != 0 and data_y_phase[ii] != 0 and data_x_phase[ii] < grid_size[0]  and data_y_phase[ii] < grid_size[1]):
            plot_x_phase.append(data_x_phase[ii])
            plot_y_phase.append(data_y_phase[ii])

    ax.plot(plot_x_phase,plot_y_phase, color = 'r', alpha = 0.4, label = 'Phase Method')
    
# Adjust plot settings and labels, then save and inform user
ax.set_title('Tip Trajectories', fontsize = 30)
ax.set_xlabel('x-space', fontsize = 24)
ax.set_ylabel('y-space', fontsize = 24)
ax.tick_params(labelsize = 16)
ax.legend(loc="upper right", bbox_to_anchor=(-.15, 1), fontsize=16, borderaxespad=0.)

try:
    os.mkdir("../Figures/") # Creates a single directory
except FileExistsError:
    print(f"Directory already exists.\n")
except PermissionError:
    print(f"Permission denied: Unable to create directory.\n")
except Exception as e:
    print(f"An error occurred: {e}\n")
plt.savefig("../Figures/" + output_name + ".png", bbox_inches='tight')
print("Saved graph")
plt.close()
