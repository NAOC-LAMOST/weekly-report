# Weekly Report 2024-04

## 2024.04.15 - 2024.04.21

### Paper revision

1. `Detection performance` subsection
    - Added introduction of metrics (R, P, F1, AP, AUC, IoU, Dice).
    - Modified description of metrics table to highlight the P, AP, IoU and Dice metrics.
    - Added confusion metrix comparasion for three models.
2. `NaN Mask` subsection
    - Removed paragraphs on data padding for leading to misunderstanding easily.
    - Focused on automated handling of NaN values within light curves instead of manually handling them in previous works.
3. `Data and Training` subsection
    - Figure of loss & dice during training & valuation.
    - Specify the total training time, device, config, etc.
4. `Application` section
    - Deleted comparasion of Yang's.
    - Deleted figure of flare events samples.

### Code Recording

#### Loss (or other metrics) with smoothing.

```python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Function to smooth the curves with boundary adjustments
def smooth_with_padding(y, box_pts):
    box = np.ones(box_pts) / box_pts
    # Padding the start and end of the data to prevent boundary issues
    y_padded = np.pad(y, pad_width=(box_pts//2, box_pts//2), mode='edge')
    y_smooth = np.convolve(y_padded, box, mode='valid')
    return y_smooth

# Load each CSV file which is export from tensorboard
train_loss = pd.read_csv('./train_loss.csv')
train_dice = pd.read_csv('./train_dice.csv')
val_dice = pd.read_csv('./val_dice.csv')
val_loss = pd.read_csv('./val_loss.csv')

# Applying the adjusted smoothing function
train_loss_smooth_pad = smooth_with_padding(train_loss['Value'], 1)
val_loss_smooth_pad = smooth_with_padding(val_loss['Value'], 3)
train_dice_smooth_pad = smooth_with_padding(train_dice['Value'], 1)
val_dice_smooth_pad = smooth_with_padding(val_dice['Value'], 3)

# Resetting the rcParams to default before applying new font settings
plt.rcdefaults()

# Customizing font sizes for better readability in the adjusted figure size
plt.rc('font', family='Times New Roman', size=12) # Set the font family and size
plt.rc('axes', titlesize=9)  # fontsize of the axes title
plt.rc('axes', labelsize=8)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=7)  # fontsize of the tick labels
plt.rc('ytick', labelsize=7)  # fontsize of the tick labels
plt.rc('legend', fontsize=8)  # legend fontsize

# Setting the figure size to 8cm x 8cm
fig, ax = plt.subplots(2, 1, figsize=(3.15, 3.15), dpi=300)

# Adjusting the subplot parameters for a better layout
plt.subplots_adjust(left=0.15, right=0.95, top=0.92, bottom=0.08, hspace=0.6)

# Loss plot (training and validation) with adjusted smoothing
ax[0].plot(train_loss['Step'], train_loss_smooth_pad, label='Training Loss', color='blue', lw=0.8)
ax[0].plot(val_loss['Step'], val_loss_smooth_pad, label='Validation Loss', color='red', lw=0.8)
ax[0].set_title('(a) Training and Validation Loss')
ax[0].set_xlabel('Step')
ax[0].set_ylabel('Loss')
ax[0].legend()

# Dice plot (training and validation) with adjusted smoothing
ax[1].plot(train_dice['Step'], train_dice_smooth_pad, label='Training Dice', color='green', lw=0.8)
ax[1].plot(val_dice['Step'], val_dice_smooth_pad, label='Validation Dice', color='purple', lw=0.8)
ax[1].set_title('(b) Training and Validation Dice')
ax[1].set_xlabel('Step')
ax[1].set_ylabel('Dice')
ax[1].legend()

# Saving the figure as a PDF file
output_pdf_path = "./training_validation_curves_spacing.pdf"
fig.savefig(output_pdf_path, bbox_inches='tight', format='pdf')
plt.show()
```

结果展示：
![train loss & dice](./images/24-04/fcn4flare_train_loss.png)

#### Confusion Matrix for multi-models

```python
import matplotlib.pyplot as plt
import numpy as np

# Assume there are three different confusion matrices
confusion_matrices = [
    np.array([
        [18192458, 1134514],
        [98918, 21902]
    ]),
    np.array([
        [17240459, 2086513],
        [58910, 61910]
    ]),
    np.array([
        [20581284, 30435],
        [40578, 76218]])
]

# Adjust figsize for better visibility and increase subplot width
fig, axes = plt.subplots(1, 3, figsize=(14/2.54, 6/2.54), dpi=300, sharey=True)  # 1 row, 3 columns
cmap = plt.cm.Blues

# Titles for each subplot
titles = ['(a) Flatwrm2', '(b) Stellar', '(c) FCN4Flare']

for ax, confusion, title in zip(axes, confusion_matrices, titles):
    # Calculate percentage for color representation
    row_sums = confusion.sum(axis=1)
    percent_confusion = confusion / row_sums[:, np.newaxis]

    # Draw the confusion matrix
    cax = ax.imshow(percent_confusion, interpolation='nearest', cmap=cmap)
    ax.set_title(title, fontsize=9)  # Set individual titles

    tick_marks = np.arange(len(['False', 'True']))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(['False', 'True'], fontsize=8)  # Adjust tick label size
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(['False', 'True'], fontsize=8, rotation=45)

    # Add annotations
    thresh = percent_confusion.max() / 2.
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            percentage = f"{percent_confusion[i, j]:.2%}"
            absolute = f"{confusion[i, j]:,}"
            ax.text(j, i, f"{absolute}\n({percentage})", ha="center", 
                    va="center", fontsize=7, color="white" if percent_confusion[i, j] > thresh else "black")

# Add a common color bar
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])  # Position for colorbar
fig.colorbar(cax, cax=cbar_ax)

# Set common labels for axes
fig.text(0.5, 0.1, 'Predicted', ha='center', va='center', fontsize=9)
fig.text(0.05, 0.5, 'Ground Truth', ha='center', va='center', rotation='vertical', fontsize=9)

# Saving the figure as a PDF file
output_pdf_path = "./confusion_matrix.pdf"
fig.savefig(output_pdf_path, bbox_inches='tight', format='pdf')
plt.show()
```

结果展示：
![confusion matrix](./images/24-04/fcn4flare_confusion_matrix.png)

## 2024.04.15

### 【本周工作总结】

1. AIDA 会议，收获最大的报告：刘超老师。
2. FCN4Flare 论文 performance 章节撰写。

### 【下周工作计划】

1. 论文修改。