import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
pd.set_option('display.max_rows', None)


def transform_data(df):
    for i in range(0, len(df)):
        if isinstance(df['val_acc'][i], list) and len(df['val_acc'][i]) == 3:
            df.loc[i, 'acc'] = round(df.loc[i, 'val_acc'][0], 4)
            df.loc[i, 'val_acc'] = round(df.loc[i, 'val_acc'][2], 4) #df['val_acc'] = mIoU_value
        # print(type(df.loc[i, 'adjusted_mIoU']))
        df["adjusted_mIoU"] = df["adjusted_mIoU"].replace('', np.nan).astype(float)

            
        # else:
        #     df.loc[i, 'val_acc'] = None
        # if isinstance(df['val_loss'][i], float) == False:
        #     df['val_loss'][i] = None
    return df


def tmp_trans(df):
    for i in range(0, len(df)):
        if isinstance(df['other_mIoU'][i], list):
            df.loc[i, 'AMIoU_2'] = round(df.loc[i, 'other_mIoU'][0], 4)
        else:
            df.loc[i, 'val_acc'] = 0
            df.loc[i, 'val_loss'] = 0
        # else:
        #     df.loc[i, 'val_acc'] = None
        # if isinstance(df['val_loss'][i], float) == False:
        #     df['val_loss'][i] = None
    df = df.drop('other_mIoU', axis=1)
    return df


danet_100epoch = pd.read_pickle('training_data/DANet/Top_100_epoch.pkl')   # must load the right file name
danet_100epoch = danet_100epoch.set_index('epoch')
danet_100epoch = transform_data(danet_100epoch)
danet_100epoch = tmp_trans(danet_100epoch)
danet_acc = danet_100epoch['acc'].max()
print("-------------------------The metrix of Base-------------------------")
print(danet_100epoch)

# Baseline_100epoch_1 = pd.read_pickle('./training_data/Baseline_0.5821_100epoch.pkl')
# print(Baseline_100epoch_1)

tanet_100epoch = pd.read_pickle('training_data/TANet_QKV_has_VFA/Top_100_epoch.pkl')  # must load the right file name
tanet_100epoch = tanet_100epoch.set_index('epoch')
tanet_100epoch= transform_data(tanet_100epoch)
tanet_100epoch = tmp_trans(tanet_100epoch)
print("-------------------------The metrix of VFA-------------------------")
print(tanet_100epoch)

DSCAMSFF_TANet_100epoch = pd.read_pickle('./training_data/DSCAMSFF_TANet_QKV_has_VFA/Top_100_epoch.pkl')
DSCAMSFF_TANet_100epoch = DSCAMSFF_TANet_100epoch.set_index('epoch')
DSCAMSFF_TANet_100epoch = transform_data(DSCAMSFF_TANet_100epoch)
DSCAMSFF_TANet_100epoch = tmp_trans(DSCAMSFF_TANet_100epoch)
print("-------------------------The metrix of msf-------------------------")
print(DSCAMSFF_TANet_100epoch)

danet_100epoch["train_loss"] = danet_100epoch["train_loss"] / 744
danet_100epoch["val_loss"] = danet_100epoch["val_loss"] / 125
danet_100epoch["val_acc"] = danet_100epoch["val_acc"] * 100
danet_best_mIOU_idx = danet_100epoch['AMIoU_2'].idxmax()



tanet_100epoch["train_loss"] = tanet_100epoch["train_loss"] / 744
tanet_100epoch["val_loss"] = tanet_100epoch["val_loss"] / 125
tanet_100epoch["val_acc"] = tanet_100epoch["val_acc"] * 100
tanet_best_mIOU_idx = tanet_100epoch['AMIoU_2'].idxmax()


DSCAMSFF_TANet_100epoch["train_loss"] = DSCAMSFF_TANet_100epoch["train_loss"] / 744
DSCAMSFF_TANet_100epoch["val_loss"] = DSCAMSFF_TANet_100epoch["val_loss"] / 125
DSCAMSFF_TANet_100epoch["val_acc"] = DSCAMSFF_TANet_100epoch["val_acc"] * 100
DSCAMSFF_TANet_best_mIOU_idx = DSCAMSFF_TANet_100epoch['AMIoU_2'].idxmax()

# Baseline_100epoch['train_loss'].plot(linestyle='-', marker='.', linewidth=2)  # 修改线条样式
# only1_vf['train_loss'].plot(linestyle='-', marker='.', linewidth=2)  # 修改线条样式
# danet_100epoch['val_acc'].iloc[lambda x: x.index % 2 - 1 == 0].plot(linestyle='-', marker='.', linewidth=2)  # 修改线条样式
# tanet_100epoch['val_acc'].iloc[lambda x: x.index % 2 - 1 == 0].plot(linestyle='-', marker='.', linewidth=2)  # 修改线条样式
# DSCAMSFF_TANet_100epoch['val_acc'].iloc[lambda x: x.index % 2 - 1 == 0].plot(linestyle='-', marker='*', linewidth=2)  # 修改线条样式
# #train_df_14['val_acc'].plot(linestyle='-', marker='.', linewidth=2)  # 修改线条样式
# #train_df_15['val_acc'].plot(linestyle='-', marker='.', linewidth=2)  # 修改线条样式
# train_df_16['val_acc'].plot(linestyle='-', marker='.', linewidth=2)
# train_df_17['val_acc'].plot(linestyle='-', marker='.', linewidth=2)

# # # plt.style.use('ggplot')
# plt.xlabel('Epoch',  fontsize=18)  # 添加横轴标签
# # plt.ylabel('Validation Loss',  fontsize=18)  # 添加纵轴标签
# plt.ylabel('MIoU (%)',  fontsize=18)  # 添加纵轴标签
# # plt.ylabel('Adjusted MIoU (%)',  fontsize=18)  # 添加纵轴标签
# # plt.title('Arcuate Scotoma', fontsize=18)  # 添加标题

# # #plt.legend(['Baseline', 'Query', 'Key', 'Value'], loc='lower right')  # 设置图例位置
# # plt.legend(['Baseline', '0 & 1', '1 & 2', '1 & 5'], loc='lower right', prop={'size': 18})
# plt.legend(['Baseline', 'TANet', 'MSCA'], loc='lower right', prop={'size': 18})

# # plt.grid(True)  # 显示网格线

# plt.tight_layout()  # 调整布局，防止标签重叠
# # plt.legend(['Baseline', 'No VF', 'Only1', 'Only3', '1and2', '1and3', 'All VF'])
# # plt.savefig('./Thesis_plot/S44_MIoU.png', dpi=300)
# plt.show()


# print("Baseline best mIoU: %.4f, corresponding acc: %.4f, adjust_mIoU: %.4f" % (danet_100epoch['AMIoU_2'][danet_best_mIOU_idx],danet_100epoch['val_acc'][danet_best_mIOU_idx],danet_100epoch['adjusted_mIoU'][danet_best_mIOU_idx]))
# print("Only VFA best mIoU: %.4f, corresponding acc: %.4f, adjust_mIoU: %.4f" % (danet_100epoch['AMIoU_2'][tanet_best_mIOU_idx], danet_100epoch['val_acc'][tanet_best_mIOU_idx], danet_100epoch['adjusted_mIoU'][tanet_best_mIOU_idx]))
# print("MSCA VFA best mIoU: %.4f, corresponding acc: %.4f, adjust_mIoU: %.4f" % (DSCAMSFF_TANet_100epoch['AMIoU_2'][DSCAMSFF_TANet_best_mIOU_idx], DSCAMSFF_TANet_100epoch['val_acc'][DSCAMSFF_TANet_best_mIOU_idx], DSCAMSFF_TANet_100epoch['adjusted_mIoU'][DSCAMSFF_TANet_best_mIOU_idx]))

#AMIoU & MIoU both in one
for df in [danet_100epoch, tanet_100epoch, DSCAMSFF_TANet_100epoch]:
    transform_data(df)
    #tmp_trans(df)
    df["train_loss"] = df["train_loss"] / 744
    df["val_loss"] = df["val_loss"] / 125
    df["val_acc"] = df["val_acc"] * 100 # Convert to percentage
    df["adjusted_mIoU"] = df["adjusted_mIoU"] * 100  # Convert to percentage

# Create a 1x2 subplot figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot MIoU vs Epoch (left plot)
ax1.plot(
    danet_100epoch.index[danet_100epoch.index % 2 - 1 == 0],
    danet_100epoch['val_acc'].iloc[danet_100epoch.index % 2 - 1 == 0],
    linestyle='-', marker='.', linewidth=2, label='Baseline'
)
ax1.plot(
    tanet_100epoch.index[tanet_100epoch.index % 2 - 1 == 0],
    tanet_100epoch['val_acc'].iloc[tanet_100epoch.index % 2 - 1 == 0],
    linestyle='-', marker='.', linewidth=2, label='TANet'
)
ax1.plot(
    DSCAMSFF_TANet_100epoch.index[DSCAMSFF_TANet_100epoch.index % 2 - 1 == 0],
    DSCAMSFF_TANet_100epoch['val_acc'].iloc[DSCAMSFF_TANet_100epoch.index % 2 - 1 == 0],
    linestyle='-', marker='*', linewidth=2, label='MSCA'
)
ax1.set_xlabel('Epoch', fontsize=14)
ax1.set_ylabel('MIoU (%)', fontsize=14)
ax1.set_title('MIoU vs Epoch', fontsize=16)
ax1.legend(loc='lower right')
ax1.grid(True)

# Plot AMIoU vs Epoch (right plot)
ax2.plot(
    danet_100epoch.index[danet_100epoch.index % 2 - 1 == 0],
    danet_100epoch['adjusted_mIoU'].iloc[danet_100epoch.index % 2 - 1 == 0],
    linestyle='-', marker='.', linewidth=2, label='Baseline'
)
ax2.plot(
    tanet_100epoch.index[tanet_100epoch.index % 2 - 1 == 0],
    tanet_100epoch['adjusted_mIoU'].iloc[tanet_100epoch.index % 2 - 1 == 0],
    linestyle='-', marker='.', linewidth=2, label='TANet'
)
ax2.plot(
    DSCAMSFF_TANet_100epoch.index[DSCAMSFF_TANet_100epoch.index % 2 - 1 == 0],
    DSCAMSFF_TANet_100epoch['adjusted_mIoU'].iloc[DSCAMSFF_TANet_100epoch.index % 2 - 1 == 0],
    linestyle='-', marker='*', linewidth=2, label='MSCA'
)
ax2.set_xlabel('Epoch', fontsize=14)
ax2.set_ylabel('AMIoU (%)', fontsize=14)
ax2.set_title('Adjusted MIoU vs Epoch', fontsize=16)
ax2.legend(loc='lower right')
ax2.grid(True)

plt.tight_layout()
plt.savefig('MIoU_AMIoU_comparison.png', dpi=300)
plt.show()

# Print best results /100 since multiplied by 100 previously
print("Baseline best mIoU: %.4f%%, AMIoU: %.4f%%" % (
    danet_100epoch['val_acc'].max()/100,
    danet_100epoch['adjusted_mIoU'].max()
))
print("TANet best mIoU: %.4f%%, AMIoU: %.4f%%" % (
    tanet_100epoch['val_acc'].max()/100,
    tanet_100epoch['adjusted_mIoU'].max()
))
print("MSCA best mIoU: %.4f%%, AMIoU: %.4f%%" % (
    DSCAMSFF_TANet_100epoch['val_acc'].max()/100,
    DSCAMSFF_TANet_100epoch['adjusted_mIoU'].max()
))