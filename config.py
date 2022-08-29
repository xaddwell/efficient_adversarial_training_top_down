
root_dir = r'D:\cjh\Adversarial_training/'
model_root_dir = r'D:\cjh\Mask_Generator\weight\pretrained/'
benign_dataset = r'D:\cjh\Mask_Generator\datasets\ImageNet/'
adv_datasets = r'D:\cjh\Mask_Generator\datasets\generated/'
save_weight = r"D:\cjh\Adversarial_training\weight/"
log_path = r'D:\cjh\Adversarial_training\log/'
train_datasets_dir = root_dir + r"\datasets/"

save_epoch_step = 5
training_lr = 0.001
training_weight_decay = 1e-4
shuffle_training_dataset = True
random_seed = 42
train_batch_size = 128
validation_batch_size = 64
generate_data_batch_size = 128
num_workers = 2

alpha1 = 1
alpha2 = 100
train_epochs = 100





