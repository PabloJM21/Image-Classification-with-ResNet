# We define device
device = setup.get_device()
device

data_dir = 'data'
batch_size = 256

# We grab data, generate split and initialize data loaders
train_set, test_set = setup.grab_data(data_dir)
train_set, val_set = setup.generate_train_val_data_split(train_set, split_seed=42, val_frac=0.2)
train_loader, val_loader, test_loader = setup.init_data_loaders(train_set, val_set, test_set, batch_size)

# We define loss function
loss_fn = nn.CrossEntropyLoss()


