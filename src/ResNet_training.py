# We train ResNet and plot loss and accuracy for training and validation set.
lr = 0.1
optimizer = optim.Adam(resnet_model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()
n_epochs = 30
# early_stopper = setup.EarlyStopper(patience=10)
train_losses, val_losses, train_accs, val_accs, confusion_matrix = setup.run_training(model=resnet_model, optimizer=optimizer, loss_function=loss_fn, 
                                                                                      device=device, num_epochs=n_epochs, 
                                                                                      train_dataloader=train_loader, val_dataloader=val_loader, 
                                                                                      early_stopper=None, verbose=False)

best_val_loss,best_val_loss_epoch = np.min(val_losses), np.argmin(val_losses)+1
best_val_acc, best_val_acc_epoch = np.max(val_accs), np.argmax(val_accs)+1

setup.plot(f'Loss vs. epoch (lr={lr})', 'Loss', train_losses, val_losses,
          extra_pt=(best_val_loss_epoch, best_val_loss), 
           extra_pt_label='Best Validation Loss')
plt.tight_layout()
print(f"Best Validation Loss (overall): {best_val_loss} @ Epoch no. {best_val_loss_epoch}")
setup.plot(f'Accuracy vs. epoch (lr={lr})', 'Accuracy', train_accs, val_accs,
           extra_pt=(best_val_acc_epoch, best_val_acc), 
           extra_pt_label='Best Validation Accuracy')
plt.tight_layout()
print(f"Best Validation Accuracy (overall): {best_val_acc} @ Epoch no. {best_val_acc_epoch}")
