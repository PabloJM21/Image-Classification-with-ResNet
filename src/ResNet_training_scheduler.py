# We change run_training to introduce the scheduler
from setup import train, validate 

def run_training(model, optimizer, loss_function, scheduler, device, num_epochs, 
                train_dataloader, val_dataloader, early_stopper=None, verbose=False):
    """Run model training.

    Args:
        model (nn.Module): Torch model to train
        optimizer: Torch optimizer object
        loss_fn: Torch loss function for training
        device (torch.device): Torch device to use for training
        num_epochs (int): Max. number of epochs to train
        train_dataloader (DataLoader): Torch DataLoader object to load the
            training data
        val_dataloader (DataLoader): Torch DataLoader object to load the
            validation data
        early_stopper (EarlyStopper, optional): If passed, model will be trained
            with early stopping. Defaults to None.
        verbose (bool, optional): Print information about model training. 
            Defaults to False.

    Returns:
        list, list, list, list, torch.Tensor shape (10,10): Return list of train
            losses, validation losses, train accuracies, validation accuracies
            per epoch and the confusion matrix evaluated in the last epoch.
    """
    start_time = time.time()
    master_bar = fastprogress.master_bar(range(num_epochs))
    train_losses, val_losses, train_accs, val_accs = [],[],[],[]

    for epoch in master_bar:
        # Train the model
        epoch_train_loss, epoch_train_acc = train(train_dataloader, optimizer, model, 
                                                  loss_function, device, master_bar)
        # Validate the model
        epoch_val_loss, epoch_val_acc, confusion_matrix = validate(val_dataloader, 
                                                                   model, loss_function, 
                                                                   device, master_bar)

        # Save loss and acc for plotting
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accs.append(epoch_train_acc)
        val_accs.append(epoch_val_acc)
        
        # Update the learning rate scheduler
        scheduler.step()
        
        if verbose:
            master_bar.write(f'Train loss: {epoch_train_loss:.2f}, val loss: {epoch_val_loss:.2f}, train acc: {epoch_train_acc:.3f}, val acc {epoch_val_acc:.3f}')
            
        if early_stopper:
            early_stopper.update(epoch_val_acc, model)
            if early_stopper.early_stop:
                model = early_stopper.load_checkpoint(model)
                print('early stopping')
                break
            # END OF YOUR CODE #
            
    time_elapsed = np.round(time.time() - start_time, 0).astype(int)
    print(f'Finished training after {time_elapsed} seconds.')
    return train_losses, val_losses, train_accs, val_accs, confusion_matrix

# We train and plot with step size 1, gamma 0.1, 10 epochs

resnet_model2 = ResNet(ResidualBlock, [2, 2])
resnet_model2.to(device)

initial_lr = 0.1
optimizer = optim.Adam(resnet_model2.parameters(), lr=initial_lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

loss_fn = nn.CrossEntropyLoss()
n_epochs = 10
# early_stopper = setup.EarlyStopper(patience=10)
train_losses, val_losses, train_accs, val_accs, confusion_matrix = setup.run_training(model=resnet_model2, optimizer=optimizer, loss_function=loss_fn, 
                                                                                      device=device, num_epochs=n_epochs, 
                                                                                      train_dataloader=train_loader, val_dataloader=val_loader, 
                                                                                      early_stopper=None, verbose=False)

best_val_loss,best_val_loss_epoch = np.min(val_losses), np.argmin(val_losses)+1
best_val_acc, best_val_acc_epoch = np.max(val_accs), np.argmax(val_accs)+1

setup.plot(f'Loss vs. epoch (lr={lr}) (step_size:1, gamma:0.1)', 'Loss', train_losses, val_losses,
          extra_pt=(best_val_loss_epoch, best_val_loss), 
           extra_pt_label='Best Validation Loss')
plt.tight_layout()
print(f"Best Validation Loss (overall): {best_val_loss} @ Epoch no. {best_val_loss_epoch}")
setup.plot(f'Accuracy vs. epoch (lr={lr}) (step_size:1, gamma:0.1)', 'Accuracy', train_accs, val_accs,
           extra_pt=(best_val_acc_epoch, best_val_acc), 
           extra_pt_label='Best Validation Accuracy')
plt.tight_layout()
print(f"Best Validation Accuracy (overall): {best_val_acc} @ Epoch no. {best_val_acc_epoch}")

# We train and plot with step size 15, gamma 0.1, 30 epochs.



resnet_model4 = ResNet(ResidualBlock, [2, 2])
resnet_model4.to(device)

initial_lr = 0.1
optimizer = optim.Adam(resnet_model4.parameters(), lr=initial_lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

loss_fn = nn.CrossEntropyLoss()
n_epochs = 30
# early_stopper = setup.EarlyStopper(patience=10)
train_losses, val_losses, train_accs, val_accs, confusion_matrix = setup.run_training(model=resnet_model4, optimizer=optimizer, loss_function=loss_fn, 
                                                                                      device=device, num_epochs=n_epochs, 
                                                                                      train_dataloader=train_loader, val_dataloader=val_loader, 
                                                                                      early_stopper=None, verbose=False)

best_val_loss,best_val_loss_epoch = np.min(val_losses), np.argmin(val_losses)+1
best_val_acc, best_val_acc_epoch = np.max(val_accs), np.argmax(val_accs)+1

setup.plot(f'Loss vs. epoch (lr={initial_lr}) (step_size:15, gamma:0.1)', 'Loss', train_losses, val_losses,
          extra_pt=(best_val_loss_epoch, best_val_loss), 
           extra_pt_label='Best Validation Loss')
plt.tight_layout()
print(f"Best Validation Loss (overall): {best_val_loss} @ Epoch no. {best_val_loss_epoch}")
setup.plot(f'Accuracy vs. epoch (lr={initial_lr})(step_size:15, gamma:0.1)', 'Accuracy', train_accs, val_accs,
           extra_pt=(best_val_acc_epoch, best_val_acc), 
           extra_pt_label='Best Validation Accuracy')
plt.tight_layout()
print(f"Best Validation Accuracy (overall): {best_val_acc} @ Epoch no. {best_val_acc_epoch}")

# We train and plot with step size 15, gamma 0.9, 30 epochs.

resnet_model5 = ResNet(ResidualBlock, [2, 2])
resnet_model5.to(device)

initial_lr = 0.1
optimizer = optim.Adam(resnet_model5.parameters(), lr=initial_lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.9)

loss_fn = nn.CrossEntropyLoss()
n_epochs = 30
# early_stopper = setup.EarlyStopper(patience=10)
train_losses, val_losses, train_accs, val_accs, confusion_matrix = setup.run_training(model=resnet_model5, optimizer=optimizer, loss_function=loss_fn, 
                                                                                      device=device, num_epochs=n_epochs, 
                                                                                      train_dataloader=train_loader, val_dataloader=val_loader, 
                                                                                      early_stopper=None, verbose=False)

best_val_loss,best_val_loss_epoch = np.min(val_losses), np.argmin(val_losses)+1
best_val_acc, best_val_acc_epoch = np.max(val_accs), np.argmax(val_accs)+1

setup.plot(f'Loss vs. epoch (lr={initial_lr}) (step_size:15, gamma:0.9)', 'Loss', train_losses, val_losses,
          extra_pt=(best_val_loss_epoch, best_val_loss), 
           extra_pt_label='Best Validation Loss')
plt.tight_layout()
print(f"Best Validation Loss (overall): {best_val_loss} @ Epoch no. {best_val_loss_epoch}")
setup.plot(f'Accuracy vs. epoch (lr={initial_lr}) (step_size:15, gamma:0.9)', 'Accuracy', train_accs, val_accs,
           extra_pt=(best_val_acc_epoch, best_val_acc), 
           extra_pt_label='Best Validation Accuracy')
plt.tight_layout()
print(f"Best Validation Accuracy (overall): {best_val_acc} @ Epoch no. {best_val_acc_epoch}")
