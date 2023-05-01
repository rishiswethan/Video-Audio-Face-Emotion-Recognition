import torch
import gc

class Callbacks:
    def __init__(
            self,
            model_save_path=None,
    ):
        self._model_save_path = model_save_path

        # model_checkpoint
        self._checkpoint_last_best = None

        # reduce_lr_on_plateau
        self._reduce_lr_last_best = None
        self._reduce_lr_last_lr = None
        self._reduce_lr_num_bad_epochs = 0

        # early_stopping
        self._early_stopping_last_best = None
        self._early_stopping_num_bad_epochs = 0

    def model_checkpoint(
                self,
                model,
                model_save_path=None,
                monitor_value=None,
                mode="max",
                verbose=1,
                indicator_text="model_checkpoint()"
    ):
        """
        Saves the model after each epoch if the monitored metric improved.

        :param model_save_path: path to save the model
        :param monitor_value: metric to monitor
        :param mode: max or min
        :param verbose: verbosity
        :param save_best_only: save only the best model
        """

        if mode == "max":
            if self._checkpoint_last_best is None:
                self._checkpoint_last_best = 0
        elif mode == "min":
            if self._checkpoint_last_best is None:
                self._checkpoint_last_best = 100000
        else:
            raise ValueError("mode must be either max or min")

        if self._model_save_path is None:
            if model_save_path is None:
                raise ValueError("model_save_path must be provided in the constructor or in the function call")

            self._model_save_path = model_save_path

        if (mode == "max" and self._checkpoint_last_best < monitor_value) or \
                (mode == "min" and self._checkpoint_last_best > monitor_value):
            if verbose:
                print(f'{indicator_text} monitor value improved from {self._checkpoint_last_best} to {monitor_value}')

            self._checkpoint_last_best = monitor_value
            torch.save(model, self._model_save_path)
        else:
            if verbose:
                print(f'{indicator_text} monitor value did not improve from {self._checkpoint_last_best}')

    def reduce_lr_on_plateau(
            self,
            optimizer,
            monitor_value=None,
            mode="max",
            factor=0.1,
            patience=10,
            verbose=1,
            min_lr=0,
            min_delta=0,
            indicator_text="reduce_lr_on_plateau()"
    ):
        """
        Reduce learning rate when a metric has stopped improving.
        Models often benefit from reducing the learning rate by a factor of 2-10 once learning stagnates.
        This callback monitors a quantity and if no improvement is seen for a 'patience' number of epochs,
        the learning rate is reduced.

        :param optimizer: optimizer to use
        :param monitor_value: metric to monitor
        :param mode: max or min
        :param factor: factor by which the learning rate will be reduced. new_lr = lr * factor
        :param patience: number of epochs with no improvement after which learning rate will be reduced.
        :param verbose: verbosity
        :param min_lr: lower bound on the learning rate
        """

        if mode == "max":
            if self._reduce_lr_last_best is None:
                self._reduce_lr_last_best = 0
        elif mode == "min":
            if self._reduce_lr_last_best is None:
                self._reduce_lr_last_best = 100000
        else:
            raise ValueError("mode must be either max or min")

        if self._reduce_lr_last_best is None:
            self._reduce_lr_last_best = monitor_value
        if self._reduce_lr_last_lr is None:
            self._reduce_lr_last_lr = optimizer.param_groups[0]['lr']

        # check if the monitored metric improved
        improved_flag = False
        if (mode == "max" and self._reduce_lr_last_best < monitor_value):
            if monitor_value - self._reduce_lr_last_best > min_delta:
                improved_flag = True
            else:
                improved_flag = False
        elif (mode == "min" and self._reduce_lr_last_best > monitor_value):
            if self._reduce_lr_last_best - monitor_value > min_delta:
                improved_flag = True
            else:
                improved_flag = False

        # update the last best value or reduce the learning rate if the monitored metric did not improve for a while
        if improved_flag:
            self._reduce_lr_last_best = monitor_value
            self._reduce_lr_num_bad_epochs = 0
            if verbose:
                print(f'{indicator_text} monitor value improved from {self._reduce_lr_last_best} to {monitor_value}')

        else:
            self._reduce_lr_num_bad_epochs += 1
            if self._reduce_lr_last_lr * factor > min_lr:
                if self._reduce_lr_num_bad_epochs >= patience:
                    self._reduce_lr_last_lr = self._reduce_lr_last_lr * factor
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = self._reduce_lr_last_lr
                    self._reduce_lr_num_bad_epochs = 0
                    if verbose:
                        print(f'{indicator_text} lr reduced to {self._reduce_lr_last_lr}. monitor value did not improve for {patience} epochs from {self._reduce_lr_last_best}')
                else:
                    if verbose:
                        print(f'{indicator_text} monitor value did not improve for {self._reduce_lr_num_bad_epochs} epochs from {self._reduce_lr_last_best}. Waiting for {patience - self._reduce_lr_num_bad_epochs} epochs more.')


    def early_stopping(
            self,
            monitor_value=None,
            patience=10,
            mode="max",
            verbose=1,
            indicator_text="early_stopping()"
    ):
        """
        Stop training when a monitored metric has stopped improving.

        :param model_save_path: path to save the model
        :param monitor_value: metric to monitor
        :param patience: number of epochs with no improvement after which training will be stopped.
        :param mode: max or min
        :param verbose: verbosity
        :return: True if training should be stopped, False otherwise
        """

        # initialize the last best value
        if mode == "max":
            if self._early_stopping_last_best is None:
                self._early_stopping_last_best = 0
        elif mode == "min":
            if self._early_stopping_last_best is None:
                self._early_stopping_last_best = 100000
        else:
            raise ValueError("mode must be either max or min")

        # check if the monitored metric improved and save the model if it did
        if (mode == "max" and self._early_stopping_last_best < monitor_value) or \
                (mode == "min" and self._early_stopping_last_best > monitor_value):
            if verbose:
                print(f'{indicator_text} monitor value improved from {self._early_stopping_last_best} to {monitor_value}')

            self._early_stopping_last_best = monitor_value
            self._early_stopping_num_bad_epochs = 0

            return False
        else:
            self._early_stopping_num_bad_epochs += 1

            if self._early_stopping_num_bad_epochs >= patience:
                if verbose:
                    print(f'{indicator_text} monitor value did not improve for {patience} epochs. Stopping training.')
                return True
            else:
                if verbose:
                    print(f'{indicator_text} monitor value did not improve for {self._early_stopping_num_bad_epochs} epochs from {self._early_stopping_last_best}. Waiting for {patience - self._early_stopping_num_bad_epochs} epochs more.')
                return False


    def clear_memory(self):
        """
        Clears the memory
        """

        torch.cuda.empty_cache()
        gc.collect()
