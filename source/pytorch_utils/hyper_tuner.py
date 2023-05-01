import random

import source.audio_analysis_utils.utils as utils
import csv
import os
from operator import itemgetter
import time
import numpy as np
import pickle

from hyperopt import hp, space_eval
from hyperopt.pyll.base import Apply
import itertools
from hyperopt.base import miscs_update_idxs_vals


class HyperTunerUtils:
    """
    This class is used to tune hyperparameters of a model using hyperopt

    How to use:
    1. Create a function that trains the model and returns the results
    2. Create a dictionary of hyperparameter ranges
    3. Create an instance of this class
    4. Create a dictionary of hyperopt hp objects using the return_full_hp_dict function
    5. Call the fmin function with the train_for_tuning function as the objective function

    Example:
        import pytorch_utils.hyper_tuner as pt_tuner
        from hyperopt import hp, STATUS_OK, fmin, tpe, space_eval, Trials

        tuner_utils = pt_tuner.HyperTunerUtils(
            best_hp_json_save_path=best_hp_json_save_path,
            tuner_csv_save_path=tuner_csv_save_path,
            tuner_obj_save_path=tuner_obj_save_path,
            tune_target=tune_target,
            tune_hp_ranges=search_space,
            max_trials=max_trials,
            train_function=train,
            load_if_exists=load_if_exists,
        )

        # Get the hp objects for each range in hyperopt
        search_space_hyperopt = tuner_utils.return_full_hp_dict(search_space)
        trials = Trials()

        best = fmin(
            tuner_utils.train_for_tuning,
            search_space_hyperopt,
            algo=rand.suggest,
            max_evals=tuner_utils.max_trials,
            trials=trials,
            trials_save_file=tuner_utils.tuner_obj_save_path,
            verbose=True,
            show_progressbar=False
        )

        print("Best: ", best)
        print(space_eval(search_space_hyperopt, best))

    """

    def __init__(
            self,
            best_hp_json_save_path,
            tuner_csv_save_path,
            tuner_obj_save_path,
            tune_target,
            tune_hp_ranges,
            max_trials,
            train_function,
            load_if_exists=False,
            ask_retune_confirmation=False,
            retune_if_already_tuned=False,
            randomise_grid_search=True,
            seed=None,
    ):
        """
        Parameters:
        ----------
        best_hp_json_save_path: str
            Path to save the best hyperparameters as a json file
        tuner_csv_save_path: str
            Path to save the tuner results as a csv file
        tuner_obj_save_path: str
            Path to save the tuner object as a pickle file
        tune_target: str
            The metric to use for tuning
        tune_hp_ranges: dict
            Dictionary of hyperparameter ranges
        max_trials: int
            The maximum number of trials to run
        train_function: function
            The function to use for training, given the hyperparameters as a dictionary
        load_if_exists: bool
            If True, load the tuner object if it exists
        ask_retune_confirmation: bool
            If True, ask the user if they really want to delete the existing tuner object, the csv file, and retune from scratch. Only applies if load_if_exists is True
        retune_if_already_tuned: bool
            If True, retune even if the results for this trail already exist in the csv file. Only applies if load_if_exists is True. Default is False.
        """

        self.best_hp_json_save_path = best_hp_json_save_path
        self.tuner_csv_save_path = tuner_csv_save_path
        self.sorted_csv_path = tuner_csv_save_path.replace(".csv", "_sorted.csv")
        self.tuner_obj_save_path = tuner_obj_save_path
        self.tune_target = tune_target
        self.tune_hp_ranges = tune_hp_ranges
        self.max_trials = max_trials
        self.train_function = train_function
        self.tune_cnt = 0
        self.start_time = time.time()
        self.load_if_exists = load_if_exists
        self.retune_if_already_tuned = retune_if_already_tuned
        self.randomise_grid_search = randomise_grid_search
        self.all_trail_details = []
        self.seed = seed

        if load_if_exists:
            # Load the previous trials from the csv file
            if not retune_if_already_tuned:
                self.all_trail_details = self.get_all_trail_details_from_csv()

        else:
            if ask_retune_confirmation:
                input("\nPress enter to delete the previous trials and start from scratch")

            # Delete the previous trials and start from scratch
            self.tune_cnt = 0
            if os.path.exists(tuner_obj_save_path):
                os.remove(tuner_obj_save_path)  # easy way to clear the trials if we want to start from scratch
            if os.path.exists(tuner_csv_save_path):
                os.remove(tuner_csv_save_path)
            if os.path.exists(self.sorted_csv_path):
                os.remove(self.sorted_csv_path)

    def train_for_tuning(self, kwargs):
        """
        Train the model for tuning. Basically the custom "optimise" function that should be passed to the hyperopt as mentioned in the docs.
        Parameters
        ----------
        kwargs - The hyperparameters to be tuned, in a dictionary format

        Returns
        -------
        The best value of the target metric that we want to tune for

        """
        def comp_2_dicts(dict1, dict2):
            """
            Compare two dictionaries by converting the values to strings and return True if they are equal
            """
            if len(dict1) != len(dict2):
                return False

            for key in dict1.keys():
                if key not in dict2.keys():
                    return False
                else:
                    if str(dict1[key]) != str(dict2[key]):
                        return False
            return True

        train_start_time = time.time()

        print(f"\n---> Tune count: {self.tune_cnt + 1} / {self.max_trials}")
        print("Kwargs: ", kwargs)

        opt_result = None

        if self.load_if_exists and not self.retune_if_already_tuned:
            # Check if the current hyperparameters are already tuned in the previous trials, based on the saved values in csv
            for trail_details in self.all_trail_details:
                trail_details_comp = trail_details.copy()
                trail_details_comp.pop(self.tune_target)

                # If the hyperparameters are already tuned, then skip this trail and return the saved value to opt_result
                if comp_2_dicts(kwargs, trail_details_comp):
                    print("\n\nAlready tuned, skipping...")
                    print("Trail details: ", trail_details_comp)
                    print(self.tune_target, ": ", trail_details[self.tune_target])

                    opt_result = trail_details[self.tune_target]
                    break

        # If the hyperparameters are not already tuned, then train the model
        if opt_result is None:
            opt_result = self.train_function(kwargs)

            # Save trial details to CSV
            trial_details = {**kwargs, self.tune_target: opt_result}
            self.save_trial_to_csv(trial_details, self.tuner_csv_save_path)

        # Save the best trail to a json file
        best_trail_details = self.get_best_trial_details_from_csv()
        print("\n----------------------------------")
        print(f"Best trial details: {best_trail_details}")
        utils.save_dict_as_json(self.best_hp_json_save_path, best_trail_details)

        # Display the time taken for training
        print(f"\nTraining time: {utils.get_minute_second_string(time.time() - train_start_time)}")
        print(f"Total time: {utils.get_minute_second_string(time.time() - self.start_time)}")
        print("----------------------------------\n")

        self.tune_cnt += 1

        return opt_result

    def save_trial_to_csv(
            self,
            trial_details: dict,
            tuner_csv_save_path=None
    ):
        """
        Save the trial details to the CSV file

        Parameters
        ----------
        trial_details - dict containing the trial details to be saved
        tuner_csv_save_path - path to the CSV file to save the trial details to

        """

        def keys_with_tuning_values(trial_details):
            """
            Return the keys of the trial details that are hyperparameters that have more than one value to tune
            """
            tune_keys = []
            for key, value in trial_details.items():
                if len(value[0]) > 1:
                    tune_keys.append(key)
            tune_keys.append(self.tune_target)
            return tune_keys

        if tuner_csv_save_path is None:
            tuner_csv_save_path = self.tuner_csv_save_path

        fieldnames = list(trial_details.keys())
        file_exists = os.path.isfile(tuner_csv_save_path)

        # Save the trial details to the original CSV file
        with open(tuner_csv_save_path, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            writer.writerow(trial_details)

        file_exists = os.path.isfile(self.sorted_csv_path)

        # Read the existing CSV file
        rows = []
        if file_exists:
            with open(self.sorted_csv_path, "r", newline="") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    row[self.tune_target] = float(row[self.tune_target])
                    rows.append(row)

        # Add the current trial details
        rows.append(trial_details)

        # Sort rows based on the tune_target column
        rows = sorted(rows, key=itemgetter(self.tune_target), reverse=True)

        # Get the keys of the trial details that are hyperparameters that have more than one value to tune
        fieldnames = keys_with_tuning_values(self.tune_hp_ranges)

        # Write the sorted rows to the new CSV file
        with open(self.sorted_csv_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()

            # Remove the tune values, if it doesn't have more than one tune value from the trial details before saving to CSV
            rows_with_tune_values = []
            for row in rows:
                row_with_tune_values = {}
                for key, value in row.items():
                    if key in fieldnames:
                        row_with_tune_values[key] = value
                rows_with_tune_values.append(row_with_tune_values)

            for row in rows_with_tune_values:
                writer.writerow(row)

    def _get_tune_cnt_from_csv(self, tuner_csv_save_path=None):
        """
        Get the number of trials that have been done so far. Useful when we want to continue tuning from where we left off.
        Parameters
        ----------
        tuner_csv_save_path - path to the CSV file to save the trial details to

        Returns
        -------
        The number of trials that have been done so far

        """

        if tuner_csv_save_path is None:
            tuner_csv_save_path = self.tuner_csv_save_path

        tune_cnt = 0
        if os.path.exists(tuner_csv_save_path):
            with open(tuner_csv_save_path, "r") as csvfile:
                reader = csv.DictReader(csvfile)
                tune_cnt = sum(1 for row in reader)

        return tune_cnt

    def get_best_trial_details_from_csv(self):
        """
        Get the best trial details from the CSV file in the form of a dictionary
        Returns
        -------
        The best trial details in the form of a dictionary

        """
        best_trial_details = None
        if os.path.exists(self.tuner_csv_save_path):
            with open(self.tuner_csv_save_path, "r") as csvfile:
                reader = csv.DictReader(csvfile)
                best_trial_details = max(reader, key=lambda row: float(row[self.tune_target]))

        # Convert the values to the correct data types
        for key, value in best_trial_details.items():
            if value.isdigit():
                best_trial_details[key] = int(value)
            elif value.replace('.', '', 1).isdigit():
                best_trial_details[key] = float(value)
            elif value == 'True':
                best_trial_details[key] = True
            elif value == 'False':
                best_trial_details[key] = False

        return best_trial_details

    def get_all_trail_details_from_csv(self):
        """
        Get all the trial details from the CSV file in the form of a dictionary
        """

        # Get all the trial details from the CSV file
        all_trial_details = []
        if os.path.exists(self.tuner_csv_save_path):
            with open(self.tuner_csv_save_path, "r") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    all_trial_details.append(row)

        # Convert the values to the correct data types
        for trial_details in all_trial_details:
            for key, value in trial_details.items():
                if value.isdigit():
                    trial_details[key] = int(value)
                elif value.replace('.', '', 1).isdigit():
                    trial_details[key] = float(value)
                elif value == 'True':
                    trial_details[key] = True
                elif value == 'False':
                    trial_details[key] = False

        print("All trial details: ", all_trial_details)
        return all_trial_details

    def set_hp_variable(
            self,
            hp_name: str,
            hp_ranges=None
    ):
        """
        This function is used to set the hyperparameter variables based on the hp_ranges dictionary
        Parameters
        ----------
        hp = keras tuner object
        hp_name = name of the hyperparameter(must be present in hp_ranges)
        hp_ranges = dictionary of hyperparameter ranges. If None, the default hp_ranges from the constructor is used

        Returns
        -------
        the set hyperparameter variable
        """

        if hp_ranges is None:
            hp_ranges = self.tune_hp_ranges

        hp_range, choice_type = hp_ranges[hp_name]
        if choice_type == 'choice':
            return hp.choice(hp_name, hp_range)

        elif choice_type == 'range':
            if len(hp_range) > 1:
                if type(hp_range[0]) == int:
                    hp_val = np.round(np.linspace(hp_range[0], hp_range[1], num=hp_range[2]).astype(int))
                    return hp.choice(hp_name, hp_val)

                elif type(hp_range[0]) == float:
                    hp_val = np.linspace(hp_range[0], hp_range[1], num=hp_range[2])
                    return hp.choice(hp_name, hp_val)

            else:
                return hp.choice(hp_name, [hp_range[0]])

    def return_full_hp_dict(self, hp_ranges=None):
        """
        This function is used to return the full hyperparameter dictionary
        Parameters
        ----------
        trail = optuna trial object
        hp_ranges = dictionary of hyperparameter ranges

        Returns
        -------
        the full hyperparameter dictionary
        """

        if hp_ranges is None:
            hp_ranges = self.tune_hp_ranges

        print("hp_ranges: ")
        hp_dict = {}
        for hp_name in hp_ranges.keys():
            hp_dict[hp_name] = self.set_hp_variable(hp_name)
            print(f"{hp_name}: {hp_ranges[hp_name]} -> {hp_dict[hp_name]}")

        return hp_dict

    def _recursive_find_nodes_grid(self, root, node_type='switch'):
        """
        Helper function for suggest_grid. Analyzes the domain instance to find nodes of a certain type.
        """

        nodes = []
        if isinstance(root, (list, tuple)):
            for node in root:
                nodes.extend(self._recursive_find_nodes_grid(node, node_type))
        elif isinstance(root, dict):
            for node in root.values():
                nodes.extend(self._recursive_find_nodes_grid(node, node_type))
        elif isinstance(root, (Apply)):
            if root.name == node_type:
                nodes.append(root)

            for node in root.pos_args:
                if node.name == node_type:
                    nodes.append(node)
            for _, node in root.named_args:
                if node.name == node_type:
                    nodes.append(node)
        return nodes

    def _parameters_grid(self, space):
        """
        Helper function for suggest_grid. Analyzes the domain instance to find parameters and their possible values.
        """

        # Analyze the domain instance to find parameters
        parameters = {}
        if isinstance(space, dict):
            space = list(space.values())
        for node in self._recursive_find_nodes_grid(space, 'switch'):
            # Find the name of this parameter
            paramNode = node.pos_args[0]
            assert paramNode.name == 'hyperopt_param'
            paramName = paramNode.pos_args[0].obj

            # Find all possible choices for this parameter
            values = [literal.obj for literal in node.pos_args[1:]]
            parameters[paramName] = np.array(range(len(values)))
        return parameters

    def suggest_grid(self, new_ids, domain, trials, seed):
        """
        A custom implementation of grid-search for hyperopt. Hyperopt does not support grid-search natively, and the algorithms supported by hyperopt may not be optimal for simply
        running through all the hyperparameter combinations.

        Feed this function to the hyperopt fmin function as the algo parameter to use grid-search

        Example:
            fmin(
            tuner_utils.train_for_tuning,
            search_space_hyperopt,
            algo=tuner_utils.suggest_grid,  # Use grid-search
            max_evals=tuner_utils.max_trials,
            trials=trials,
            trials_save_file=tuner_utils.tuner_obj_save_path,
            verbose=True,
            show_progressbar=False
            )

        """

        # Analyze the domain instance to find parameters
        params = self._parameters_grid(domain.expr)

        # Compute all possible combinations
        s = [[(name, value) for value in values] for name, values in params.items()]
        values = list(itertools.product(*s))

        # randomize the order of the values
        if self.seed is not None:
            np.random.seed(self.seed)
        if self.randomise_grid_search:
            np.random.shuffle(values)

        rval = []
        for i, new_id in enumerate(new_ids):
            # -- sample new specs, idxs, vals
            idxs = {name: np.array([new_id]) for name in params.keys()}
            vals = {name: np.array([value]) for name, value in values[new_id]}

            new_result = domain.new_result()
            new_misc = dict(tid=new_id, cmd=domain.cmd, workdir=domain.workdir)
            miscs_update_idxs_vals([new_misc], idxs, vals)

            new_trail = trials.new_trial_docs([new_id],
                                              [None], [new_result], [new_misc])

            rval.extend(new_trail)
        return rval
