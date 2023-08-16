import numpy as np
import pandas as pd
from tqdm import tqdm
from eval import evaluate_multi_split, METRICS
from utils import seed_generator

import time
import datetime
from data import FairDataset
from sklearn.dummy import DummyClassifier
from fairens import FairAugEnsemble
from fairlearn.postprocessing import ThresholdOptimizer


class Trainer:
    def __init__(self, metrics=METRICS['all'], random_state=None) -> None:
        """
        Initialize the Trainer object.

        Parameters:
            models (dict): A dictionary containing model names as keys and the corresponding model instances as values.
            datasets (dict): A dictionary containing dataset names as keys and FairDataset instances as values.
            random_state (int, optional): Seed used by the random number generator for reproducibility. Defaults to None.
        """
        self.random_state = random_state
        self.metrics = metrics
        self.name_max_len = {
            'model': 0,
            'dataset': 0,
        }

    def run_single_model(
        self,
        model,
        model_name,
        dataset: FairDataset,
        n_runs: int = 5,
        verbose: bool = False,
        pbar: bool = False,
        exception: str = 'print',
    ):
        """
        Run a single model on a given dataset multiple times and evaluate its performance and fairness metrics.

        Parameters:
            model: The trained predictive model to be evaluated.
            model_name (str): Name of the model for identification.
            dataset (FairDataset): The dataset on which the model will be evaluated.
            n_runs (int, optional): Number of times to run the model on the dataset. Defaults to 5.
            verbose (bool, optional): Whether to print detailed results during execution. Defaults to False.
            pbar (bool, optional): Whether to display a progress bar. Defaults to False.
            exception (str, optional): Whether to raise an exception if the model cannot be fitted on the dataset.

        Returns:
            pandas DataFrame: A DataFrame containing the evaluation results for each run of the model on the dataset.
        """
        seeds = seed_generator(self.random_state)
        results = {}
        iterable = tqdm(
            range(n_runs),
            disable=not pbar,
            desc=f"Data: {dataset.fullname:<{self.name_max_len['dataset']}s} | Model: {model_name:<{self.name_max_len['model']}s} ",
        )
        for i in iterable:
            seed = next(seeds)
            # update split data with new seed
            dataset.update_split(random_state=seed)
            (X_train, y_train, s_train) = dataset.split['train']
            # update based model with new seed
            try:
                model.set_params(random_state=seed)
            except:
                pass

            # train model
            try:
                try:
                    model.fit(X_train, y_train, sensitive_features=s_train)
                except:
                    model.fit(X_train, y_train)
            except Exception as e:
                if exception == 'ignore':
                    return None
                elif exception == 'raise':
                    raise ValueError(
                        f'Cannot fit mode {model} on dataset {dataset.fullname}: {e}'
                    )
                elif exception == 'print':
                    print(f'Cannot fit mode {model} on dataset {dataset.fullname}.')
                    return None
                else:
                    raise ValueError(
                        f'Invalid exception value: {exception}, must be "ignore", "raise", or "print".'
                    )

            # evaluate model
            results[i] = evaluate_multi_split(model, dataset.split, random_state=seed)

            # last run, compute mean and std of all runs
            if i + 1 == n_runs:
                results = pd.DataFrame(results).T
                df_res = pd.DataFrame()
                for splitname in results.columns:
                    for metric in self.metrics:
                        df_res[f'{splitname}_{metric}'] = results[splitname].map(
                            lambda x: x[metric]
                        )
                    # add the predicted advantage group
                    df_res[f'{splitname}_advg'] = results[splitname].map(
                        lambda x: x['g_adv']
                    )
                # add model and dataset info
                df_res['dataset'] = dataset.fullname
                if len(dataset.fullname) > self.name_max_len['dataset']:
                    self.name_max_len['dataset'] = len(dataset.fullname)
                df_res['model'] = model_name
                if len(model_name) > self.name_max_len['model']:
                    self.name_max_len['model'] = len(model_name)
                df_res['model_class'] = model.__class__.__name__
                if verbose:
                    # self.print_results(df_res, group_by='dataset', dataset_header=False)
                    info = self.get_single_model_data_results(df_res)
                    iterable.set_postfix_str(info)

        return df_res

    def run_all_models_single_data(self, models: dict, dataset: FairDataset, **kwargs):
        """
        Run all models on a single dataset and evaluate their performance and fairness metrics.

        Parameters:
            models (dict): A dictionary containing model names as keys and the corresponding model instances as values.
            dataset (FairDataset): The dataset on which the models will be evaluated.
            **kwargs: Additional arguments to be passed to the run_single_model function.

        Returns:
            pandas DataFrame: A DataFrame containing the evaluation results for each model on the given dataset.
        """
        models = self.models if models is None else models
        df_res = pd.DataFrame()
        for model_name, model in models.items():
            df_res = df_res.append(
                self.run_single_model(model, model_name, dataset, **kwargs)
            )
        return df_res

    def run(self, models: dict, datasets: dict, verbose_header: bool = True, **kwargs):
        """
        Run all models on all datasets and evaluate their performance and fairness metrics.

        Parameters:
            models (dict, optional): A dictionary containing model names as keys and the corresponding model instances as values.
                                     If None, uses the models provided during initialization. Defaults to None.
            datasets (dict, optional): A dictionary containing dataset names as keys and FairDataset instances as values.
                                       If None, uses the datasets provided during initialization. Defaults to None.
            **kwargs: Additional arguments to be passed to the run_single_model function.

        Returns:
            pandas DataFrame: A DataFrame containing the evaluation results for each model on all datasets.
        """
        assert (
            max([len(model_name.split('_')) for model_name in models.keys()]) <= 2
        ), "'_' is used to separate base model name and fair method, given model names have more than 1 '_'."
        if verbose_header:
            print(
                f'Models:   {list(models.keys())}\n'
                f'Datasets: {list(datasets.keys())}'
            )
        self.name_max_len = {
            'model': max([len(model_name) for model_name in models.keys()]),
            'dataset': max([len(dataset.fullname) for dataset in datasets.values()]),
        }
        df_res = pd.DataFrame()
        for dataset_name, dataset in datasets.items():
            df_res = df_res.append(
                self.run_all_models_single_data(models, dataset, **kwargs)
            )
        self.df_res = df_res
        return df_res

    def print_results(self, df_res=None, group_by='dataset', dataset_header=True):
        """
        Print the evaluation results in a formatted and verbose manner.

        Parameters:
            df_res (pandas DataFrame, optional): A DataFrame containing the evaluation results to be printed.
                                                If None, uses the stored evaluation results. Defaults to None.
            dataset_header (bool, optional): Whether to print dataset headers before each dataset's results. Defaults to True.
        """
        df_res = self.df_res if df_res is None else df_res
        metrics = self.metrics
        models, datasets = df_res.model.unique(), df_res.dataset.unique()
        if group_by == 'dataset':
            for dataset in datasets:
                if dataset_header:
                    print(f'Dataset: {dataset}')
                for model in models:
                    df = df_res[(df_res.model == model) & (df_res.dataset == dataset)]
                    info = f"Model: {model:<{self.name_max_len['model']}s}"
                    for metric in metrics:
                        score, std = (
                            df[f'test_{metric}'].mean(),
                            df[f'test_{metric}'].std(),
                        )
                        info += f" | {metric.upper()} {score:.3f}±{std:.3f}"
                    print(info)
        elif group_by == 'model':
            for model in models:
                print(f'Model: {model}')
                for dataset in datasets:
                    df = df_res[(df_res.model == model) & (df_res.dataset == dataset)]
                    info = f"Data:  {dataset:<{self.name_max_len['dataset']}s}"
                    for metric in metrics:
                        score, std = (
                            df[f'test_{metric}'].mean(),
                            df[f'test_{metric}'].std(),
                        )
                        info += f" | {metric.upper()} {score:.3f}±{std:.3f}"
                    print(info)
        else:
            raise ValueError(
                f'Invalid group_by value: {group_by}, must be "dataset" or "model".'
            )

    def get_single_model_data_results(self, df_res):
        models, datasets = df_res.model.unique(), df_res.dataset.unique()
        assert len(models) == 1, 'df_res contains results for multiple models.'
        assert len(datasets) == 1, 'df_res contains results for multiple datasets.'
        info = ""
        for metric in self.metrics:
            score, std = (
                df_res[f'test_{metric}'].mean(),
                df_res[f'test_{metric}'].std(),
            )
            info += f"{metric.upper()} {score:.3f}±{std:.3f} | "
        # print the advantage group in prediction
        # (dict: key is the group, value is number of runs that the group is advantaged)
        advg_stats = df_res['test_advg'].value_counts().to_dict()
        info += f"AdvG {advg_stats}"
        return info


class Benchmarker:
    def __init__(
        self,
        base_models,
        baselines,
        datasets,
        random_state,
        metrics=METRICS['all'],
        dummy_strategy='prior',
    ) -> None:
        """
        Initialize the Benchmarker object.

        Parameters:
            base_models (dict): A dictionary containing the base models.
                                Key: Model name (str).
                                Value: Base model instance.
            baselines (dict): A dictionary containing the baseline techniques to be evaluated.
                              Key: Technique name (str).
                              Value: Tuple with the baseline class and keyword arguments.
            datasets (dict): A dictionary containing the datasets to be used for evaluation.
                             Key: Dataset name (str).
                             Value: FairDataset object.
            random_state (int): Seed used by the random number generator for reproducibility.
            dummy_strategy (str, optional): Strategy used by the DummyClassifier. Defaults to 'prior'.
        """

        self.base_models = base_models
        self.baselines = baselines
        self.datasets = datasets
        self.random_state = random_state
        self.metrics = metrics
        self.dummy_strategy = dummy_strategy
        for baseline_name in baselines.keys():
            if '_' in baseline_name:
                raise ValueError(f"Baseline name cannot contain '_': {baseline_name}")
        print(
            f"Initializing Benchmarker with:\n"
            f"Random seed: {random_state}\n"
            f"Base models: {list(base_models.keys())}\n"
            f"Techniques:  {list(baselines.keys())}\n"
            f"Datasets:    {list(datasets.keys())}"
        )
        all_models = {'Dummy': DummyClassifier(strategy=dummy_strategy)}
        for base_clf_name, base_clf in base_models.items():
            all_models[base_clf_name] = base_clf
            for baseline_name, (baseline_class, baseline_kwargs) in baselines.items():
                baseline_model = baseline_class(estimator=base_clf, **baseline_kwargs)
                all_models[f'{base_clf_name}_{baseline_name}'] = baseline_model
        self.all_models = all_models
        self.name_max_len = {
            'model': max([len(name) for name in all_models.keys()]),
            'dataset': max([len(name) for name in datasets.keys()]),
        }
        print(f"# models:    {len(all_models)-1}\n" f"# datasets:  {len(datasets)}\n")

    def run(self, n_runs, group_by='dataset', exception: str = 'print'):
        """
        Run the evaluation for all models and datasets.

        Parameters:
            n_runs (int): Number of runs for each model and dataset combination.
            group_by (str, optional): The grouping method to use for the results.
                                      'dataset' groups the results by dataset.
                                      Defaults to 'dataset'.
            exception (bool, optional): Whether to raise an exception if the model
                                        cannot be fitted on the dataset. Defaults to False.
        """
        self.n_runs = n_runs
        print(f"Running All models ...")
        start_time = time.time()
        if group_by == 'dataset':
            all_res = []
            for dataname, dataset in self.datasets.items():
                data_start_time = time.time()
                print(f"========== Start Running on Dataset: {dataname} ==========")
                trainer = Trainer(random_state=self.random_state)
                trainer.run(
                    models=self.all_models,
                    datasets={dataname: dataset},
                    verbose_header=False,
                    n_runs=n_runs,
                    verbose=True,
                    pbar=True,
                    exception=exception,
                )
                t = time.time() - data_start_time
                print(
                    f"========== Run Time on Dataset {dataname}: {t//60:.0f}m{t%60:.0f}s =========="
                )
                self.print_results_with_relative_gain(trainer.df_res)
                all_res.append(trainer.df_res)
            self.df_res = pd.concat(all_res)
        else:
            raise NotImplementedError
        # print the time consumed in miniutes+seconds
        t = time.time() - start_time
        print(f"========== Total Run Time: {t//60:.0f}m{t%60:.0f}s ==========")

    def print_results_with_relative_gain(self, df_res=None):
        """
        Print the evaluation results with relative gains for each model and dataset.

        Parameters:
            df_res (pd.DataFrame, optional): DataFrame containing the evaluation results.
                                             If not provided, the internal df_res will be used.
        """
        df_res = self.df_res if df_res is None else df_res
        models, datasets = df_res.model.unique(), df_res.dataset.unique()

        for dataname in datasets:
            print(f"Results on Data: {dataname}")

            # dummy_scores: dict for computing the relative ACC loss/gain w.r.t. the base model
            df = df_res[(df_res.model == 'Dummy') & (df_res.dataset == dataname)]
            info = f"Model: {'Dummy':<{self.name_max_len['model']}s}"
            dummy_scores = {}
            for metric in self.metrics:
                score, std = df[f'test_{metric}'].mean(), df[f'test_{metric}'].std()
                info += f" | {metric.upper()} {score:.3f}±{std:.3f}           "
                dummy_scores[metric] = score
            print(info)

            for base_model_name in self.base_models.keys():
                df = df_res[
                    (df_res.model == base_model_name) & (df_res.dataset == dataname)
                ]
                info = f"Model: {base_model_name:<{self.name_max_len['model']}s}"
                # base_scores: dict that stores the performance scores of the base model
                base_scores = {}
                for metric in self.metrics:
                    score, std = df[f'test_{metric}'].mean(), df[f'test_{metric}'].std()
                    info += f" | {metric.upper()} {score:.3f}±{std:.3f}           "
                    base_scores[metric] = score
                print(info)
                for baseline in self.baselines.keys():
                    model_name = f'{base_model_name}_{baseline}'
                    df = df_res[
                        (df_res.model == model_name) & (df_res.dataset == dataname)
                    ]
                    if len(df) == 0:
                        continue
                    info = f"Model: {model_name:<{self.name_max_len['model']}s}"
                    for metric in self.metrics:
                        score, std = (
                            df[f'test_{metric}'].mean(),
                            df[f'test_{metric}'].std(),
                        )
                        relative_gain = compute_gain(
                            df, metric, base_scores, dummy_scores
                        )
                        gain = f"({relative_gain:+.2%})"
                        info += f" | {metric.upper()} {score:.3f}±{std:.3f} {gain:<10s}"
                    # fairness utility relative gain
                    furg_score = compute_fairness_utlity_metric(
                        df, self.metrics, dummy_scores, base_scores, how='relative_gain'
                    )
                    furg = f"{furg_score*100:.2f}"
                    futr_score = compute_fairness_utlity_metric(
                        df,
                        self.metrics,
                        dummy_scores,
                        base_scores,
                        how='tradeoff_ratio',
                    )
                    futr = f"{futr_score:.2f}"
                    print(info + f" | FURG {furg:<8s} | FUTR {futr:<8s}")

    def save_results(self, path: str, prefix: str):
        """
        Save the evaluation results to a file.

        Parameters:
            filename (str): The name of the file to save the results to.
        """
        assert type(path) == str, 'path must be a string.'
        assert type(prefix) == str, 'prefix must be a string.'
        file_name = (
            f"{prefix}_"
            f"nBaseModel({len(self.base_models)})_"
            f"nBaseline({len(self.baselines)})_"
            f"nDataset({len(self.datasets)})_"
            f"nRuns({self.n_runs})_"
            f"RandSeed({self.random_state})_"
            f"DummyStrategy({self.dummy_strategy})_"
            f"Time({datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')})"
            f".csv"
        )
        file_path = f"{path}/{file_name}"
        print(f"Saving results to {file_path} ... ", end='')
        self.df_res.to_csv(file_path, index=False)
        print('Done.')


def compute_utility_gain(df, metric: str, base_scores: dict, dummy_scores: dict):
    assert df.model.nunique() == 1
    assert df.dataset.nunique() == 1
    assert metric in METRICS['utility']
    score = df[f'test_{metric}'].mean()
    relative_gain = (score - base_scores[metric]) / (
        base_scores[metric] - dummy_scores[metric]
    )
    return relative_gain


def compute_fairness_gain(df, metric: str, base_scores: dict):
    assert df.model.nunique() == 1
    assert df.dataset.nunique() == 1
    assert metric in METRICS['fairness']
    score = df[f'test_{metric}'].mean()
    relative_gain = (score - base_scores[metric]) / base_scores[metric]
    return relative_gain


def compute_gain(df, metric: str, base_scores: dict, dummy_scores: dict):
    assert df.model.nunique() == 1
    assert df.dataset.nunique() == 1

    if metric in METRICS['utility']:
        return compute_utility_gain(df, metric, base_scores, dummy_scores)
    elif metric in METRICS['fairness']:
        return compute_fairness_gain(df, metric, base_scores)
    else:
        raise ValueError(f'Invalid metric: {metric}')


def compute_fairness_utlity_metric(
    df, metrics: list, dummy_scores: dict, base_scores: dict, how: str
):
    utility_gain = {}
    fairness_gain = {}
    for metric in metrics:
        relevant_gain = compute_gain(df, metric, base_scores, dummy_scores)
        if metric in METRICS['utility']:
            utility_gain[metric] = relevant_gain
        elif metric in METRICS['fairness']:
            fairness_gain[metric] = relevant_gain
        else:
            raise ValueError(f'Invalid metric: {metric}')

    avg_utility_gain = np.mean(list(utility_gain.values()))
    avg_fairness_gain = np.mean(list(fairness_gain.values()))

    if how == 'relative_gain':
        return avg_utility_gain - avg_fairness_gain
    elif how == 'tradeoff_ratio':
        avg_utility_gain = min(avg_utility_gain, -0.01)
        return avg_fairness_gain / avg_utility_gain
    else:
        raise ValueError(
            f'Invalid how value: {how}, must be "relative_gain" or "tradeoff_ratio".'
        )
