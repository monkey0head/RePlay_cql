import os
import warnings
from argparse import ArgumentParser
from math import floor

import numpy as np
import pandas as pd
import psutil
import torch
from d3rlpy.torch_utility import eval_api, freeze, unfreeze
from optuna.exceptions import ExperimentalWarning

from replay.constants import REC_SCHEMA
from replay.session_handler import State, get_spark_session


print('Load test.py')


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=ExperimentalWarning)
    torch.set_num_threads(1)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

    use_gpu = torch.cuda.is_available()

    parser = ArgumentParser()
    parser.add_argument('--part', dest='partitions', type=float, required=False, default=0.8)
    parser.add_argument('--mem', dest='memory', type=float, required=False, default=0.7)
    parser.add_argument(
        '--scale', dest='action_randomization_scale', type=float, required=False, default=0.01
    )

    args = parser.parse_args()

    spark = get_spark_session(
        spark_memory=floor(psutil.virtual_memory().total / 1024 ** 3 * args.memory),
        shuffle_partitions=floor(os.cpu_count() * args.partitions),
    )
    spark = State(session=spark).session
    spark.sparkContext.setLogLevel('ERROR')

    K = 10
    train = spark.read.parquet('./train.parquet')
    test = spark.read.parquet('./test.parquet')
    test = test.drop('timestamp')
    test_users = test.select('user_idx').distinct()
    available_items = test.toPandas()["item_idx"].values
    user_item_pairs = pd.DataFrame({
        'user_idx': np.repeat(1, len(available_items)),
        'item_idx': available_items
    })
    inp = torch.from_numpy(user_item_pairs.to_numpy()).float().cpu()

    # from replay.models.cql import CQL
    # model = CQL(
    #     use_gpu=use_gpu, top_k=K, n_epochs=1,
    #     action_randomization_scale=args.action_randomization_scale,
    #     batch_size=2048
    # )
    # model.fit(log=train)
    # torch.save(model.model._impl, './model.pt')
    # save_policy(model.model._impl, './policy.pt', batch_size=10) #len(available_items))

    # model = torch.load('./model.pt')
    # print(model.predict_best_action(inp))
    model = torch.jit.load('./policy.pt', map_location=torch.device('cpu'))
    t = min(inp.size(dim=0), 10)
    print('size=', inp.size(dim=0), 't=', t)
    with torch.no_grad():
        print(
            model.forward(inp[:, :]).numpy()
        )

    def grouped_map(log_slice: pd.DataFrame) -> pd.DataFrame:
        return _rate_user_items(
            model=None,
            user_idx=log_slice["user_idx"][0],
            items=available_items,
        )[["user_idx", "item_idx", "relevance"]]

    res = test.groupby("user_idx").applyInPandas(grouped_map, REC_SCHEMA)
    res = res.cache()
    res.count()

    # pred = model.predict(log=train, k=K, users=test_users)
    # pred.cache()
    # pred.count()

    print(res.show(10))
    print('Experiment: OK')


def _rate_user_items(
    model,
    user_idx: int,
    items: np.ndarray,
) -> pd.DataFrame:
    user_item_pairs = pd.DataFrame({
        'user_idx': np.repeat(user_idx, len(items)),
        'item_idx': items
    })
    print('===>')
    # # with io.BytesIO(model) as buffer:
    # #     model = torch.load(buffer, map_location=torch.device('cpu'))
    #
    if model is None:
        model = torch.jit.load('./policy.pt', map_location=torch.device('cpu'))

    # res = model.predict_best_action(user_item_pairs.to_numpy())
    res = model.forward(torch.from_numpy(user_item_pairs.to_numpy()).float().cpu())

    # res = model.forward(inp).numpy()
    # print(type(res))
    # print(res)
    # print(
    #     model.forward(
    #         torch.from_numpy(
    #             user_item_pairs.to_numpy()[:2]
    #         ).float()
    #     ).detach().cpu().numpy()
    # )
    # t = min(inp.size(dim=0), 10)
    # print('size=', inp.size(dim=0), 't=', t)
    # with torch.no_grad():
    #     print(
    #         model.forward(inp[:, :]).numpy()
    #     )
    # res = np.repeat(1., len(items))

    # it doesn't explicitly filter seen items and doesn't return top k items
    # instead, it keeps all predictions as is to be filtered further by base methods

    user_item_pairs['relevance'] = res
    print('<===')
    return user_item_pairs



@eval_api
def save_policy(model, fname: str, batch_size) -> None:
    dummy_x = torch.rand(batch_size, *model.observation_shape, device=model._device)

    # workaround until version 1.6
    freeze(model)

    # dummy function to select best actions
    def _func(x: torch.Tensor) -> torch.Tensor:
        if model._scaler:
            x = model._scaler.transform(x)

        action = model._predict_best_action(x)

        if model._action_scaler:
            action = model._action_scaler.reverse_transform(action)

        return action

    traced_script = torch.jit.trace(_func, dummy_x, check_trace=False)
    traced_script.save(fname)

    # workaround until version 1.6
    unfreeze(model)


if __name__ == '__main__':
    main()
