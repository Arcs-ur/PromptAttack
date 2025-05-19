import os
import pickle
import argparse
from tqdm import tqdm
import numpy as np
import concurrent.futures
from Predict import Predict
from datasets import load_dataset
from PromptAttack import PromptAttack

parser = argparse.ArgumentParser(description="Prompt-Based Adversarial Attack")
parser.add_argument("--API_key", type=str, default="", help="OpenAI key")
# parser.add_argument(
#     "--API_base", type=str, default="https://api.openai.com/v1", help="LLM server"
# )
parser.add_argument(
    "--provider", type=str, default="OpenAI", help="LLM server"
)
#模型的服务商
parser.add_argument(
    "--version", type=str, default="gpt-3.5-turbo", help="version of GPT"
)
#具体的模型版本
parser.add_argument(
    "--dataset",
    type=str,
    default="SST-2",
    help="dataset [SST-2, QQP, MNLI-m, MNLI-mm, RTE, QNLI]",
)
#所使用的数据集，这里数据集的具体形式和格式需要查一下【todo】
parser.add_argument(
    "--tau_1",
    type=float,
    default=0.15,
    help="threshold of word modification rate [0,1]",
)
#这里后面具体是干什么的需要查一下论文里面
parser.add_argument(
    "--tau_2", type=float, default=0.92, help="threshold of BERTScore [0,1]"
)
#这里后面具体是干什么的需要查一下论文里面
parser.add_argument(
    "--pertub_type",
    type=int,
    default=0,
    help="the index of perturbation index {0,1,2,3,4,5,6,7,8}",
)
#施加扰动的类型
parser.add_argument(
    "--t_a",
    type=int,
    default=0,
    help="the index of the type of the sentence to be perturbed",
)
#这和上面有什么区别？
parser.add_argument(
    "--few_shot", action="store_true", help="whether to use few-shot strategy"
)
#采用上下文学习的策略(few-shot)
parser.add_argument(
    "--ensemble", action="store_true", help="whether to use ensemble strategy"
)
#采用集成侧路essemble的策略，需要发很多遍
parser.add_argument("--batch_size", type=int, default=8, help="batch_size of data")
#每次处理的数据数量
parser.add_argument(
    "--attack_log_file",
    type=str,
    default="attack.db",
    help="file to save LLM attack result",
)
#用来保存攻击后得到的攻击后样本
parser.add_argument(
    "--check_log_file",
    type=str,
    default="check.db",
    help="file to save LLM check result",
)
#用来保存用攻击样本攻击模型的结果
args = parser.parse_args()

args.dataset = args.dataset.lower()
if args.dataset == "sst-2":
    args.dataset = "sst2"


def get_dataset(dataset):
    if "mnli" in dataset:
        assert "mnli-m" in dataset
        if dataset == "mnli-m":
            dataset_ = load_dataset("glue", "mnli", split="validation_matched")
        if dataset == "mnli-mm":
            dataset_ = load_dataset("glue", "mnli", split="validation_mismatched")
    else:
        dataset_ = load_dataset("glue", dataset, split="validation")

    test_loader = [
        [
            [
                [key, value]
                for (key, value) in dataset_[i].items()
                if key != "label" and key != "idx"
            ],
            dataset_[i]["label"],
        ]
        for i in range(dataset_.num_rows)
    ]
    test_loader = [
        test_loader[i : i + args.batch_size]
        for i in range(0, len(test_loader), args.batch_size)
    ]
    label_list = dataset_.features["label"]._int2str
    return test_loader, label_list


def get_td(td_index, dataset):
    with open(os.path.join("info", "{}_info.pkl".format(dataset)), "rb") as f:
        td_fsexample_info = pickle.load(f)
    td = td_fsexample_info["td"][td_index]
    return td


def get_accuracy(pred, label):
    assert len(pred) == len(label)
    correct = [i == j for (i, j) in zip(pred, label)]
    return sum(correct) / len(label)


def get_ASR(pred, adv_pred, label):
    assert len(pred) == len(label) and len(pred) == len(adv_pred)
    correct = [i == j for (i, j) in zip(pred, label)]
    adv_wrong = [
        (pred[i] == label[i] and adv_pred[i] != label[i]) for i in range(len(label))
    ]
    return sum(adv_wrong) / sum(correct)


test_loader, label_list = get_dataset(args.dataset)
predictor = Predict(
    log_file=args.check_log_file,
    API_key=args.API_key,
    # API_base=args.API_base,
    label_list=label_list,
    version=args.version,
    provider=args.provider,
)
adv_generator = PromptAttack(
    log_file=args.attack_log_file,
    API_key=args.API_key,
    # API_base=args.API_base,
    dataset=args.dataset,
    label_list=label_list,
    predictor=predictor,
    version=args.version,
    provider=args.provider,
)


def get_pred(loader, td):
    results = []
    for batch in loader:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            result = list(
                executor.map(predictor, [x for x, y in batch], [td] * len(batch))
            )
        results += result

    return results


natural_acc = []
robust_acc = []
ASR = []

label = [y for batch in test_loader for x, y in batch]

# We used 12 different task descriptions for each task. You can see these task descriptions in pickle files in ''info'' folder.
for td_index in tqdm(range(12), desc="Outer Loop"):
    task_description = get_td(td_index, args.dataset)
    pred = get_pred(test_loader, task_description)
    print(pred)
    # print("_" * 10 + "normal" + "_" * 10)
    # print(pred)
    adv_loader = []
    for batch in tqdm(test_loader, desc="Inner Loop", leave=False):
        batch_x = [x for (x, y) in batch]
        batch_y = [y for (x, y) in batch]
        batch_adv_x = adv_generator.batch_attack(
            batch_x,
            batch_y,
            args.pertub_type,
            args.t_a,
            args.tau_1,
            args.tau_2,
            few_shot=args.few_shot,
            ensemble=args.ensemble,
            task_description=task_description,
        )

        adv_loader.append([[adv_x, y] for (adv_x, y) in zip(batch_adv_x, batch_y)])

    adv_pred = get_pred(adv_loader, task_description)
    # print("_" * 10 + "adversarial" + "_" * 10)
    # print(adv_pred)
    natural_acc.append(get_accuracy(pred, label))
    robust_acc.append(get_accuracy(adv_pred, label))
    ASR.append(get_ASR(pred, adv_pred, label))

    print(
        "Task Description Index: {} \t Natural Accuracy: {} Robust Accuracy: {} \t Attack Success Rate: {}".format(
            td_index, natural_acc[td_index], robust_acc[td_index], ASR[td_index]
        )
    )

print(
    "Average Natural Accuracy: {} Average Robust Accuracy: {} \t Average Attack Success Rate: {}".format(
        np.mean(natural_acc), np.mean(robust_acc), np.mean(ASR)
    )
)
