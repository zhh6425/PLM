from enum import Enum

import numpy as np
import torch
import torch.distributed as dist

IGNORE_INDEX = -100
POINT_TOKEN_INDEX = -200
DEFAULT_POINT_TOKEN = "<point>"
DEFAULT_POINT_PATCH_TOKEN = "<point_patch>"
DEFAULT_PC_START_TOKEN = "<point_start>"
DEFAULT_PC_END_TOKEN = "<point_end>"

SHORT_QUESTION_LIST = [
    DEFAULT_POINT_TOKEN + "\n" + "Can you segment the {class_name} category in this point cloud?", 
    DEFAULT_POINT_TOKEN + "\n" + "Please segment the {class_name} category in this point cloud.",
    DEFAULT_POINT_TOKEN + "\n" + "What is {class_name} category in this point cloud? Please respond with segmentation mask.",
    DEFAULT_POINT_TOKEN + "\n" + "What is {class_name} category in this point cloud? Please output segmentation mask.",
]

REFER_QUESTION_LIST = [
    DEFAULT_POINT_TOKEN + "\n" + "With a description: {description} Please respond with segmentation mask.",
    DEFAULT_POINT_TOKEN + "\n" + "Giving the referring sentence: {description} Please output segmentation mask.",
    DEFAULT_POINT_TOKEN + "\n" + "Where is the object: {description} Can you segment the described object?",
    DEFAULT_POINT_TOKEN + "\n" + "In this scene: {description} Please segment the described object.",
]

ANSWER_LIST = [
    "There is no mask [SEG].",
    "It is [SEG].",
    "Sure, segmented [SEG].",
    "Sure, it is [SEG].",
    "I have segmented the required [SEG].",
]

SCENE_QUESTION_LIST = [
    DEFAULT_POINT_TOKEN + "\n" + "Can you provide a brief description of this indoor scene?",
    DEFAULT_POINT_TOKEN + "\n" + "How would you summarize the overall layout of this space?",
    DEFAULT_POINT_TOKEN + "\n" + "Could you give an overview of the room's purpose and design?",
    DEFAULT_POINT_TOKEN + "\n" + "How is the furniture arranged in this space?",
    DEFAULT_POINT_TOKEN + "\n" + "Can you describe any visible textures or materials used in this room?",
    DEFAULT_POINT_TOKEN + "\n" + "How is the space utilized in this room?",
    DEFAULT_POINT_TOKEN + "\n" + "How would you describe the flow of the space?",
    DEFAULT_POINT_TOKEN + "\n" + "What elements of design are prominent in this room?",
    DEFAULT_POINT_TOKEN + "\n" + "Can you give a general overview of this indoor scene?",
    DEFAULT_POINT_TOKEN + "\n" + "How would you describe the essence of this room?",
    DEFAULT_POINT_TOKEN + "\n" + "What is the primary impression this room gives?",
    DEFAULT_POINT_TOKEN + "\n" + "Can you summarize the character of this indoor setting?",
    DEFAULT_POINT_TOKEN + "\n" + "What overall feeling does this room evoke?",
    DEFAULT_POINT_TOKEN + "\n" + "How does this space represent its intended function?",
    DEFAULT_POINT_TOKEN + "\n" + "Can you encapsulate the vibe of this room in a few sentences?",
    DEFAULT_POINT_TOKEN + "\n" + "What general themes are conveyed in this room's design?",
    DEFAULT_POINT_TOKEN + "\n" + "How would you characterize the setting of this room?",
    DEFAULT_POINT_TOKEN + "\n" + "Could you provide a broad description of this room's atmosphere?",
    DEFAULT_POINT_TOKEN + "\n" + "What is the holistic approach to the design of this room?"
]

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(self.sum, np.ndarray):
            total = torch.tensor(
                self.sum.tolist()
                + [
                    self.count,
                ],
                dtype=torch.float32,
                device=device,
            )
        else:
            total = torch.tensor(
                [self.sum, self.count], dtype=torch.float32, device=device
            )

        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        if total.shape[0] > 2:
            self.sum, self.count = total[:-1].cpu().numpy(), total[-1].cpu().item()
        else:
            self.sum, self.count = total.tolist()
        self.avg = self.sum / (self.count + 1e-5)

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K - 1)
    area_output = torch.histc(output, bins=K, min=0, max=K - 1)
    area_target = torch.histc(target, bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


# def dict_to_cuda(input_dict):
#     for k, v in input_dict.items():
#         if isinstance(v, torch.Tensor):
#             input_dict[k] = v.cuda(non_blocking=True)
#         elif isinstance(v, list):
#             if len(v) > 0:
#                 if isinstance(v[0], torch.Tensor):
#                     input_dict[k] = [ele.cuda(non_blocking=True) for ele in v]
#                 elif isinstance(v[0], list) and len(v[0] > 0) and isinstance(v[0][0], torch.Tensor):
#                     input_dict[k] = [
#                         [ele.cuda(non_blocking=True) for ele in sublist] for sublist in v
#                     ]
#     return input_dict


def to_cuda_recursive(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cuda(non_blocking=True)
    elif isinstance(obj, list):
        return [to_cuda_recursive(ele) for ele in obj]
    else:
        return obj

def dict_to_cuda(input_dict):
    return {k: to_cuda_recursive(v) for k, v in input_dict.items()}


