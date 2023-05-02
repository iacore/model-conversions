import sys
from torch import load as torch_load
from safetensors.torch import save_file

try:
    file = sys.argv[1]
    fout = sys.argv[2]
except IndexError:
    print("Usage: THIS_SCRIPT abc.pth abc.safetensors")
    exit(1)

weights = torch_load(file, "cpu")

for k in weights.keys():
    print(k, weights[k].size(), weights[k].dtype)
    weights[k] = weights[k].float() # to float32

save_file(weights, fout)
