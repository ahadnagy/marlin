import unittest

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import marlin


seed = 0
np.random.seed(seed)
torch.random.manual_seed(seed)

DEV = torch.device('cuda:0')


def gen_quant4(m, n, groupsize=-1):
    tile = 16
    maxq = 2 ** 4 - 1
    w = torch.randn((m, n), dtype=torch.half, device=DEV)
    if groupsize != -1:
        w = w.reshape((-1, groupsize, n))
        w = w.permute(1, 0, 2)
        w = w.reshape((groupsize, -1))
    s = torch.max(torch.abs(w), 0, keepdim=True)[0]
    s *= 2 / maxq
    w = torch.round(w / s).int()
    w += (maxq + 1) // 2
    w = torch.clamp(w, 0, maxq)
    ref = (w - (maxq + 1) // 2).half() * s
    if groupsize != -1:
        def reshape(w):
            w = w.reshape((groupsize, -1, n))
            w = w.permute(1, 0, 2)
            w = w.reshape((m, n)).contiguous()
            return w
        ref = reshape(ref)
        w = reshape(w)
    s = s.reshape((-1, n)).contiguous()
    linear = nn.Linear(m, n)
    linear.weight.data = ref.t()
    # Workaround to test some special cases that are forbidden by the API
    layer = marlin.Layer(256, 256, groupsize=groupsize)
    if groupsize == -1:
        groupsize = m
    layer.k = m
    layer.n = n
    layer.groupsize = groupsize
    layer.B = torch.empty((m // 16, n * 16 // 8), dtype=torch.int, device=DEV)
    layer.s = torch.empty((m // groupsize, n), dtype=torch.half, device=DEV)
    layer.pack(linear, s.t())
    q = layer.B
    s = layer.s
    return ref, q, s


def gen_quant4_identity(n, groupsize=-1):
    tile = 16
    maxq = 2 ** 4 - 1
    w = torch.eye(n, dtype=torch.half, device=DEV)
    if groupsize != -1:
        w = w.reshape((-1, groupsize, n))
        w = w.permute(1, 0, 2)
        w = w.reshape((groupsize, -1))
    s = torch.max(torch.abs(w), 0, keepdim=True)[0]
    s *= 2 / maxq
    w = torch.round(w / s).int()
    w += (maxq + 1) // 2
    w = torch.clamp(w, 0, maxq)
    ref = (w - (maxq + 1) // 2).half() * s
    if groupsize != -1:
        def reshape(w):
            w = w.reshape((groupsize, -1, n))
            w = w.permute(1, 0, 2)
            w = w.reshape((n, n)).contiguous()
            return w
        ref = reshape(ref)
        w = reshape(w)
    s = s.reshape((-1, n)).contiguous()
    linear = nn.Linear(n, n)
    linear.weight.data = ref.t()
    # Workaround to test some special cases that are forbidden by the API
    layer = marlin.Layer(256, 256, groupsize=groupsize)
    if groupsize == -1:
        groupsize = n
    layer.k = n
    layer.n = n
    layer.groupsize = groupsize
    layer.B = torch.empty((n // 16, n * 16 // 8), dtype=torch.int, device=DEV)
    layer.s = torch.empty((n // groupsize, n), dtype=torch.half, device=DEV)
    layer.pack(linear, s.t())
    q = layer.B
    s = layer.s
    return ref, q, s

def compare_tensors(tensor1, tensor2):
    if tensor1.shape != tensor2.shape:
        raise ValueError("Tensors must have the same shape for comparison.")

    # Find indices where the tensors differ
    diff_indices = torch.nonzero(tensor1 != tensor2, as_tuple=False)
    if diff_indices.numel() == 0:
        return []
    else:
        return diff_indices.tolist()

class Test(unittest.TestCase):

    def run_problem(self, m, n, k, thread_k, thread_n, groupsize=-1):
        print('% 5d % 6d % 6d % 4d % 4d % 4d' % (m, n, k, thread_k, thread_n, groupsize))
        A = torch.randn((m, k), dtype=torch.half, device=DEV)
        B_ref, B, s = gen_quant4(k, n, groupsize=groupsize)
        C = torch.zeros((m, n), dtype=torch.half, device=DEV)
        C_ref = torch.matmul(A, B_ref)
        workspace = torch.zeros(n // 128 * 16, device=DEV)
        marlin.mul(A, B, C, s, workspace, thread_k, thread_n, -1)
        torch.cuda.synchronize()
        self.assertLess(torch.mean(torch.abs(C - C_ref)) / torch.mean(torch.abs(C_ref)), 0.001)

    def test_tiles(self):
        print()
        for m in [1, 2, 3, 4, 8, 12, 16, 24, 32, 48, 64, 118, 128, 152, 768, 1024]:
            for thread_k, thread_n in [(64, 256), (128, 128)]:
                if m > 16 and thread_k == 128:
                    continue
                self.run_problem(m, 2 * 256, 1024, thread_k, thread_n)

    def test_k_stages_divisibility(self):
        print()
        for k in [3 * 64 + 64 * 4 * 2 + 64 * i for i in range(1, 4)]:
            self.run_problem(16, 2 * 256, k, 64, 256)

    def test_very_few_stages(self):
        print()
        for k in [64, 128, 192]:
            self.run_problem(16, 2 * 256, k, 64, 256)

    def test_llama_shapes(self):
        print()
        return
        MODELS = {
            ' 7B': [
                (4096, 3 * 4096),
                (4096, 4096),
                (4096, 2 * 10752),
                (10752, 4096)
            ],
            '13B': [
                (5120, 3 * 5120),
                (5120, 5120),
                (5120, 2 * 13568),
                (13568, 5120)
            ],
            '33B': [
                (6656, 3 * 6656),
                (6656, 6656),
                (6656, 2 * 17664),
                (17664, 6656)
            ],
            '70B': [
                (8192, 3 * 8192),
                (8192, 8192),
                (8192, 2 * 21760),
                (21760, 8192)
            ]
        }
        for _, layers in MODELS.items():
            for layer in layers:
                for thread_k, thread_n in [(-1, -1)]:
                    for batch in [1, 16]:
                        self.run_problem(batch, layer[1], layer[0], thread_k, thread_n)

    def test_errors(self):
        print()
        m, n, k = 16, 256, 64
        A = torch.randn((m, k), dtype=torch.half, device=DEV)
        B_ref, B, s = gen_quant4(k, n)
        C = torch.zeros((m, n), dtype=torch.half, device=DEV)
        workspace = torch.zeros(n // 128, device=DEV)
        err = False
        try:
            marlin.mul(A, B, C, s, workspace, 128, 128, -1)
        except:
            err = True 
        self.assertTrue(err)
        err = False
        try:
            marlin.mul(A, B, C, s, workspace, 256, 256, -1)
        except:
            err = True 
        self.assertTrue(err)
        s = torch.zeros((2, n), dtype=torch.half, device=DEV)
        err = False
        try:
            marlin.mul(A, B, C, s, workspace, 256, 256, -1)
        except:
            err = True 
        self.assertTrue(err)

    def test_groups(self):
        print()
        for m in [64]:
            for groupsize in [128]:
                for n, k in [(256, 1024), (512, 2048), (512 * 128, 2048)]:
                    for thread_shape in [(-1, -1), (-1, -1)]:
                        self.run_problem(m, n, k, *thread_shape, groupsize)

    def test_race_condition_failure(self):
        dtype = torch.float16
        group_size = 128
        n = 32768
        #n = 256
        k = 32

        A = torch.ones((k, n), dtype=dtype, device=torch.device("cuda"))
        _, B, s = gen_quant4_identity(n, groupsize=group_size)
        C1 = torch.zeros((k, n), dtype=torch.half, device=DEV)
        C2 = torch.zeros((k, n), dtype=torch.half, device=DEV)
        workspace = torch.zeros(n // 128 * 16, device=DEV)
        marlin.mul(A, B, C1, s, workspace, -1, -1, -1)
        torch.cuda.synchronize()
        marlin.mul(A, B, C2, s, workspace, -1, -1, -1)
        torch.cuda.synchronize()
        diff = C1 - C2
        print(f"Differing indices: {len(compare_tensors(C1, C2))}")
        np.savetxt('diff.out', diff.cpu().numpy(), fmt="%f", delimiter=',')
        assert torch.allclose(C1, C2, atol=1e-2), "Output matrices match"


    def testfunc(self, k, n):
        dtype = torch.float16
        group_size = 128

        A = torch.ones((k, n), dtype=dtype, device=torch.device("cuda"))
        _, B, s = gen_quant4_identity(n, groupsize=group_size)
        C1 = torch.zeros((k, n), dtype=torch.half, device=DEV)
        C2 = torch.zeros((k, n), dtype=torch.half, device=DEV)
        workspace = torch.zeros(n // 128 * 16, device=DEV)
        marlin.mul(A, B, C1, s, workspace, -1, -1, -1)
        torch.cuda.synchronize()
        marlin.mul(A, B, C2, s, workspace, -1, -1, -1)
        torch.cuda.synchronize()
        diff = C1 - C2
        return len(compare_tensors(C1, C2)) == 0

    def test_parameter_space(self):
        # Generate logarithmic points in base 2 for k and n
        k_values = [2 ** i for i in range(5, 8)]  # 2^5 (32) to 2^14 (16384)
        n_values = [2 ** i for i in range(8, 16)]

        # Containers for successes and failures
        successes = []
        failures = []

        # Test the function on the 2D plane
        for k in k_values:
            for n in n_values:
                if self.testfunc(k, n):
                    successes.append((k, n))
                else:
                    failures.append((k, n))

        # Separate successes and failures into x and y coordinates for plotting
        successes_x, successes_y = zip(*successes) if successes else ([], [])
        failures_x, failures_y = zip(*failures) if failures else ([], [])

        # Plot the results
        plt.figure(figsize=(10, 6))
        plt.scatter(successes_x, successes_y, color='green', label='Success', alpha=0.7)
        plt.scatter(failures_x, failures_y, color='red', label='Failure', alpha=0.7)
        plt.xscale('log', base=2)
        plt.yscale('log', base=2)
        plt.xlabel('k values (log)')
        plt.ylabel('n values (log)')
        plt.title('Kernel success/failure for different k, n sizes')
        plt.legend()
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.savefig("plot.png")



if __name__ == '__main__':
    unittest.main()
