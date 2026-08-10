"""工作记忆槽：每序列固定尺寸 scratchpad，存推理环中间误差痕迹，供"主动采样看一眼"读取。"""
import torch


class Scratchpad:
    def __init__(self, cap=16):
        self.cap = cap
        self.clear()

    def clear(self):
        self.slots = []

    def write(self, vec):
        self.slots.append(vec)
        if len(self.slots) > self.cap:
            self.slots.pop(0)

    def read(self):
        return self.slots[-1] if self.slots else None

    def __len__(self):
        return len(self.slots)
