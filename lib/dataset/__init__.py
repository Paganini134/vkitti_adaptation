# Copyright 2021 Garena Online Private Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    from lib.dataset.panoptic import Panoptic as panoptic
except Exception as exc:
    print(f"optional dataset import skipped: panoptic: {exc}")
try:
    from lib.dataset.h36m import H36M as h36m
except Exception as exc:
    print(f"optional dataset import skipped: h36m: {exc}")
try:
    from lib.dataset.campus_seq1 import Campus as campus_seq1
except Exception as exc:
    print(f"optional dataset import skipped: campus_seq1: {exc}")
try:
    from lib.dataset.shelf import Shelf as shelf
except Exception as exc:
    print(f"optional dataset import skipped: shelf: {exc}")
from lib.dataset.vkitti_keypoints_raft import VKITTIKeypointsRAFT as vkitti_keypoints_raft
