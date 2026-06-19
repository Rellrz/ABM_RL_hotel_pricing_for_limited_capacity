#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 对外统一导出，便于后续逐步从 `from configs.config import ...`
# 迁移到 `from configs import ...`。
from .config import *  # noqa: F401,F403

