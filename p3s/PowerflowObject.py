# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC, abstractmethod

from scipy.sparse import csr_matrix


class PowerflowObject(ABC):
    def __init__(self):
        self._offset: int = 0

    @abstractmethod
    def create_J(self, V) -> csr_matrix:
        pass

    @abstractmethod
    def evaluate_Results(self, dx, voltage):
        pass

    @abstractmethod
    def evaluate_Fx(self, Sbus, V):
        pass

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, offset: int):
        self._offset = offset
