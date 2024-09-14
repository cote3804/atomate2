"""JDFTx out file parsing class.

A class to read and process a JDFTx out file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Callable

from monty.io import zopen

from atomate2.jdftx.io.jdftxoutfileslice import JDFTXOutfileSlice
from atomate2.jdftx.io.jdftxoutfileslice_helpers import get_start_lines


def check_file_exists(func: Callable) -> Any:
    """Check if file exists.

    Check if file exists (and continue normally) or raise an exception if
    it does not.
    """

    @wraps(func)
    def wrapper(filename: str) -> Any:
        filepath = Path(filename)
        if not filepath.is_file():
            raise OSError(f"'{filename}' file doesn't exist!")
        return func(filename)

    return wrapper


@check_file_exists
def read_file(file_name: str) -> list[str]:
    """
    Read file into a list of str.

    Parameters
    ----------
    filename: Path or str
        name of file to read

    Returns
    -------
    text: list[str]
        list of strings from file
    """
    with zopen(file_name, "r") as f:
        text = f.readlines()
    f.close()
    return text


def read_outfile_slices(file_name: str) -> list[list[str]]:
    """
    Read slice of out file into a list of str.

    Parameters
    ----------
    filename: Path or str
        name of file to read
    out_slice_idx: int
        index of slice to read from file

    Returns
    -------
    texts: list[list[str]]
        list of out file slices (individual calls of JDFTx)
    """
    _text = read_file(file_name)
    start_lines = get_start_lines(_text, add_end=True)
    texts = []
    for i in range(len(start_lines) - 1):
        text = _text[start_lines[i] : start_lines[i + 1]]
        texts.append(text)
    return texts


@dataclass
class JDFTXOutfile:
    """JDFTx out file parsing class.

    A class to read and process a JDFTx out file.
    """

    slices: list[JDFTXOutfileSlice] = field(default_factory=list)

    @classmethod
    def from_file(cls, file_path: str) -> JDFTXOutfile:
        """Return JDFTXOutfile object.

        Create a JDFTXOutfile object from a JDFTx out file.

        Parameters
        ----------
        file_path: str
            The path to the JDFTx out file

        Returns
        -------
        instance: JDFTXOutfile
            The JDFTXOutfile object
        """
        texts = read_outfile_slices(file_path)
        slices = [JDFTXOutfileSlice.from_out_slice(text) for text in texts]
        instance = cls()
        instance.slices = slices
        return instance

    def __getitem__(self, key: int | str) -> JDFTXOutfileSlice | Any:
        """Return item.

        Return the value of an item.

        Parameters
        ----------
        key: int | str
            The key of the item

        Returns
        -------
        val
            The value of the item
        """
        val = None
        if type(key) is int:
            val = self.slices[key]
        elif type(key) is str:
            val = getattr(self, key)
        else:
            raise TypeError(f"Invalid key type: {type(key)}")
        return val

    def __len__(self) -> int:
        """Return length of JDFTXOutfile object.

        Returns the number of JDFTx calls in the
        JDFTXOutfile object.

        Returns
        -------
        length: int
            The number of geometric optimization steps in the JDFTXOutfile
            object
        """
        return len(self.slices)
