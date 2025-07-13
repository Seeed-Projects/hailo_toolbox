from typing import Tuple, Optional, AnyStr, List, Union, Dict, Any
from multiprocessing import shared_memory
from multiprocessing import resource_tracker
import numpy as np
import threading


class UntrackedSharedMemory(shared_memory.SharedMemory):
    # https://github.com/python/cpython/issues/82300#issuecomment-2169035092

    __lock = threading.Lock()

    def __init__(
        self,
        name: Optional[str] = None,
        create: bool = False,
        size: int = 0,
        *,
        track: bool = False,
    ) -> None:
        self._track = track

        # if tracking, normal init will suffice
        if track:
            return super().__init__(name=name, create=create, size=size)

        # lock so that other threads don't attempt to use the
        # register function during this time
        with self.__lock:
            # temporarily disable registration during initialization
            orig_register = resource_tracker.register
            resource_tracker.register = self.__tmp_register

            # initialize; ensure original register function is
            # re-instated
            try:
                super().__init__(name=name, create=create, size=size)
            finally:
                resource_tracker.register = orig_register

    @staticmethod
    def __tmp_register(*args, **kwargs) -> None:
        return

    def unlink(self) -> None:
        if shared_memory._USE_POSIX and self._name:
            shared_memory._posixshmem.shm_unlink(self._name)
            if self._track:
                resource_tracker.unregister(self._name, "shared_memory")


class ShareMemoryManager:
    def __init__(self, max_size: Optional[int] = 1024 * 1024 * 3 * 10):
        self.max_size = max_size
        self.shm_dict: Dict[str, UntrackedSharedMemory] = {}

    def create_shm(self, name: str):
        shm = UntrackedSharedMemory(name=name, create=True, size=self.max_size)
        self.shm_dict[name] = shm
        return shm

    def write_dict(self, data: Dict[str, np.ndarray]):
        shm_info = []
        for key, value in data.items():
            key = key.replace("/", "_")
            if isinstance(value, np.ndarray):
                shm_info.append(self.write(value, key))
            elif isinstance(value, list):
                value = np.array(value, dtype=object)
                shm_info.append(self.write(value, key))
        return shm_info

    def read_dict(self, shm_info: List[Dict[str, Any]]):
        data = {}
        for info in shm_info:
            if isinstance(info, dict):
                data[info["name"]] = self.read(**info)
        return data

    def write(self, data: np.ndarray, name: str, size: Optional[int] = None):
        shape = data.shape
        size = data.nbytes
        dtype = data.dtype.name
        # try:
        if name not in self.shm_dict:
            try:
                shm = UntrackedSharedMemory(name=name, create=True, size=self.max_size)
            except FileExistsError:
                shm = UntrackedSharedMemory(name=name, size=self.max_size)
            self.shm_dict[name] = shm
        else:
            shm = self.shm_dict[name]
        shm.buf[:size] = data.tobytes()

        return {
            "name": name,
            "size": size,
            "shape": shape,
            "dtype": dtype,
        }
        # except Exception as e:
        #     print(f"write {name} failed: {e}")
        #     return False

    def read(
        self, name: str, size: int, dtype: Union[np.dtype, str], shape: Tuple[int, ...]
    ):
        if name not in self.shm_dict:
            try:
                shm = UntrackedSharedMemory(name=name, create=True, size=self.max_size)
            except FileExistsError:
                shm = UntrackedSharedMemory(name=name, size=self.max_size)
            self.shm_dict[name] = shm

        else:
            shm = self.shm_dict[name]
        data = np.frombuffer(shm.buf[:size], dtype=dtype).reshape(shape)

        return data

    def cleanup_single(self, name: str):
        """
        Clean up a single shared memory block by name.

        Args:
            name: Name of the shared memory block to clean up
        """
        if name in self.shm_dict:
            try:
                shm = self.shm_dict[name]
                shm.close()
                # Don't unlink here as it might be used by other processes
                del self.shm_dict[name]
            except Exception as e:
                print(f"Error cleaning up shared memory {name}: {e}")

    def cleanup(self):
        """Clean up all shared memory blocks."""
        for name, shm in list(self.shm_dict.items()):
            try:
                shm.close()
                shm.unlink()
            except Exception as e:
                print(f"Error cleaning up shared memory {name}: {e}")
        self.shm_dict.clear()

    def __del__(self):
        self.cleanup()
